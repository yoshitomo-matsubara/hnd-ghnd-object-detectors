import argparse
import datetime
import math
import sys
import time

import torch
import torch.utils.data
from torch import nn

from models import get_model, load_ckpt, save_ckpt
from models.ext.backbone import check_if_valid_target
from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util
from utils import data_util, main_util, misc_util


def get_argparser():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', required=True, help='yaml config file')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('--json', help='dictionary to overwrite config')
    argparser.add_argument('-train', action='store_true', help='train a model')
    # distributed training parameters
    argparser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    argparser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return argparser


def convert_target2ext_targets(targets, device):
    ext_targets = [1 if check_if_valid_target(target) else 0 for target in targets]
    return torch.FloatTensor(ext_targets).unsqueeze(1).to(device)


def train_model(model, optimizer, data_loader, device, epoch, log_freq):
    model.train()
    model.ext_training = True
    metric_logger = misc_util.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc_util.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000.0
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = main_util.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, log_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        ext_logits = model(images, targets)
        ext_targets = convert_target2ext_targets(targets, ext_logits.device)

        ext_cls_loss = nn.functional.binary_cross_entropy_with_logits(ext_logits, ext_targets)
        loss_dict = {'loss_ext_classifier': ext_cls_loss}
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = misc_util.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])


def evaluate(model, data_loader, device, split_name='Validation'):
    correct_count = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        ext_logits = model(images, targets)
        ext_targets = convert_target2ext_targets(targets, ext_logits.device)
        correct_count += ext_logits.eq(ext_targets).sum().item()

    accuracy = correct_count / len(data_loader.dataset)
    print('{} accuracy: {}'.format(split_name, accuracy))
    return accuracy


def train(model, train_sampler, train_data_loader, val_data_loader, device, distributed, config, args, ckpt_file_path):
    train_config = config['train']
    optim_config = train_config['optimizer']
    ext_classifier = model.backbone.body.ext_classifier
    optimizer = func_util.get_optimizer(ext_classifier, optim_config['type'], optim_config['params'])
    scheduler_config = train_config['scheduler']
    lr_scheduler = func_util.get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
    if file_util.check_if_exists(ckpt_file_path):
        load_ckpt(ckpt_file_path, model=ext_classifier, optimizer=optimizer, lr_scheduler=lr_scheduler)

    best_val_acc = 0.0
    num_epochs = train_config['num_epochs']
    log_freq = train_config['log_freq']
    for epoch in range(1, num_epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)

        train_model(model, optimizer, train_data_loader, device, epoch, log_freq)
        lr_scheduler.step()

        # evaluate after every epoch
        val_acc = evaluate(model, val_data_loader, device, split_name='Validation')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_ckpt(ext_classifier, optimizer, lr_scheduler, config, args, ckpt_file_path)


def main(args):
    distributed, device_ids = main_util.init_distributed_mode(args.world_size, args.dist_url)
    config = yaml_util.load_yaml_file(args.config)
    if args.json is not None:
        main_util.overwrite_config(config, args.json)

    device = torch.device(args.device)
    print(args)

    print('Loading data')
    train_config = config['train']
    train_sampler, train_data_loader, val_data_loader, test_data_loader =\
        data_util.get_coco_data_loaders(config['dataset'], train_config['batch_size'], distributed)

    print('Creating model')
    model_config = config['model']
    model = get_model(model_config, device, strict=False)
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    if args.train:
        print('Start training')
        start_time = time.time()
        train(model, train_sampler, train_data_loader, val_data_loader, device, distributed,
              config, args, model_config['params']['ext_config']['ckpt'])
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    evaluate(model, test_data_loader, device=device)


if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
