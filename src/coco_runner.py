r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.
"""
import argparse
import datetime
import math
import sys
import time

import torch
import torch.utils.data
from torch import nn

from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util
from utils import data_util, main_util, misc_util, model_util
from utils.coco_eval_util import CocoEvaluator
from utils.coco_util import get_coco_api_from_dataset


def get_argparser():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', required=True, help='yaml config file')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('-train', action='store_true', help='train a model')
    # distributed training parameters
    argparser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    argparser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    return argparser


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device('cpu')
    model.eval()
    metric_logger = misc_util.MetricLogger(delimiter='  ')
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = model_util.get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def train_model(model, optimizer, data_loader, device, epoch, log_freq):
    model.train()
    metric_logger = misc_util.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc_util.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000.0
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = misc_util.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, log_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
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


def train(model, train_sampler, train_data_loader, val_data_loader, device, distributed, config, args, ckpt_file_path):
    train_config = config['train']
    optim_config = train_config['optimizer']
    optimizer = func_util.get_optimizer(model, optim_config['type'], optim_config['params'])
    scheduler_config = train_config['scheduler']
    lr_scheduler = func_util.get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
    if file_util.check_if_exists(ckpt_file_path):
        model_util.load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    best_val_map = 0.0
    num_epochs = train_config['num_epochs']
    log_freq = train_config['log_freq']
    for epoch in range(num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        train_model(model, optimizer, train_data_loader, device, epoch, log_freq)
        lr_scheduler.step()

        # evaluate after every epoch
        coco_evaluator = evaluate(model, val_data_loader, device=device)
        # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        val_map = coco_evaluator.coco_eval['bbox'].stats[0]
        if val_map > best_val_map:
            best_val_map = val_map
            model_util.save_ckpt(model, optimizer, lr_scheduler, config, args, ckpt_file_path)


def main(args):
    distributed, device_ids = main_util.init_distributed_mode(args.world_size, args.dist_url)
    config = yaml_util.load_yaml_file(args.config)
    device = torch.device(args.device)
    print(args)

    print('Loading data')
    train_config = config['train']
    train_sampler, train_data_loader, val_data_loader, test_data_loader =\
        data_util.get_coco_data_loaders(config['dataset'], train_config['batch_size'], distributed)

    print('Creating model')
    model_config = config['model']
    model = model_util.get_model(model_config, device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    if args.train:
        print('Start training')
        start_time = time.time()
        train(model, train_sampler, train_data_loader, val_data_loader, device, distributed,
              config, args, model_config['ckpt'])
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    evaluate(model, test_data_loader, device=device)


if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
