import argparse
import datetime
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn import metrics
from torch import nn

from models import get_model, load_ckpt, save_ckpt
from models.ext.backbone import check_if_valid_target
from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util, module_util
from utils import data_util, main_util, misc_util


def get_argparser():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', required=True, help='yaml config file')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('--json', help='dictionary to overwrite config')
    argparser.add_argument('--min_recall', type=float, default=0.9, help='minimum recall to decide a threshold')
    argparser.add_argument('-train', action='store_true', help='train a model')
    # distributed training parameters
    argparser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    argparser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return argparser


def convert_target2ext_targets(targets, device):
    ext_targets = [1 if check_if_valid_target(target) else 0 for target in targets]
    return torch.LongTensor(ext_targets).to(device)


def train_model(model, optimizer, data_loader, device, epoch, log_freq):
    model.train()
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
        ext_targets = convert_target2ext_targets(targets, device)
        ext_cls_loss = nn.functional.cross_entropy(ext_logits, ext_targets)
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


def evaluate(model, data_loader, device, min_recall, split_name='Validation'):
    model.eval()
    correct_count = 0
    pos_correct_count = 0
    pos_count = 0
    prob_list = list()
    label_list = list()
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            ext_logits = model(images, targets)
            ext_targets = convert_target2ext_targets(targets, device)
            prob_list.append(ext_logits[:, 1].cpu().numpy())
            label_list.append(ext_targets.cpu().numpy())
            preds = ext_logits.argmax(dim=1)
            correct_count += preds.eq(ext_targets).sum().item()
            pos_correct_count += preds[ext_targets.nonzero().flatten()].sum().item()
            pos_count += ext_targets.sum().item()

    num_samples = len(data_loader.dataset)
    accuracy = correct_count / num_samples
    recall = pos_correct_count / pos_count
    specificity = (correct_count - pos_correct_count) / (num_samples - pos_count)
    probs = np.concatenate(prob_list)
    labels = np.concatenate(label_list)
    roc_auc = metrics.roc_auc_score(labels, probs)
    print('[{}]'.format(split_name))
    print('\tAccuracy: {:.4f} ({} / {})'.format(accuracy, correct_count, num_samples))
    print('\tRecall: {:.4f} ({} / {})'.format(recall, pos_correct_count, pos_count))
    print('\tSpecificity: {:.4f} ({} / {})'.format(specificity, correct_count - pos_correct_count,
                                                   num_samples - pos_count))
    print('\tROC-AUC: {:.4f}'.format(roc_auc))
    if split_name == 'Test':
        fprs, tprs, thrs = metrics.roc_curve(labels, probs, pos_label=1)
        idx = np.searchsorted(tprs, min_recall)

        data_frame =\
            pd.DataFrame(np.array([thrs[idx:], tprs[idx:], fprs[idx:]]).T, columns=['Threshold', 'TPR (Recall)', 'FPR'])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(data_frame)
    return roc_auc


def train(model, train_sampler, train_data_loader, val_data_loader, device, distributed, config, args, ckpt_file_path):
    train_config = config['train']
    optim_config = train_config['optimizer']
    ext_classifier = model.backbone.body.ext_classifier
    optimizer = func_util.get_optimizer(ext_classifier, optim_config['type'], optim_config['params'])
    scheduler_config = train_config['scheduler']
    lr_scheduler = func_util.get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
    if file_util.check_if_exists(ckpt_file_path):
        load_ckpt(ckpt_file_path, model=ext_classifier, optimizer=optimizer, lr_scheduler=lr_scheduler)

    best_val_roc_auc = 0.0
    num_epochs = train_config['num_epochs']
    log_freq = train_config['log_freq']
    for epoch in range(num_epochs):
        if distributed:
            train_sampler.set_epoch(epoch)

        train_model(model, optimizer, train_data_loader, device, epoch, log_freq)
        lr_scheduler.step()

        # evaluate after every epoch
        val_roc_auc = evaluate(model, val_data_loader, device, min_recall=args.min_recall, split_name='Validation')
        if val_roc_auc > best_val_roc_auc:
            print('Updating ckpt (ROC-AUC: {:.4f} > {:.4f})'.format(val_roc_auc, best_val_roc_auc))
            best_val_roc_auc = val_roc_auc
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
    module_util.freeze_module_params(model)
    module_util.unfreeze_module_params(model.backbone.body.ext_classifier)
    model.ext_training = True
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

    if args.train:
        print('Start training')
        start_time = time.time()
        ckpt_file_path = model_config['backbone']['ext_config']['ckpt']
        train(model, train_sampler, train_data_loader, val_data_loader, device, distributed,
              config, args, ckpt_file_path)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        load_ckpt(ckpt_file_path, model=model.backbone.body.ext_classifier)
    evaluate(model, test_data_loader, device=device, min_recall=args.min_recall, split_name='Test')


if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
