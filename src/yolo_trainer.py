import argparse
import logging
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from myutils.common import log_util, yaml_util
from myutils.pytorch import func_util, module_util
from utils import eval_util, model_util, yolo_util


def get_args():
    argparser = argparse.ArgumentParser(description=os.path.basename(__file__))
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--epoch', type=int, help='epoch (higher priority than config if set)')
    argparser.add_argument('--lr', type=float, help='learning rate (higher priority than config if set)')
    argparser.add_argument('--log', default='./log/', help='log dir path')
    argparser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    argparser.add_argument('-eval', action='store_true', help='evaluation only (i.e. no training)')
    return argparser.parse_args()


def get_optimizer(train_config, conv_weight_decay, model):
    module_list = list()
    module_util.extract_all_child_modules(model, module_list)
    params_list = list()
    for module in module_list:
        params_dict = dict(module.named_parameters())
        for key, value in params_dict.items():
            weight_decay = conv_weight_decay if isinstance(module, nn.Conv2d) and key.endswith('.weight') else 0.0
            params_list.append({'params': value, 'weight_decay': weight_decay})

    optimizer_config = train_config['optimizer']
    optimizer_config['params']['weight_decay'] = conv_weight_decay
    return func_util.get_optimizer(params_list, optimizer_config['type'], optimizer_config['params'])


def update_lr(optimizer, current_lr, batch_size, subdivision):
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr / batch_size / subdivision


def train(model, train_dataset, val_dataset, optimizer, epoch, batch_size, subdivision, random_resize,
          train_misc_config, device, ckpt_file_path):
    steps, burn_in, ckpt_interval =\
        train_misc_config['steps'], train_misc_config['burn_in'], train_misc_config['ckpt_interval']
    train_data_loader = yolo_util.get_data_loader(train_dataset, batch_size)
    train_data_iterator = iter(train_data_loader)
    base_lr = optimizer.param_groups[0]['lr']
    current_lr = base_lr
    best_val_ap50 = 0
    model.train()
    # start training loop
    for iter_i in range(epoch):
        # learning rate scheduling
        if iter_i < burn_in:
            current_lr = base_lr * pow(iter_i / burn_in, 4)
            update_lr(optimizer, current_lr, batch_size, subdivision)
        elif iter_i == burn_in:
            current_lr = base_lr
            update_lr(optimizer, current_lr, batch_size, subdivision)
        elif iter_i in steps:
            current_lr = current_lr * 0.1
            update_lr(optimizer, current_lr, batch_size, subdivision)

        # subdivision loop
        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            try:
                imgs, targets, _, _ = next(train_data_iterator)  # load a batch
            except StopIteration:
                train_data_iterator = iter(train_data_loader)
                imgs, targets, _, _ = next(train_data_iterator)  # load a batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            loss = model(imgs, targets)
            loss.backward()

        optimizer.step()
        if iter_i % 10 == 0 and random_resize:
            img_size = (random.randint(0, 9) % 10 + 10) * 32
            train_dataset.img_shape = (img_size, img_size)
            train_dataset.img_size = img_size
            train_data_loader = yolo_util.get_data_loader(train_dataset, batch_size)
            train_data_iterator = iter(train_data_loader)

        # COCO evaluation
        # save checkpoint
        if iter_i > 0 and (iter_i % ckpt_interval == 0):
            # logging
            logging.info('[Iter {:d}/{:d}] [lr: {:1.5f}] Losses: xy {:1.5f}, wh {:1.5f}, conf {:1.5f}, '
                         'cls {:1.5f}, total {:1.5f}'.format(iter_i + 1, epoch, current_lr, model.loss_dict['xy'],
                                                             model.loss_dict['wh'], model.loss_dict['conf'],
                                                             model.loss_dict['cls'], model.loss_dict['l2']))
            ap50_95, ap50 = eval_util.evaluate_coco4yolo(val_dataset, model, device)
            model.train()
            if ap50 > best_val_ap50:
                best_val_ap50 = ap50
                model_util.save_ckpt(model, ckpt_file_path)


def build_model(args, device, config):
    dataset_config = config['dataset']
    train_dataset, val_dataset = yolo_util.get_datasets(dataset_config)
    train_config = config['train']
    model_config = config['model']
    ckpt_file_path = model_config['ckpt']
    model = yolo_util.get_model(device, ckpt_file_path, **model_config['params'])
    if args.eval:
        eval_util.evaluate_coco4yolo(val_dataset, model, device)
        return

    epoch = args.epoch if args.epoch is not None else train_config['epoch']
    if args.lr is not None:
        train_config['optimizer']['lr'] = args.lr

    random_resize = dataset_config['data']['train']['augment']['random_resize']
    train_misc_config = train_config['misc']
    batch_size = train_config['batch_size']
    decay = train_misc_config['decay']
    subdivision = train_misc_config['subdivision']

    # optimizer setup
    # set weight decay only on conv.weight
    optimizer = get_optimizer(train_config, decay * batch_size * subdivision, model)
    train(model, train_dataset, val_dataset, optimizer, epoch, batch_size, subdivision, random_resize,
          train_misc_config, device, ckpt_file_path)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    config = yaml_util.load_yaml_file(args.config)
    log_util.setup_logging(config['train']['log'])
    logging.info('CUDA is {}available'.format('' if torch.cuda.is_available() else 'not '))
    logging.info('Device: {}'.format(device))
    build_model(args, device, config)


if __name__ == '__main__':
    main(get_args())
