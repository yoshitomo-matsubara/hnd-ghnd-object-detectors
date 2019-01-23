import argparse
import logging
import os

import collections
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from myutils.common import file_util, log_util, yaml_util
from myutils.pytorch import func_util
from structure.datasets import CocoDataset, CSVDataset
from utils import eval_util, retinanet_util


def get_args():
    argparser = argparse.ArgumentParser(description=os.path.basename(__file__))
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--epoch', type=int, help='epoch (higher priority than config if set)')
    argparser.add_argument('--lr', type=float, help='learning rate (higher priority than config if set)')
    argparser.add_argument('--log', default='./log/', help='log dir path')
    argparser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    argparser.add_argument('-eval', action='store_true', help='evaluation only (i.e. no training)')
    return argparser.parse_args()


def train(retinanet, train_dataloader, optimizer, loss_hist_list, epoch, num_logs):
    retinanet.train()
    retinanet.module.freeze_bn()
    epoch_loss_list = list()
    num_batches = len(train_dataloader)
    unit_size = num_batches // num_logs
    for i, data in enumerate(train_dataloader):
        try:
            optimizer.zero_grad()
            classification_losses, regression_losses = retinanet([data['img'].cuda().float(), data['annot']])
            classification_loss = classification_losses.mean()
            regression_loss = regression_losses.mean()
            loss = classification_loss + regression_loss
            if loss == 0:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()
            loss_hist_list.append(float(loss))
            epoch_loss_list.append(float(loss))
            if i > 0 and i % unit_size == 0:
                logging.info('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} '
                             '| Running loss: {:1.5f}'.format(epoch, i + 1, float(classification_loss),
                                                              float(regression_loss), np.mean(loss_hist_list)))

            del classification_loss
            del regression_loss
        except Exception as e:
            logging.info(e)
    return epoch_loss_list


def evaluate(val_dataset, model):
    if isinstance(val_dataset, CocoDataset):
        logging.info('Evaluating dataset')
        eval_util.evaluate_coco(val_dataset, model)
    elif isinstance(val_dataset, CSVDataset):
        logging.info('Evaluating dataset')
        meam_ap = eval_util.evaluate_csv(val_dataset, model)
        logging.info('mAP: {}'.format(meam_ap))


def save_ckpt(model, file_path):
    file_util.make_parent_dirs(file_path)
    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), file_path)


def build_model(args, device, config):
    train_dataset, val_dataset = retinanet_util.get_datasets(config['dataset'])
    train_config = config['train']
    train_data_loader = retinanet_util.get_train_data_loader(train_dataset, train_config['batch_size'])

    model_config = config['model']
    ckpt_file_path = model_config['ckpt']
    model = retinanet_util.get_model(device, ckpt_file_path, **model_config['params'])
    if args.eval:
        evaluate(val_dataset, model)
        return

    num_epochs = train_config['epoch'] if args.epoch is None else args.epoch
    num_logs = train_config['num_logs']
    optimizer_config = train_config['optimizer']
    optimizer = func_util.get_optimizer(model, optimizer_config['type'], optimizer_config['params'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist_list = collections.deque(maxlen=500)

    model.train()
    model.module.freeze_bn()
    logging.info('Num training images: {}'.format(len(train_dataset)))
    for epoch in range(num_epochs):
        epoch_losses = train(model, train_data_loader, optimizer, loss_hist_list, epoch, num_logs)
        if val_dataset is None:
            continue

        evaluate(val_dataset, model)
        scheduler.step(np.mean(epoch_losses))
        save_ckpt(model, ckpt_file_path)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    log_util.setup_logging(os.path.join(args.log, args.model))
    logging.info('CUDA is {}available'.format('' if torch.cuda.is_available() else 'not '))
    logging.info('Device: {}'.format(device))
    config = yaml_util.load_yaml_file(args.config)
    build_model(args, device, config)


if __name__ == '__main__':
    main(get_args())
