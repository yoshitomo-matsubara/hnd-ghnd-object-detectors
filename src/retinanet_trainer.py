import argparse
import collections
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from models.org.retinanet import get_retinanet
from myutils.common import file_util, yaml_util
from structure.datasets import CocoDataset, CSVDataset
from utils import eval_util, retinanet_util


def get_args():
    argparser = argparse.ArgumentParser(description=os.path.basename(__file__))
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--epoch', type=int, help='epoch (higher priority than config if set)')
    argparser.add_argument('--lr', type=float, help='learning rate (higher priority than config if set)')
    argparser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    return argparser.parse_args()


def get_model(device, **kwargs):
    model = get_retinanet(**kwargs)
    model = model.to(device)
    model.training = True
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
    return model


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
                print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f}'
                      ' | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch, i + 1, float(classification_loss), float(regression_loss), np.mean(loss_hist_list)))

            del classification_loss
            del regression_loss
        except Exception as e:
            print(e)
    return epoch_loss_list


def save_ckpt(model, file_path):
    file_util.make_parent_dirs(file_path)
    torch.save(model.state_dict(), file_path)


def build_model(args, device, config):
    model_config = config['model']
    # Create the data loaders
    train_dataset, val_dataset = retinanet_util.get_datasets(config['dataset'])
    train_data_loader = retinanet_util.get_train_data_loader(train_dataset)

    # Create the model
    model = get_model(device, num_classes=train_dataset.num_classes(), **model_config['params'])

    train_config = config['train']
    num_epochs = train_config['epoch'] if args.epoch is None else args.epoch
    num_logs = train_config['num_logs']
    ckpt_file_path = model_config['ckpt']

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist_list = collections.deque(maxlen=500)
    model.train()
    model.module.freeze_bn()
    print('Num training images: {}'.format(len(train_dataset)))
    for epoch in range(num_epochs):
        epoch_losses = train(model, train_data_loader, optimizer, loss_hist_list, epoch, num_logs)
        if val_dataset is None:
            continue

        if isinstance(val_dataset, CocoDataset):
            print('Evaluating dataset')
            eval_util.evaluate_coco(val_dataset, model)
        elif isinstance(val_dataset, CSVDataset):
            print('Evaluating dataset')
            meam_ap = eval_util.evaluate_csv(val_dataset, model)
            print('mAP: {}'.format(meam_ap))

        scheduler.step(np.mean(epoch_losses))
        save_ckpt(model, ckpt_file_path)


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    print('CUDA is {}available'.format('' if torch.cuda.is_available() else 'not '))
    print('Device setting is `{}`'.format(device))
    config = yaml_util.load_yaml_file(args.config)
    build_model(args, device, config)


if __name__ == '__main__':
    run(get_args())
