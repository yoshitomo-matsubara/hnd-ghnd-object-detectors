import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.org.retinanet import get_retinanet
from myutils.common import file_util
from structure.datasets import CocoDataset, CSVDataset
from structure.samplers import AspectRatioBasedSampler
from structure.transformers import Resizer, Augmenter, Normalizer
from structure.transformers import collater
from utils import eval_util


def load_published_pretrained_state_dict(ckpt_file_path):
    # load pretrained state dict published in https://github.com/yhenon/pytorch-retinanet
    state_dict = torch.load(ckpt_file_path)
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith('layer') or key == 'conv1.weight' or key.startswith('bn1.'):
            state_dict['backbone.' + key] = state_dict.pop(key)
        elif key.startswith('regressionModel'):
            state_dict[key.replace('regressionModel', 'regressor')] = state_dict.pop(key)
        elif key.startswith('classificationModel'):
            state_dict[key.replace('classificationModel', 'classifier')] = state_dict.pop(key)
        elif key.startswith('fpn.'):
            state_dict[key.lower()] = state_dict.pop(key)
    return state_dict


def get_model(device, ckpt_file_path, **kwargs):
    model = get_retinanet(**kwargs)
    if file_util.check_if_exists(ckpt_file_path):
        state_dict = load_published_pretrained_state_dict(ckpt_file_path)\
            if os.path.basename(ckpt_file_path) == 'coco_resnet_50_map_0_335_state_dict.pt'\
            else torch.load(ckpt_file_path)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.training = True
    if device == 'cuda':
        model = nn.DataParallel(model)
    return model


def get_datasets(dataset_config):
    dataset_name = dataset_config['name']
    data_config = dataset_config['data']
    if dataset_name.startswith('coco'):
        train_data_config = data_config['train']
        train_dataset = CocoDataset(train_data_config['annotation'], train_data_config['img_dir'],
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        val_data_config = data_config['val']
        val_dataset = CocoDataset(val_data_config['annotation'], val_data_config['img_dir'],
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
        return train_dataset, val_dataset
    elif dataset_name == 'csv':
        class_file_path = data_config['class']
        train_dataset = CSVDataset(data_file_path=data_config['train'], class_list=class_file_path,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        val_file_path = data_config['val']
        if not file_util.check_if_exists(val_file_path):
            val_dataset = None
            logging.info('No validation annotations provided.')
        else:
            val_dataset = CSVDataset(data_file_path=val_file_path, class_list=class_file_path,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))
        return train_dataset, val_dataset
    raise ValueError('dataset_name `{}` is not expected'.format(dataset_name))


def get_train_data_loader(train_dataset, batch_size=2, drop_last=False, num_workers=3):
    train_sampler = AspectRatioBasedSampler(train_dataset, batch_size=batch_size, drop_last=drop_last)
    return DataLoader(train_dataset, num_workers=num_workers, collate_fn=collater, batch_sampler=train_sampler)


def evaluate(dataset, model, output_file_path='tmp.json'):
    if isinstance(dataset, CocoDataset):
        logging.info('Using COCO dataset')
        return eval_util.evaluate_coco(dataset, model, output_file_path)
    elif isinstance(dataset, CSVDataset):
        logging.info('Using CSV dataset')
        mean_ap = eval_util.evaluate_csv(dataset, model)
        logging.info('mAP: {}'.format(mean_ap))
        return mean_ap
    else:
        raise ValueError('type of dataset `{}` is not expected'.format(type(dataset)))
