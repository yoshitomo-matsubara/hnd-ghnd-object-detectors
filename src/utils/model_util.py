import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from myutils.common import file_util
from utils import yolo_util


def get_model(config, device):
    model_config = config['model']
    model_type = model_config['type']
    ckpt_file_path = model_config['ckpt']
    model_params_config = model_config['params']
    if model_type.startswith('yolo'):
        return yolo_util.get_model(device, ckpt_file_path, **model_params_config)
    raise ValueError('teacher_model_type `{}` is not expected'.format(model_type))


def get_data_loaders(dataset_config, model_type, batch_size):
    raise ValueError('model_type `{}` is not expected'.format(model_type))


def save_ckpt(model, file_path):
    file_util.make_parent_dirs(file_path)
    torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), file_path)
