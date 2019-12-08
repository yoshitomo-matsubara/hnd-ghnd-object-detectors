import torch
from torch import nn
from torchvision.models import detection

from models.org import rcnn
from myutils.common import file_util
from utils import misc_util
from structure.transformer import get_bottleneck_transformer


def save_ckpt(model, optimizer, lr_scheduler, best_value, config, args, output_file_path):
    file_util.make_parent_dirs(output_file_path)
    model_state_dict =\
        model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()
    misc_util.save_on_master({'model': model_state_dict, 'optimizer': optimizer.state_dict(), 'best_value': best_value,
                              'lr_scheduler': lr_scheduler.state_dict(), 'config': config, 'args': args},
                             output_file_path)


def load_ckpt(ckpt_file_path, model=None, optimizer=None, lr_scheduler=None, strict=True):
    if not file_util.check_if_exists(ckpt_file_path):
        print('ckpt file is not found at `{}`'.format(ckpt_file_path))
        return None, None

    ckpt = torch.load(ckpt_file_path, map_location='cpu')
    if model is not None:
        print('Loading model parameters')
        model.load_state_dict(ckpt['model'], strict=strict)
    if optimizer is not None:
        print('Loading optimizer parameters')
        optimizer.load_state_dict(ckpt['optimizer'])
    if lr_scheduler is not None:
        print('Loading scheduler parameters')
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
    return ckpt.get('best_value', 0.0), ckpt['config'], ckpt['args']


def get_model(model_config, device, strict=True, bottleneck_transformer=None):
    model_name = model_config['name']
    ckpt_file_path = model_config['ckpt']
    model_params_config = model_config['params']
    if model_name in rcnn.MODEL_CLASS_DICT:
        backbone_config = model_config['backbone']
        if bottleneck_transformer is None and 'bottleneck_transformer' in model_config:
            bottleneck_transformer = get_bottleneck_transformer(model_config['bottleneck_transformer'])

        model = rcnn.get_model(model_name, backbone_config=backbone_config, strict=strict,
                               bottleneck_transformer=bottleneck_transformer, **model_params_config)
        if 'ext_config' in backbone_config:
            ext_config = backbone_config['ext_config']
            load_ckpt(ext_config['ckpt'], model=model.backbone.body.get_ext_classifier())
            strict = False
    else:
        raise ValueError('model_name `{}` is not expected'.format(model_name))

    load_ckpt(ckpt_file_path, model=model, strict=strict)
    return model.to(device)


def get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module

    iou_type_list = ['bbox']
    if isinstance(model_without_ddp, (detection.MaskRCNN, rcnn.MaskRCNN)):
        iou_type_list.append('segm')
    if isinstance(model_without_ddp, (detection.KeypointRCNN, rcnn.KeypointRCNN)):
        iou_type_list.append('keypoints')
    return iou_type_list
