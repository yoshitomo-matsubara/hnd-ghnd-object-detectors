import logging
import os
import re

import torch

from models.mimic.retinanet_mimic import *
from myutils.common import yaml_util
from myutils.pytorch import module_util
from utils import model_util


def resume_from_ckpt(ckpt_file_path, model, is_student=False):
    if not os.path.exists(ckpt_file_path):
        logging.info('{} checkpoint was not found at {}'.format("Student" if is_student else "Teacher", ckpt_file_path))
        if is_student:
            return 1, 1e60
        return 1

    logging.info('Resuming from checkpoint..')
    checkpoint = torch.load(ckpt_file_path)
    if 'model' not in checkpoint or 'epoch' not in checkpoint:
        model.load_state_dict(checkpoint)
        return None

    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    start_epoch = checkpoint['epoch']
    if is_student:
        return start_epoch, checkpoint['best_avg_loss']
    return start_epoch


def extract_teacher_model(model, input_shape, device, teacher_model_config):
    start_idx = teacher_model_config['start_idx']
    end_idx = teacher_model_config['end_idx']
    if start_idx is None or end_idx is None:
        return model

    modules = list()
    module_util.extract_decomposable_modules(model, torch.rand(1, *input_shape).to(device), modules)
    if start_idx > 0:
        frozen_module = nn.Sequential(*modules[:start_idx])
        module_util.freeze_module_params(frozen_module)
        return nn.Sequential(frozen_module, *modules[start_idx:end_idx])
    return nn.Sequential(*modules[start_idx:end_idx])


def get_teacher_model(teacher_model_config, input_shape, device):
    teacher_config = yaml_util.load_yaml_file(teacher_model_config['config'])
    model = model_util.get_model(teacher_config, device)
    model_config = teacher_config['model']
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    teacher_model = extract_teacher_model(target_model.backbone, input_shape, device, teacher_model_config)
    module_util.freeze_module_params(teacher_model)
    return teacher_model.to(device), model_config['type']


def get_student_model(teacher_model_type, student_model_config):
    student_model_type = student_model_config['type']
    if teacher_model_type.startswith('retinanet') and re.search(r'^retinanet.*_whole_backbone_mimic', student_model_type):
        return RetinaNetWholeBackboneMimic(teacher_model_type, student_model_config['version'])
    elif teacher_model_type.startswith('retinanet') and re.search(r'^retinanet.*_mimic', student_model_type):
        return RetinaNetHeadMimic(teacher_model_type, student_model_config['version'])
    raise ValueError('teacher_model_type `{}` and/or '
                     'student_model_type `{}` are not expected'.format(teacher_model_type, student_model_type))


def load_student_model(student_config, teacher_model_type, device):
    student_model_config = student_config['student_model']
    student_model = get_student_model(teacher_model_type, student_model_config)
    student_model = student_model.to(device)
    resume_from_ckpt(student_model_config['ckpt'], student_model, True)
    return student_model


def get_org_model(teacher_model_config, device):
    teacher_config = yaml_util.load_yaml_file(teacher_model_config['config'])
    model = model_util.get_model(teacher_config, device)
    if device == 'cuda':
        model = nn.DataParallel(model)
    return model, teacher_config['model']['type']


def get_tail_network(config, org_model, tail_modules):
    mimic_model_config = config['mimic_model']
    mimic_type = mimic_model_config['type']
    if mimic_type.startswith('retinanet'):
        return RetinaNetMimic(org_model, tail_modules)
    raise ValueError('mimic_type `{}` is not expected'.format(mimic_type))


def get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device):
    student_model = load_student_model(config, teacher_model_type, device)
    if isinstance(student_model, BaseBackboneMimic):
        mimic_model = student_model
    elif isinstance(student_model, BaseHeadMimic):
        org_modules = list()
        input_batch = torch.rand(config['input_shape']).unsqueeze(0).to(device)
        target_model = org_model.module if isinstance(org_model, nn.DataParallel) else org_model
        module_util.extract_decomposable_modules(target_model.backbone, input_batch, org_modules)
        end_idx = teacher_model_config['end_idx']
        mimic_model_config = config['mimic_model']
        mimic_type = mimic_model_config['type']
        if mimic_type.startswith('retinanet'):
            mimic_model = RetinaNetMimic(target_model, student_model, org_modules[end_idx:], len(org_modules))
        else:
            raise ValueError('mimic_type `{}` is not expected'.format(mimic_type))
    else:
        raise ValueError('type `{}` is not expected'.format(type(student_model)))

    mimic_model = mimic_model.to(device)
    if device == 'cuda':
        mimic_model = nn.DataParallel(mimic_model)
    return mimic_model
