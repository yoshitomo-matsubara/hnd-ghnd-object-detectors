import logging
import os

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
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    start_epoch = checkpoint['epoch']
    if is_student:
        return start_epoch, checkpoint['best_avg_loss']
    return start_epoch


def extract_teacher_model(model, input_shape, device, teacher_model_config):
    modules = list()
    module_util.extract_decomposable_modules(model, torch.rand(1, *input_shape).to(device), modules)
    start_idx = teacher_model_config['start_idx']
    end_idx = teacher_model_config['end_idx']
    if start_idx > 0:
        frozen_module = nn.Sequential(*modules[:start_idx])
        module_util.freeze_module_params(frozen_module)
        return nn.Sequential(frozen_module, *modules[start_idx:end_idx])
    return nn.Sequential(*modules[start_idx:end_idx])


def get_teacher_model(teacher_model_config, input_shape, device):
    teacher_config = yaml_util.load_yaml_file(teacher_model_config['config'])
    model = model_util.get_model(teacher_config, device)
    model_config = teacher_config['model']
    resume_from_ckpt(model_config['ckpt'], model)
    return extract_teacher_model(model.backbone, input_shape, device, teacher_model_config), model_config['type']


def get_student_model(teacher_model_type, student_model_config):
    student_model_type = student_model_config['type']
    if teacher_model_type.startswith('retinanet') and student_model_type == 'retinanet_head_mimic':
        return RetinaNetHeadMimic(teacher_model_type, student_model_config['version'])
    raise ValueError('teacher_model_type `{}` is not expected'.format(teacher_model_type))


def load_student_model(student_config, teacher_model_type, device):
    student_model_config = student_config['student_model']
    student_model = get_student_model(teacher_model_type, student_model_config)
    student_model = student_model.to(device)
    resume_from_ckpt(student_model_config['ckpt'], student_model, True)
    return student_model


def get_org_model(teacher_model_config, device):
    teacher_config = yaml_util.load_yaml_file(teacher_model_config['config'])
    model = model_util.get_model(teacher_config, device)
    model_config = teacher_config['model']
    resume_from_ckpt(model_config['ckpt'], model)
    return model, model_config['type']


def get_tail_network(config, org_model, tail_modules):
    mimic_model_config = config['mimic_model']
    mimic_type = mimic_model_config['type']
    if mimic_type.startswith('retinanet'):
        return RetinaNetMimic(org_model, tail_modules)
    raise ValueError('mimic_type `{}` is not expected'.format(mimic_type))


def get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device):
    student_model = load_student_model(config, teacher_model_type, device)
    org_modules = list()
    input_batch = torch.rand(config['input_shape']).unsqueeze(0).to(device)
    module_util.extract_decomposable_modules(org_model.backbone, input_batch, org_modules)
    end_idx = teacher_model_config['end_idx']
    mimic_modules = [student_model]
    mimic_modules.extend(org_modules[end_idx:])
    mimic_model_config = config['mimic_model']
    mimic_type = mimic_model_config['type']
    if mimic_type.startswith('retinanet'):
        return RetinaNetMimic(org_model, mimic_modules)
    raise ValueError('mimic_type `{}` is not expected'.format(mimic_type))
