import copy

import torch.nn as nn

from .base import BaseHeadMimic


def mimic_version1(make_bottleneck=False):
    if make_bottleneck:
        return nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
    return nn.Sequential(
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, kernel_size=2, stride=1, bias=False),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


def mimic_version2(teacher_model_type, make_bottleneck=False):
    prior_head_mimic = mimic_version1(make_bottleneck)
    if teacher_model_type == 'retinanet50':
        return nn.Sequential(
            prior_head_mimic,
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=2, stride=2, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
    elif teacher_model_type == 'retinanet101':
        return nn.Sequential(
            prior_head_mimic,
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=2, stride=2, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
    raise ValueError('teacher_model_type `{}` is not expected'.format(teacher_model_type))


def mimic_version3(teacher_model_type, make_bottleneck=False):
    prior_head_mimic = mimic_version2(make_bottleneck)
    if teacher_model_type == 'retinanet50':
        return nn.Sequential(
            prior_head_mimic,
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=2, stride=2, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
    elif teacher_model_type == 'retinanet101':
        return nn.Sequential(
            prior_head_mimic,
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=2, stride=2, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
    raise ValueError('teacher_model_type `{}` is not expected'.format(teacher_model_type))


class RetinaNetHeadMimic(BaseHeadMimic):
    def __init__(self, teacher_model_type, version):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        if version in ['1', '1b']:
            self.module_seq = mimic_version1(version == '1b')
        elif version in ['2', '2b']:
            self.module_seq = mimic_version2(teacher_model_type, version == '2b')
        elif version in ['3', '3b']:
            self.module_seq = mimic_version3(teacher_model_type, version == '3b')
        else:
            raise ValueError('version `{}` is not expected'.format(version))

        self.teacher_model_type = teacher_model_type
        self.initialize_weights()

    def forward(self, sample_batch):
        zs = self.extractor(sample_batch)
        return self.module_seq(zs)


class RetinaNetMimic(nn.Module):
    def __init__(self, org_model, mimic_modules):
        super().__init__()
        self.org_model = copy.deepcopy(org_model.module if isinstance(org_model, nn.DataParallel)
                                       else copy.deepcopy(org_model))
        self.org_model.backbone = nn.Sequential(*mimic_modules)

    def forward(self, *input):
        self.org_model(*input)
