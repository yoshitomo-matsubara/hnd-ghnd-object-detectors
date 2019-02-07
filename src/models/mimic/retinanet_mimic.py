import copy

import torch.nn as nn

from .base import BaseBackboneMimic, BaseHeadMimic


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


def mimic_version2(teacher_model_type):
    if teacher_model_type == 'retinanet50':
        return nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=2, stride=2, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
    elif teacher_model_type == 'retinanet101':
        return nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=2, stride=2, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
    raise ValueError('teacher_model_type `{}` is not expected'.format(teacher_model_type))


def mimic_version3(teacher_model_type):
    if teacher_model_type == 'retinanet50':
        return nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 2048, kernel_size=2, stride=2, padding=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
    elif teacher_model_type == 'retinanet101':
        return nn.Sequential(
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

        module_seq = mimic_version1(version.endswith('b'))
        if version in ['2', '2b', '3', '3b']:
            module_seq = nn.Sequential(module_seq, mimic_version2(teacher_model_type))
        if version in ['3', '3b']:
            module_seq = nn.Sequential(module_seq, mimic_version3(teacher_model_type))
        else:
            raise ValueError('version `{}` is not expected'.format(version))

        self.module_seq = module_seq
        self.teacher_model_type = teacher_model_type
        self.initialize_weights()

    def forward(self, sample_batch):
        zs = self.extractor(sample_batch)
        return self.module_seq(zs)


class RetinaNetBackboneMimic(nn.Module):
    def __init__(self, head_model, tail_modules, org_length):
        super().__init__()
        self.head_model = head_model
        diff_length = org_length - len(tail_modules)
        if org_length == 20:
            # RetinaNet-50
            layer2_end_idx = 11 - diff_length
            layer3_end_idx = 17 - diff_length
            self.layer2 = nn.Sequential(*tail_modules[:layer2_end_idx])
            self.layer3 = nn.Sequential(*tail_modules[layer2_end_idx:layer3_end_idx])
            self.layer4 = nn.Sequential(*tail_modules[layer3_end_idx:])
        else:
            raise ValueError('org_length `{}` is not expected'.format(org_length))

    def forward(self, *input):
        z = self.head_model(*input)
        z2 = self.layer2(z)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)
        return z2, z3, z4


class RetinaNetWholeBackboneMimic(BaseBackboneMimic):
    def __init__(self, teacher_model_type, version):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = mimic_version1(version.endswith('b'))
        self.layer3 = mimic_version2(teacher_model_type)
        self.layer4 = mimic_version3(teacher_model_type)

    def forward(self, *input):
        z = self.extractor(*input)
        z2 = self.layer2(z)
        z3 = self.layer3(z2)
        z4 = self.layer4(z3)
        return z2, z3, z4


class RetinaNetMimic(nn.Module):
    def __init__(self, org_model, head_model, tail_modules, org_length):
        super().__init__()
        self.org_model = copy.deepcopy(org_model.module if isinstance(org_model, nn.DataParallel)
                                       else copy.deepcopy(org_model))
        self.org_model.backbone = RetinaNetBackboneMimic(head_model, tail_modules, org_length)

    def forward(self, *input):
        return self.org_model(*input)
