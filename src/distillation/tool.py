import random

from torch import nn

from distillation.loss import get_loss
from models.org.rcnn import KeypointRCNN
from myutils.pytorch import module_util


class DistillationBox(nn.Module):
    def __init__(self, teacher_model, student_model, criterion_config):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.target_module_pairs = list()

        def extract_output(self, input, output):
            self.__dict__['distillation_box']['output'] = output

        for loss_name, loss_config in criterion_config['terms'].items():
            teacher_path, student_path = loss_config['ts_modules']
            self.target_module_pairs.append((teacher_path, student_path))
            teacher_module = module_util.get_module(self.teacher_model, teacher_path)
            student_module = module_util.get_module(self.student_model, student_path)
            teacher_module.__dict__['distillation_box'] = {'loss_name': loss_name, 'path_from_root': teacher_path,
                                                           'is_teacher': True}
            student_module.__dict__['distillation_box'] = {'loss_name': loss_name, 'path_from_root': student_path,
                                                           'is_teacher': False}
            teacher_module.register_forward_hook(extract_output)
            student_module.register_forward_hook(extract_output)

        self.criterion = get_loss(criterion_config)
        self.require_adjustment = isinstance(self.student_model, KeypointRCNN)

    def forward(self, images, targets):
        fixed_sizes = [random.choice(self.teacher_model.min_size) for _ in images]
        self.teacher_model(images, fixed_sizes=fixed_sizes)
        org_loss_dict = self.student_model(images, targets, fixed_sizes=fixed_sizes)
        output_dict = dict()
        for teacher_path, student_path in self.target_module_pairs:
            teacher_dict = module_util.get_module(self.teacher_model, teacher_path).__dict__['distillation_box']
            student_dict = module_util.get_module(self.student_model, student_path).__dict__['distillation_box']
            output_dict[teacher_dict['loss_name']] = ((teacher_dict['path_from_root'], teacher_dict['output']),
                                                      (student_dict['path_from_root'], student_dict['output']))

        total_loss = self.criterion(output_dict, org_loss_dict)
        return total_loss
