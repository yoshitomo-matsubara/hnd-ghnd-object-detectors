from torch import nn

from distillation.loss import get_loss
from myutils.pytorch import module_util


class DistillationBox(nn.Module):
    def __init__(self, teacher_model, student_model, criterion_config):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.target_module_pairs = list()
        output_dict = dict()

        def extract_output(self, input, output):
            global output_dict
            key = output_dict[self.__dict__['loss_name']]
            index = 0 if self.__dict__['is_teacher'] else 1
            output_dict[key][index] = (self.__dict__['path_from_root'], output)

        for loss_name, loss_config in criterion_config['terms'].items():
            teacher_path, student_path = loss_config['ts_modules']
            teacher_module = module_util.get_module(self.teacher_model, teacher_path)
            student_module = module_util.get_module(self.student_model, student_path)
            teacher_module.__dict__['loss_name'] = loss_name
            student_module.__dict__['loss_name'] = loss_name
            teacher_module.__dict__['path_from_root'] = teacher_path
            student_module.__dict__['path_from_root'] = student_path
            teacher_module.__dict__['is_teacher'] = True
            student_module.__dict__['is_teacher'] = False
            teacher_module.register_forward_hook(extract_output)
            student_module.register_forward_hook(extract_output)
            output_dict[loss_name] = [None, None]

        self.criterion = get_loss(criterion_config)
        self.output_dict = output_dict

    def forward(self, images, targets):
        for key in self.output_dict:
            self.output_dict[key] = [None, None]

        self.teacher_model(images)
        org_loss_dict = self.student_model(images, targets)
        total_loss = self.criterion(self.output_dict, org_loss_dict)
        return total_loss
