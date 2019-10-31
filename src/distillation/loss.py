from torch import nn

from myutils.pytorch import func_util


class CustomLoss(nn.Module):
    def __init__(self, criterion_config):
        super().__init__()
        self.org_loss_factor = criterion_config['params']['org_loss_factor']
        term_dict = dict()
        for loss_name, loss_config in criterion_config['terms'].items():
            sub_criterion_config = loss_config['criterion']
            sub_criterion = func_util.get_loss(sub_criterion_config['type'], sub_criterion_config['params'])
            term_dict[loss_name] = (loss_config['ts_modules'], sub_criterion, loss_config['factor'])
        self.term_dict = term_dict

    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward function is not implemented')


class HKDLoss4RCNN(CustomLoss):
    """
    Hierarchical Knowledge Distillation loss (when org_loss_factor = 0.0) defined in
        ``Learning Lightweight Pedestrian Detector with Hierarchical Knowledge Distillation''
        https://ieeexplore.ieee.org/document/8803079
    """

    def __init__(self, criterion_config):
        super().__init__(criterion_config)

    @staticmethod
    def loss_as_dict(teacher_output, student_output, criterion, mean=True):
        sum_loss = sum(criterion(student_output[key], teacher_output[key]) for key in teacher_output.keys())
        return sum_loss / len(teacher_output.keys) if mean else sum_loss

    def forward(self, output_dict, org_loss_dict):
        kd_loss_dict = dict()
        for loss_name, ((teacher_path, teacher_output), (student_path, student_output)) in output_dict.items():
            _, criterion, factor = self.term_dict[loss_name]
            if teacher_path == 'backbone.body.fpn':
                kd_loss_dict[loss_name] = self.loss_as_dict(teacher_output, student_output, criterion) * factor
            elif teacher_path == 'roi_heads.box_predictor':
                # TODO: check shape of each output to see #proposals
                kd_loss_dict[loss_name] =\
                    sum([criterion(to_i, so_i) for to_i, so_i in zip(teacher_output, student_output)]) / 2 * factor
            else:
                kd_loss_dict[loss_name] = criterion(teacher_output, student_output) * factor

        org_loss = sum(loss for loss in org_loss_dict.values())
        return org_loss * self.org_loss_factor + sum([loss for loss in kd_loss_dict.values()])


LOSS_DICT = {
    'hkd4rcnn': HKDLoss4RCNN
}


def get_loss(criterion_config):
    criterion_type = criterion_config['type']
    if criterion_type in LOSS_DICT:
        return LOSS_DICT[criterion_type](criterion_config)
    raise ValueError('criterion_type `{}` is not expected'.format(criterion_type))
