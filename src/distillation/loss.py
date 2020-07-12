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


class GeneralizedCustomLoss(CustomLoss):
    def __init__(self, criterion_config):
        super().__init__(criterion_config)

    def forward(self, output_dict, org_loss_dict):
        loss_dict = dict()
        for loss_name, ((teacher_path, teacher_output), (student_path, student_output)) in output_dict.items():
            _, criterion, factor = self.term_dict[loss_name]
            loss_dict[loss_name] = criterion(teacher_output, student_output) * factor

        sub_total_loss = sum(loss for loss in loss_dict.values())
        if self.org_loss_factor == 0:
            return sub_total_loss
        return sub_total_loss + self.org_loss_factor * sum(loss for loss in org_loss_dict.values())


LOSS_DICT = {
    'general': GeneralizedCustomLoss
}


def get_loss(criterion_config):
    criterion_type = criterion_config['type']
    if criterion_type in LOSS_DICT:
        return LOSS_DICT[criterion_type](criterion_config)
    raise ValueError('criterion_type `{}` is not expected'.format(criterion_type))
