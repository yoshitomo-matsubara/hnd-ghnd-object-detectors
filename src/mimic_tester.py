import argparse
import logging

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from myutils.common import file_util, log_util, yaml_util
from utils import mimic_util, retinanet_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Tester')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('-init', action='store_true', help='overwrite checkpoint')
    return argparser


def save_ckpt(student_model, epoch, best_avg_loss, ckpt_file_path, teacher_model_type):
    target_model = student_model.module if isinstance(student_model, nn.DataParallel) else student_model
    logging.info('Saving..')
    state = {
        'type': teacher_model_type,
        'model': target_model.state_dict(),
        'epoch': epoch + 1,
        'best_avg_loss': best_avg_loss,
        'student': True
    }
    file_util.make_parent_dirs(ckpt_file_path)
    torch.save(state, ckpt_file_path)


def evaluate(org_model, mimic_model, teacher_model_type, config):
    if teacher_model_type.startswith('retinanet'):
        _, val_dataset = retinanet_util.get_datasets(config['dataset'])
        logging.info('Evaluating original model')
        org_result = retinanet_util.evaluate(val_dataset, org_model)
        logging.info(org_result)
        logging.info('Evaluating mimic model')
        mimic_result = retinanet_util.evaluate(val_dataset, mimic_model)
        logging.info(mimic_result)
    else:
        raise ValueError('teacher_model_type `{}` is not expected'.format(teacher_model_type))


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    config = yaml_util.load_yaml_file(args.config)
    log_util.setup_logging(config['test']['log'])
    logging.info('CUDA is {}available'.format('' if torch.cuda.is_available() else 'not '))
    logging.info('Device: {}'.format(device))
    teacher_model_config = config['teacher_model']
    org_model, teacher_model_type = mimic_util.get_org_model(teacher_model_config, device)
    mimic_model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device)
    evaluate(org_model, mimic_model, teacher_model_type, config)
    file_util.save_pickle(mimic_model, config['mimic_model']['ckpt'])


if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
