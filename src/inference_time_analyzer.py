import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from myutils.common import misc_util, yaml_util
from utils import mimic_util, model_util, module_spec_util, retinanet_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Inference Time Analyzer')
    argparser.add_argument('--config', required=True, help='yaml file path')
    return argparser


def evaluate(model, model_type, config):
    if model_type.startswith('retinanet'):
        _, val_dataset = retinanet_util.get_datasets(config['dataset'])
        result_config = config['test']['result']
        print('Evaluating original model')
        retinanet_util.evaluate(val_dataset, model, result_config['org'])
    else:
        raise ValueError('model_type `{}` is not expected'.format(model_type))


def extract_timestamps(module, output_tuple_list):
    child_modules = list(module.children())
    if not child_modules:
        output_tuple_list.append((type(module).__name__, np.array(module.timestamp_list)))
        return

    for child_module in child_modules:
        extract_timestamps(child_module, output_tuple_list)


def calculate_inference_time(model):
    model = model.module if isinstance(model, nn.DataParallel) else model
    start_timestamps = np.array(model.timestamps_dict['start'])
    end_timestamps = np.array(model.timestamps_dict['end'])
    inference_times = end_timestamps - start_timestamps
    print('Inference Time: {} \xb1 {} [ms]'.format(np.mean(inference_times), np.std(inference_times)))
    tuple_list = [('Start', start_timestamps)]
    extract_timestamps(model, tuple_list)
    tuple_list.append(('End', end_timestamps))
    tuple_list = sorted(tuple_list, key=lambda x: x[1][0])
    for i in range(len(tuple_list) - 1, 0, -1):
        tuple_list[i][0] = tuple_list[i - 1][0]
    return tuple_list


def plot_inference_time(results):
    module_name_list = list()
    mean_exec_time_list = list()
    std_exec_time_list = list()
    print('Module name\tExecution time[ms]')
    for i, module_name, exec_times in enumerate(results[1:]):
        module_name_list.append(module_name + ': {}'.format(i + 1))
        mean_exec_time_list.append(exec_times.mean())
        std_exec_time_list.append(exec_times.std())
        print('{}\t{} \xb1 {}'.format(module_name_list[-1], mean_exec_time_list[-1], std_exec_time_list[-1]))

    if misc_util.check_if_plottable():
        xs = np.array(list(range(len(module_name_list))))
        plt.plot(xs, mean_exec_time_list)
        plt.xlabel('Modules')
        plt.ylabel('Execution time [ms]')
        plt.xticks(xs, module_name_list, rotation=90, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=13)
        plt.tight_layout()
        plt.show()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    config = yaml_util.load_yaml_file(args.config)
    print('CUDA is {}available'.format('' if torch.cuda.is_available() else 'not '))
    print('Device: {}'.format(device))
    if 'teacher_model' in config:
        teacher_model_config = config['teacher_model']
        org_model, teacher_model_type = mimic_util.get_org_model(teacher_model_config, device)
        model = mimic_util.get_mimic_model(config, org_model, teacher_model_type, teacher_model_config, device)
        model_type = teacher_model_type
    else:
        model = model_util.get_model(config, device)
        model_type = config['model']['type']

    module_spec_util.register_forward_hook(model, module_spec_util.time_record_hook)
    evaluate(model, model_type, config)
    results = calculate_inference_time(model)
    plot_inference_time(results)


if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
