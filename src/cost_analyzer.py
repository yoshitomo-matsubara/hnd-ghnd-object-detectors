import argparse
import os
import time
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional

from models import get_model, get_iou_types
from models.mimic.split_rcnn import split_rcnn_model
from myutils.common import yaml_util
from myutils.pytorch import module_util
from structure.transformer import Compose, DataLogger, ToTensor
from utils import coco_util, main_util, misc_util
from utils.coco_eval_util import CocoEvaluator


def get_argparser():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', required=True, help='yaml config file')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('--json', help='dictionary to overwrite config')
    argparser.add_argument('-model_params', help='dictionary to overwrite config')
    argparser.add_argument('--modules', nargs='+', help='list of specific modules you want to count parameters')
    argparser.add_argument('--data_size', help='dataset split name to analyze data size')
    argparser.add_argument('-resized', action='store_true',
                           help='resize input image, following the preprocessing approach used in R-CNNs')
    argparser.add_argument('--bottleneck_size', help='dataset split name to analyze size of bottleneck in model')
    argparser.add_argument('--split_model', help='dataset split name to measure inference time for split models')
    argparser.add_argument('--quantize', type=int, help='quantize bottleneck tensor with 16 / 8 bits')
    argparser.add_argument('-skip_tail', action='store_true', help='skip measuring inference time for tail model')
    return argparser


def analyze_model_params(model, module_paths):
    print('Analyzing model parameters')
    print('[Whole model]')
    print('# parameters: {}'.format(module_util.count_params(model)))
    if module_paths is None or len(module_paths) == 0:
        return

    print('[Specified module(s)]')
    modules = module_util.get_components(module_paths)
    pair_list = list()
    for module, module_path in zip(modules, module_paths):
        pair_list.append([module_path, module_util.count_params(module)])

    data_frame = pd.DataFrame(pair_list)
    print('Total # parameters: {}'.format(data_frame[1].sum()))
    print(data_frame)


def summarize_data_sizes(data_sizes, title, data_rates=None):
    if data_rates is None:
        data_rates = np.hstack(([0.001], np.arange(0.5, 10.5, 0.5)))

    data_sizes = np.array(data_sizes)
    print('[{}]'.format(title))
    print('Data size:\t{:.4f} ± {:.4f} [KB]'.format(data_sizes.mean(), data_sizes.std()))
    print('# Files:\t{}\n'.format(len(data_sizes)))
    data_frame = pd.DataFrame(columns=['Data rate [Mbps]', 'Communication delay [sec]'])
    for i, data_rate in enumerate(data_rates):
        comm_delay = data_sizes * 8 / (data_rate * 1000)
        data_frame.loc[i] = [data_rate, '{:.4f} ± {:.4f}'.format(comm_delay.mean(), comm_delay.std())]
    print(data_frame)


def summarize_tensor_shape(channels, heights, widths):
    channels, heights, widths = np.array(channels), np.array(heights), np.array(widths)
    print('Tensor shape')
    print('Channel:\t{:.4f} ± {:.4f}'.format(channels.mean(), channels.std()))
    print('Height:\t{:.4f} ± {:.4f}'.format(heights.mean(), heights.std()))
    print('Width:\t{:.4f} ± {:.4f}'.format(widths.mean(), widths.std()))


def resize_for_rcnns(image, min_size=800, max_size=1333):
    width, height = image.size
    img_min_size = float(min(width, height))
    img_max_size = float(max(width, height))
    scale_factor = min_size / img_min_size
    if img_max_size * scale_factor > max_size:
        scale_factor = max_size / img_max_size
    return image.resize((int(width * scale_factor), int(height * scale_factor)), resample=Image.BILINEAR)


def analyze_data_size(dataset_config, split_name='test', resized=False):
    print('Analyzing {} data size'.format(split_name))
    split_config = dataset_config['splits'][split_name]
    dataset = coco_util.get_coco(split_config['images'], split_config['annotations'], None,
                                 split_config['remove_non_annotated_imgs'], split_config['jpeg_quality'])
    coco = dataset.coco
    org_data_size_list = list()
    comp_data_size_list = list()
    channel_list, height_list, width_list = list(), list(), list()
    min_shape, max_shape = None, None
    min_size, max_size = None, None
    for index in range(len(dataset.ids)):
        img_id = dataset.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(dataset.root, path)).convert('RGB')
        if resized:
            img = resize_for_rcnns(img)

        shape = functional.to_tensor(img).shape
        channel_list.append(shape[0])
        height_list.append(shape[1])
        width_list.append(shape[2])
        img_buffer = BytesIO()
        img.save(img_buffer, 'JPEG', quality=95)
        # Original data size [KB]
        org_data_size_list.append(img_buffer.tell() / 1024)
        img_buffer.close()
        if dataset.jpeg_quality is not None:
            img_buffer = BytesIO()
            img.save(img_buffer, 'JPEG', quality=dataset.jpeg_quality)
            # JPEG-compressed data size [KB]
            comp_data_size_list.append(img_buffer.tell() / 1024)
            img_buffer.close()

        tensor_size = np.prod(shape)
        if min_size is None or tensor_size < min_size:
            min_size = tensor_size
            min_shape = [shape[0], shape[1], shape[2]]

        if max_size is None or tensor_size > max_size:
            max_size = tensor_size
            max_shape = [shape[0], shape[1], shape[2]]

    summarize_data_sizes(org_data_size_list, 'Original')
    print('Min tensor shape: {}'.format(min_shape))
    print('Max tensor shape: {}'.format(max_shape))
    if len(comp_data_size_list) > 0:
        summarize_data_sizes(comp_data_size_list, 'JPEG quality = {}'.format(dataset.jpeg_quality))
    summarize_tensor_shape(channel_list, height_list, width_list)


@torch.no_grad()
def analyze_bottleneck_size(model, data_size_logger, device, dataset_config, split_name='test'):
    print('Analyzing size of bottleneck in model for {} dataset'.format(split_name))
    split_config = dataset_config['splits'][split_name]
    dataset = coco_util.get_coco(split_config['images'], split_config['annotations'], Compose([ToTensor()]),
                                 split_config['remove_non_annotated_imgs'], split_config['jpeg_quality'])
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=misc_util.collate_fn,
                                              num_workers=dataset_config['num_workers'])
    model.eval()
    with torch.no_grad():
        for images, _ in data_loader:
            images = list(image.to(device) for image in images)
            model(images)

    data_sizes, fp16_data_sizes, quantized_data_sizes, tensor_shapes = data_size_logger.get_data()
    channel_list, height_list, width_list = list(), list(), list()
    min_shape, max_shape = None, None
    min_size, max_size = None, None
    for channel, height, width in tensor_shapes:
        channel_list.append(channel)
        height_list.append(height)
        width_list.append(width)
        tensor_size = channel * height * width
        if min_size is None or tensor_size < min_size:
            min_size = tensor_size
            min_shape = [channel, height, width]

        if max_size is None or tensor_size > max_size:
            max_size = tensor_size
            max_shape = [channel, height, width]

    summarize_data_sizes(data_sizes, 'Bottleneck')
    print('Min tensor shape: {}'.format(min_shape))
    print('Max tensor shape: {}'.format(max_shape))
    if fp16_data_sizes[0] is not None:
        summarize_data_sizes(fp16_data_sizes, 'Quantized (16-bit) Bottleneck')
    if quantized_data_sizes[0] is not None:
        summarize_data_sizes(quantized_data_sizes, 'Quantized (8-bit) Bottleneck')
    summarize_tensor_shape(channel_list, height_list, width_list)


def summarize_inference_time(head_proc_times, tail_proc_times, total_proc_times):
    head_proc_times, tail_proc_times, total_proc_times =\
        np.array(head_proc_times), np.array(tail_proc_times), np.array(total_proc_times)
    print('[Inference time]')
    print('Head model delay:\t{:.4f} ± {:.4f} [sec]'.format(head_proc_times.mean(), head_proc_times.std()))
    print('Tail model delay:\t{:.4f} ± {:.4f} [sec]'.format(tail_proc_times.mean(), tail_proc_times.std()))
    print('Total model delay:\t{:.4f} ± {:.4f} [sec]'.format(total_proc_times.mean(), total_proc_times.std()))


@torch.no_grad()
def analyze_split_model_inference(model, device, quantization, head_only, dataset_config, split_name='test'):
    head_model, tail_model = split_rcnn_model(model, quantization)
    head_model.eval()
    tail_model.eval()
    if head_only:
        del tail_model
        tail_model = None

    cpu_device = torch.device('cpu')
    split_config = dataset_config['splits'][split_name]
    dataset = coco_util.get_coco(split_config['images'], split_config['annotations'], Compose([ToTensor()]),
                                 split_config['remove_non_annotated_imgs'], split_config['jpeg_quality'])
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=misc_util.collate_fn,
                                              num_workers=dataset_config['num_workers'])
    iou_types = get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco_util.get_coco_api_from_dataset(data_loader.dataset), iou_types)
    head_proc_time_list = list()
    tail_proc_time_list = list()
    total_proc_time_list = list()
    filtered_count = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        head_start_time = time.time()
        head_output = head_model(images)
        head_proc_time = time.time() - head_start_time
        head_proc_time_list.append(head_proc_time)
        if head_output is None or tail_model is None:
            tail_proc_time = 0.0
            if head_output is None:
                filtered_count += 1
                ch, height, width = images[0].shape
                outputs = [{'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.int64),
                            'scores': torch.empty(0), 'masks': torch.zeros(100, ch, height, width),
                            'keypoints': torch.empty(0, 17, 3), 'keypoints_scores': torch.empty(0, 17)}]
            else:
                outputs = None
        else:
            tail_start_time = time.time()
            outputs = tail_model(*head_output)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            tail_proc_time = time.time() - tail_start_time

        if outputs is not None:
            res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

        tail_proc_time_list.append(tail_proc_time)
        total_proc_time_list.append(head_proc_time + tail_proc_time)

    if not head_only:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    print('{} / {} images were filtered away by head model'.format(filtered_count, len(head_proc_time_list)))
    summarize_inference_time(head_proc_time_list, tail_proc_time_list, total_proc_time_list)


def main(args):
    config = yaml_util.load_yaml_file(args.config)
    if args.json is not None:
        main_util.overwrite_config(config, args.json)

    device = torch.device(args.device)
    print(args)
    model_config = config.get('model', None)
    if args.model_params and model_config is not None:
        model = get_model(model_config, device)
        analyze_model_params(model, args.modules)

    if args.data_size is not None:
        analyze_data_size(config['dataset'], split_name=args.data_size, resized=args.resized)

    student_model_config = config.get('student_model', None)
    if args.bottleneck_size is not None and (student_model_config is not None or
                                             (model_config is not None and 'ext_config' in model_config['backbone'])):
        data_logger = DataLogger()
        tmp_model_config = student_model_config if student_model_config is not None else model_config
        model = get_model(tmp_model_config, device, bottleneck_transformer=data_logger)
        analyze_bottleneck_size(model, data_logger, device, config['dataset'], split_name=args.bottleneck_size)

    if student_model_config is not None:
        model_config = student_model_config

    if args.split_model is not None:
        model = get_model(model_config, device)
        analyze_split_model_inference(model, device, args.quantize, args.skip_tail, config['dataset'], args.split_model)


if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
