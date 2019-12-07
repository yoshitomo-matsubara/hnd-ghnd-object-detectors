import argparse
import os
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional

from models import get_model
from myutils.common import yaml_util
from myutils.pytorch import module_util
from structure.transformer import Compose, ToTensor
from structure.transformer import DataLogger
from utils import coco_util, main_util, misc_util


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


def summarize_data_sizes(data_sizes, title):
    data_sizes = np.array(data_sizes)
    print('[{}]'.format(title))
    print('Data size:\t{:.4f} ± {:.4f} [KB]'.format(data_sizes.mean(), data_sizes.std()))
    print('# Files:\t{}\n'.format(len(data_sizes)))


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

    data_sizes, quantized_data_sizes, tensor_shapes = data_size_logger.get_data()
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
    if quantized_data_sizes[0] is not None:
        summarize_data_sizes(quantized_data_sizes, 'Quantized Bottleneck')
    summarize_tensor_shape(channel_list, height_list, width_list)


def main(args):
    config = yaml_util.load_yaml_file(args.config)
    if args.json is not None:
        main_util.overwrite_config(config, args.json)

    device = torch.device(args.device)
    print(args)
    if args.model_params:
        model = get_model(config['model'], device)
        analyze_model_params(model, args.modules)

    if args.data_size is not None:
        analyze_data_size(config['dataset'], split_name=args.data_size, resized=args.resized)

    if args.bottleneck_size is not None:
        data_logger = DataLogger()
        model = get_model(config['student_model'], device, bottleneck_transformer=data_logger)
        analyze_bottleneck_size(model, data_logger, device, config['dataset'], split_name=args.bottleneck_size)


if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
