import argparse
import os

import torch
from PIL import Image

from models import get_model
from myutils.common import file_util, yaml_util
from utils import main_util, visual_util


def get_argparser():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', required=True, help='yaml config file')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('--json', help='dictionary to overwrite config')
    argparser.add_argument('--image', default=True, nargs='+',
                           help='image dir/file paths for visualization (prioritized if given)')
    argparser.add_argument('--output', required=True, help='output dir path')
    return argparser


def predict(model, input_file_path, device, output_dir_path):
    result, output, top_pred = visual_util.predict(model, Image.open(input_file_path).convert('RGB'), device)
    output_file_path = os.path.join(output_dir_path, os.path.basename(input_file_path))
    vis_img = Image.fromarray(result[:, :, [2, 1, 0]])
    vis_img.save(output_file_path)


def visualize_predictions(model, input_file_paths, device, output_dir_path):
    model.eval()
    file_util.make_dirs(output_dir_path)
    for input_file_path in input_file_paths:
        if not file_util.check_if_exists(input_file_path):
            print('`{}` is not found.'.format(input_file_path))
            continue

        if os.path.isfile(input_file_path):
            predict(model, input_file_path, device, output_dir_path)
        else:
            for sub_input_file_path in file_util.get_file_path_list(input_file_path, is_recursive=True):
                predict(model, sub_input_file_path, device, output_dir_path)


def main(args):
    config = yaml_util.load_yaml_file(args.config)
    if args.json is not None:
        main_util.overwrite_config(config, args.json)

    device = torch.device(args.device)
    print(args)
    print('Creating model')
    model_config = config['model']
    model = get_model(model_config, device, strict=False)
    visualize_predictions(model, args.image, device, args.output)


if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
