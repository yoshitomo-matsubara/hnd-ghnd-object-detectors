import argparse

from distillation import hnd, kd
from myutils.common import yaml_util
from utils import main_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Runner')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--method', required=True,
                           help='`kd`: knowledge distillation, `hnd`: head-network distillation')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('--json', help='dictionary to overwrite config')
    argparser.add_argument('-train', action='store_true', help='train a model')
    # distributed training parameters
    argparser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    argparser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    return argparser


def main(args):
    config = yaml_util.load_yaml_file(args.config)
    if args.json is not None:
        main_util.overwrite_config(config, args.json)

    method_type = args.method
    if method_type == 'kd':
        kd.main(config, args)
    elif method_type == 'hnd':
        hnd.main(config, args)
    else:
        raise ValueError('method_type `{}` is not expected'.format(method_type))


if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
