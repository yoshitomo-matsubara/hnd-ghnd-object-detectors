import pickle
import sys
from collections import namedtuple

import torch

from structure.sampler import GroupedBatchSampler, create_aspect_ratio_groups
from structure.transformer import ToTensor, RandomHorizontalFlip, Compose
from utils import misc_util
from utils.coco_util import get_coco

QuantizedTensor = namedtuple('QuantizedTensor', ['tensor', 'scale', 'zero_point'])


def get_coco_dataset(split_dict, is_train):
    transforms = [ToTensor()]
    if is_train:
        transforms.append(RandomHorizontalFlip(0.5))
    return get_coco(img_dir_path=split_dict['images'], ann_file_path=split_dict['annotations'],
                    transforms=Compose(transforms), remove_non_annotated_imgs=split_dict['remove_non_annotated_imgs'],
                    jpeg_quality=split_dict['jpeg_quality'])


def get_coco_data_loaders(dataset_config, batch_size, distributed):
    num_workers = dataset_config['num_workers']
    aspect_ratio_group_factor = dataset_config['aspect_ratio_group_factor']
    dataset_splits = dataset_config['splits']
    train_dataset = get_coco_dataset(dataset_splits['train'], True)
    val_dataset = get_coco_dataset(dataset_splits['val'], False)
    test_dataset = get_coco_dataset(dataset_splits['test'], False)

    print('Creating data loaders')
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                                    num_workers=num_workers, collate_fn=misc_util.collate_fn)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=test_sampler,
                                                  num_workers=num_workers, collate_fn=misc_util.collate_fn)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler,
                                                   num_workers=num_workers, collate_fn=misc_util.collate_fn)
    return train_sampler, train_data_loader, val_data_loader, test_data_loader


# Referred to https://github.com/eladhoffer/utils.pytorch/blob/master/quantize.py
#  and http://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf
def quantize_tensor(x, num_bits=8):
    qmin = 0.0
    qmax = 2.0 ** num_bits - 1.0
    min_val, max_val = x.min(), x.max()
    scale = (max_val - min_val) / (qmax - qmin)
    initial_zero_point = qmin - min_val / scale
    zero_point = qmin if initial_zero_point < qmin else qmax if initial_zero_point > qmax else initial_zero_point
    zero_point = int(zero_point)
    qx = zero_point + x / scale
    qx.clamp_(qmin, qmax).round_()
    qx = qx.round().byte()
    return QuantizedTensor(tensor=qx, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


def compute_data_size(obj, num_bits=8):
    bo = pickle.dumps(obj)
    output_data_size = sys.getsizeof(bo) / 1024
    quantized_output_data_size = None
    if isinstance(obj, torch.Tensor):
        bqo = pickle.dumps(quantize_tensor(obj, num_bits))
        quantized_output_data_size = sys.getsizeof(bqo) / 1024
    return output_data_size, quantized_output_data_size
