import torch

from structure.sampler import GroupedBatchSampler, create_aspect_ratio_groups
from structure.transformer import ToTensor, RandomHorizontalFlip, Compose
from utils import misc_util
from utils.coco_util import get_coco, get_coco_kp


def get_coco_dataset(name, root_dir_path, split_name, is_train):
    transforms = [ToTensor()]
    if is_train:
        transforms.append(RandomHorizontalFlip(0.5))

    paths = {
        'coco': (root_dir_path, get_coco, 91),
        'coco_kp': (root_dir_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]
    ds = ds_fn(p, split_name=split_name, transforms=Compose(transforms))
    return ds, num_classes


def get_coco_data_loaders(dataset_config, batch_size, distributed):
    dataset_name = dataset_config['name']
    root_dir_path = dataset_config['root']
    num_workers = dataset_config['num_workers']
    aspect_ratio_group_factor = dataset_config['aspect_ratio_group_factor']
    dataset_splits = dataset_config['splits']

    train_dataset, num_classes = get_coco_dataset(dataset_name, root_dir_path, dataset_splits['train'], True)
    val_dataset, _ = get_coco_dataset(dataset_name, root_dir_path, dataset_splits['val'], False)
    test_dataset, _ = get_coco_dataset(dataset_name, root_dir_path, dataset_splits['test'], False)

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
    return num_classes, train_sampler, train_data_loader, val_data_loader, test_data_loader
