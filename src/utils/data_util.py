import torch

from structure.sampler import GroupedBatchSampler, create_aspect_ratio_groups
from structure.transformer import ToTensor, RandomHorizontalFlip, Compose
from utils import misc_util
from utils.coco_util import get_coco


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
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    if aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(train_dataset, k=aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                                    num_workers=num_workers, collate_fn=misc_util.collate_fn)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=val_sampler,
                                                  num_workers=num_workers, collate_fn=misc_util.collate_fn)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler,
                                                   num_workers=num_workers, collate_fn=misc_util.collate_fn)
    return train_sampler, train_data_loader, val_data_loader, test_data_loader
