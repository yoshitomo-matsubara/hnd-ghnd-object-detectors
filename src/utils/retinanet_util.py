from torch.utils.data import DataLoader
from torchvision import transforms

from myutils.common import file_util
from structure.datasets import CocoDataset, CSVDataset
from structure.samplers import AspectRatioBasedSampler
from structure.transformers import Resizer, Augmenter, Normalizer
from structure.transformers import collater


def get_datasets(dataset_config):
    dataset_name = dataset_config['name']
    data_config = dataset_config['data']
    if dataset_name == 'coco':
        train_data_config = data_config['train']
        train_dataset = CocoDataset(train_data_config['annotation'], train_data_config['img_dir'],
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        val_data_config = data_config['val']
        valid_dataset = CocoDataset(val_data_config['annotation'], val_data_config['img_dir'],
                                    transform=transforms.Compose([Normalizer(), Resizer()]))
        return train_dataset, valid_dataset
    elif dataset_name == 'csv':
        class_file_path = data_config['class']
        train_dataset = CSVDataset(data_file_path=data_config['train'], class_list=class_file_path,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        val_file_path = data_config['val']
        if not file_util.check_if_exists(val_file_path):
            valid_dataset = None
            print('No validation annotations provided.')
        else:
            valid_dataset = CSVDataset(data_file_path=val_file_path, class_list=class_file_path,
                                       transform=transforms.Compose([Normalizer(), Resizer()]))
        return train_dataset, valid_dataset
    raise ValueError('dataset_name `{}` is not expected'.format(dataset_name))


def get_train_data_loader(train_dataset, batch_size=2, drop_last=False, num_workers=3):
    train_sampler = AspectRatioBasedSampler(train_dataset, batch_size=batch_size, drop_last=drop_last)
    return DataLoader(train_dataset, num_workers=num_workers, collate_fn=collater, batch_sampler=train_sampler)
