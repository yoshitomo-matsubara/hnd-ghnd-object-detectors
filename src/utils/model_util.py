from utils import retinanet_util
from torch.utils.data import DataLoader


def get_data_loaders(dataset_config, model_type, batch_size):
    if model_type == 'retinanet':
        train_dataset, val_dataset = retinanet_util.get_datasets(dataset_config)
        train_data_loader = retinanet_util.get_train_data_loader(train_dataset, batch_size=batch_size)
        val_data_loader = DataLoader(val_dataset, num_workers=3)
        return train_data_loader, val_data_loader
    raise ValueError('model_type `{}` is not expected'.format(model_type))
