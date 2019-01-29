from torch.utils.data import DataLoader

from utils import retinanet_util


def get_model(config, device):
    model_config = config['model']
    model_type = model_config['type']
    ckpt_file_path = model_config['ckpt']
    model_params_config = model_config['params']
    if model_type.startswith('retinanet'):
        return retinanet_util.get_model(device, ckpt_file_path, **model_params_config)
    raise ValueError('teacher_model_type `{}` is not expected'.format(model_type))


def get_data_loaders(dataset_config, model_type, batch_size):
    if model_type.startswith('retinanet'):
        train_dataset, val_dataset = retinanet_util.get_datasets(dataset_config)
        train_data_loader = retinanet_util.get_train_data_loader(train_dataset, batch_size=batch_size)
        val_data_loader = DataLoader(val_dataset, num_workers=3)
        return train_data_loader, val_data_loader
    raise ValueError('model_type `{}` is not expected'.format(model_type))
