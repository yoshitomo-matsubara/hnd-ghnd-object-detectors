import torch

from models import load_ckpt, get_model
from myutils.common import file_util
from myutils.pytorch import func_util
from utils import data_util, main_util


def evaluate(teacher_model, student_model, test_data_loader, device, config):
    teacher_model.eval()
    student_model.eval()


def train(teacher_model, student_model, train_sampler, train_data_loader, val_data_loader, device, config):
    teacher_model.eval()
    student_model.train()
    ckpt_file_path = config['student_model']['ckpt']
    train_config = config['train']
    optim_config = train_config['optimizer']
    optimizer = func_util.get_optimizer(student_model, optim_config['type'], optim_config['params'])
    scheduler_config = train_config['scheduler']
    lr_scheduler = func_util.get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
    if file_util.check_if_exists(ckpt_file_path):
        load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    best_val_map = 0.0
    num_epochs = train_config['num_epochs']


def run(config, args):
    distributed, device_ids = main_util.init_distributed_mode(args.world_size, args.dist_url)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    teacher_model = get_model(config['teacher_model'], device)
    student_model = get_model(config['student_model'], device)
    train_config = config['train']
    train_sampler, train_data_loader, val_data_loader, test_data_loader = \
        data_util.get_coco_data_loaders(config['dataset'], train_config['batch_size'], distributed)
    if args.train:
        train(teacher_model, student_model, train_sampler, train_data_loader, val_data_loader, device, config)
    evaluate(teacher_model, student_model, test_data_loader, device, config)
