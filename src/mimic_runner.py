import argparse

import torch

from distillation.tool import DistillationBox
from models import load_ckpt, get_model, save_ckpt
from myutils.common import file_util, yaml_util
from myutils.pytorch import func_util, module_util
from utils import data_util, main_util, misc_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Runner')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--device', default='cuda', help='device')
    argparser.add_argument('--json', help='dictionary to overwrite config')
    argparser.add_argument('-distill', action='store_true', help='distill a teacher model')
    # distributed training parameters
    argparser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    argparser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return argparser


def freeze_modules(student_model, student_model_config):
    for student_path in student_model_config['frozen_modules']:
        student_module = module_util.get_module(student_model, student_path)
        module_util.freeze_module_params(student_module)


def distill_model(distillation_box, data_loader, optimizer, lr_scheduler, log_freq, device, epoch):
    metric_logger = misc_util.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc_util.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if epoch == 0:
        warmup_factor = 1.0 / 1000.0
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = main_util.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, log_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = distillation_box(images, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])


def distill(teacher_model, student_model, train_sampler, train_data_loader, val_data_loader,
            device, distributed, distill_backbone_only, config, args):
    train_config = config['train']
    distillation_box = DistillationBox(teacher_model, student_model, train_config['criterion'])
    ckpt_file_path = config['student_model']['ckpt']
    optim_config = train_config['optimizer']
    optimizer = func_util.get_optimizer(student_model, optim_config['type'], optim_config['params'])
    scheduler_config = train_config['scheduler']
    lr_scheduler = func_util.get_scheduler(optimizer, scheduler_config['type'], scheduler_config['params'])
    if file_util.check_if_exists(ckpt_file_path):
        load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    best_val_map = 0.0
    num_epochs = train_config['num_epochs']
    log_freq = train_config['log_freq']
    for epoch in range(1, num_epochs + 1):
        if distributed:
            train_sampler.set_epoch(epoch)

        teacher_model.eval()
        student_model.train()
        teacher_model.distill_backbone_only = distill_backbone_only
        student_model.distill_backbone_only = distill_backbone_only
        distill_model(distillation_box, train_data_loader, optimizer, lr_scheduler, log_freq, device, epoch)
        coco_evaluator = main_util.evaluate(student_model, val_data_loader, device=device)
        # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
        val_map = coco_evaluator.coco_eval['bbox'].stats[0]
        if val_map > best_val_map:
            best_val_map = val_map
            save_ckpt(student_model, optimizer, lr_scheduler, config, args, ckpt_file_path)


def evaluate(teacher_model, student_model, test_data_loader, device):
    teacher_model.distill_backbone_only = False
    student_model.distill_backbone_only = False
    print('[Teacher model]')
    main_util.evaluate(teacher_model, test_data_loader, device=device)
    print('\n[Student model]')
    main_util.evaluate(student_model, test_data_loader, device=device)


def main(args):
    config = yaml_util.load_yaml_file(args.config)
    if args.json is not None:
        main_util.overwrite_config(config, args.json)

    distributed, _ = main_util.init_distributed_mode(args.world_size, args.dist_url)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    teacher_model = get_model(config['teacher_model'], device)
    student_model_config = config['student_model']
    student_model = get_model(student_model_config, device)
    freeze_modules(student_model, student_model_config)
    distill_backbone_only = student_model_config['distill_backbone_only']
    train_config = config['train']
    train_sampler, train_data_loader, val_data_loader, test_data_loader = \
        data_util.get_coco_data_loaders(config['dataset'], train_config['batch_size'], distributed)
    if args.distill:
        distill(teacher_model, student_model, train_sampler, train_data_loader, val_data_loader,
                device, distributed, distill_backbone_only, config, args)
    evaluate(teacher_model, student_model, test_data_loader, device)


if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
