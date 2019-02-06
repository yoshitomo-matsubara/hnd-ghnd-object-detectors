import argparse
import logging

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from myutils.common import file_util, log_util, yaml_util
from myutils.pytorch import func_util
from utils import mimic_util, model_util


def get_argparser():
    argparser = argparse.ArgumentParser(description='Mimic Trainer')
    argparser.add_argument('--config', required=True, help='yaml file path')
    argparser.add_argument('--epoch', type=int, help='epoch (higher priority than config if set)')
    argparser.add_argument('--lr', type=float, help='learning rate (higher priority than config if set)')
    argparser.add_argument('--log', default='./log/', help='log dir path')
    argparser.add_argument('-init', action='store_true', help='initialize checkpoint')
    return argparser


def train(student_model, teacher_model, train_loader, optimizer, criterion, epoch, device, interval):
    logging.info('\nEpoch: %d' % epoch)
    student_model.train()
    teacher_model.eval()
    num_samples = len(train_loader.dataset)
    num_batches = len(train_loader)
    train_loss = 0
    total = 0
    for batch_idx, inputs in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs['img'].to(device).float()
        student_outputs = student_model(inputs)
        teacher_outputs = teacher_model(inputs)
        if isinstance(student_outputs, list) and isinstance(teacher_outputs, list):
            loss = 0
            for student_output, teacher_output in zip(student_outputs, teacher_outputs):
                loss += criterion(student_output, teacher_output)
        else:
            loss = criterion(student_outputs, teacher_outputs)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        batch_size = inputs.size(0)
        total += batch_size
        if batch_idx > 0 and batch_idx % interval == 0:
            logging.info('[{}/{} ({:.0f}%)]\tAvg Loss: {:.6f}'.format(batch_idx * len(inputs), num_samples,
                                                                      100.0 * batch_idx / num_batches,
                                                                      loss.item() / batch_size))


def validate(student_model, teacher_model, val_loader, criterion, device):
    student_model.eval()
    teacher_model.eval()
    val_loss = 0
    total = 0
    with torch.no_grad():
        for inputs in val_loader.dataset:
            inputs = inputs['img'].permute(2, 0, 1).to(device).float().unsqueeze(dim=0)
            student_outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs)
            if isinstance(student_outputs, list) and isinstance(teacher_outputs, list):
                loss = 0
                for student_output, teacher_output in zip(student_outputs, teacher_outputs):
                    loss += criterion(student_output, teacher_output)
            else:
                loss = criterion(student_outputs, teacher_outputs)

            val_loss += loss.item()
            total += inputs.size(0)

    avg_val_loss = val_loss / total
    logging.info('Validation Loss: {:.6f}\tAvg Loss: {:.6f}'.format(val_loss, avg_val_loss))
    return avg_val_loss


def save_ckpt(student_model, epoch, best_avg_loss, ckpt_file_path, teacher_model_type):
    target_model = student_model.module if isinstance(student_model, nn.DataParallel) else student_model
    logging.info('Saving..')
    state = {
        'type': teacher_model_type,
        'model': target_model.state_dict(),
        'epoch': epoch + 1,
        'best_avg_loss': best_avg_loss,
        'student': True
    }
    file_util.make_parent_dirs(ckpt_file_path)
    torch.save(state, ckpt_file_path)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True

    config = yaml_util.load_yaml_file(args.config)
    train_config = config['train']
    log_util.setup_logging(train_config['log'])
    logging.info('CUDA is {}available'.format('' if torch.cuda.is_available() else 'not '))
    logging.info('Device: {}'.format(device))
    input_shape = config['input_shape']
    teacher_model_config = config['teacher_model']
    teacher_model, teacher_model_type = mimic_util.get_teacher_model(teacher_model_config, input_shape, device)
    student_model_config = config['student_model']
    student_model = mimic_util.get_student_model(teacher_model_type, student_model_config)
    student_model = student_model.to(device)
    start_epoch, best_avg_loss = mimic_util.resume_from_ckpt(student_model_config['ckpt'], student_model,
                                                             is_student=True)
    train_loader, val_loader =\
        model_util.get_data_loaders(config['dataset'], teacher_model_type, batch_size=train_config['batch_size'])
    criterion_config = train_config['criterion']
    criterion = func_util.get_loss(criterion_config['type'], criterion_config['params'])
    optim_config = train_config['optimizer']
    if args.lr is not None:
        optim_config['params']['lr'] = args.lr

    optimizer = func_util.get_optimizer(student_model, optim_config['type'], optim_config['params'])
    interval = train_config['interval']
    ckpt_file_path = student_model_config['ckpt']
    end_epoch = start_epoch + train_config['epoch'] if args.epoch is None else start_epoch + args.epoch
    for epoch in range(start_epoch, end_epoch):
        train(student_model, teacher_model, train_loader, optimizer, criterion, epoch, device, interval)
        avg_val_loss = validate(student_model, teacher_model, val_loader, criterion, device)
        if avg_val_loss < best_avg_loss:
            best_avg_loss = avg_val_loss
            save_ckpt(student_model, epoch, best_avg_loss, ckpt_file_path, teacher_model_type)


if __name__ == '__main__':
    parser = get_argparser()
    main(parser.parse_args())
