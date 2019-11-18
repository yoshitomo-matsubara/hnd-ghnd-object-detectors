import builtins as __builtin__
import json
import os
import time

import torch

from models import get_iou_types
from utils import misc_util
from utils.coco_eval_util import CocoEvaluator
from utils.coco_util import get_coco_api_from_dataset


def overwrite_dict(org_dict, sub_dict):
    for sub_key, sub_value in sub_dict.items():
        if sub_key in org_dict:
            if isinstance(sub_value, dict):
                overwrite_dict(org_dict[sub_key], sub_value)
            else:
                org_dict[sub_key] = sub_value
        else:
            org_dict[sub_key] = sub_value


def overwrite_config(config, json_str):
    overwrite_dict(config, json.loads(json_str))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(world_size=1, dist_url='env://'):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device_id = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        device_id = rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        return False, None

    torch.cuda.set_device(device_id)
    dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(rank, dist_url), flush=True)
    torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url,
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)
    return True, [device_id]


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device('cpu')
    model.eval()
    metric_logger = misc_util.MetricLogger(delimiter='  ')
    header = 'Test:'
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
