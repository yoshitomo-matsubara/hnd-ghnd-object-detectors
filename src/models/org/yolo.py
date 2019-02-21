from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import yolo_util

ONNX_EXPORT = False


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
            upsample = Upsample(scale_factor=int(module_def['stride']), mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            img_height = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, img_height, anchor_idxs, cfg=hyperparams['cfg'])
            modules.add_module('yolo_%d' % i, yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)
    return hyperparams, module_list


def create_conv_seq(num_in_channels, num_out_channels, kernel_size, stride, padding, bias=False):
    return nn.Sequential(
        nn.Conv2d(num_in_channels, num_out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(num_out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.LeakyReLU(negative_slope=0.1)
    )


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class ShortcutBlock(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.first_module = modules[0]
        self.seq = nn.Sequential(*modules[1:])

    def forward(self, x):
        z = self.first_module(x)
        return z + self.seq(z)


class RouteBlock(nn.Module):
    def __init__(self, modules, target_indices):
        super().__init__()
        self.modules = modules
        self.target_idx_set = set(target_indices)

    def forward(self, x):
        z = x
        output_list = list()
        for i, module in enumerate(self.modules):
            z = module(z)
            if i in self.target_idx_set:
                output_list.append(z)
        return torch.cat(output_list, 1)


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break

                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )
    return output


class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim, anchor_idxs, cfg):
        super(YOLOLayer, self).__init__()

        anchors = [(a_w, a_h) for a_w, a_h in anchors]  # (pixels)
        num_anchors = len(anchors)

        self.anchors = anchors
        self.num_anchors = num_anchors  # number of anchors (3)
        self.num_classes = num_classes  # number of classes (80)
        self.bbox_attrs = 5 + num_classes
        self.img_dim = img_dim  # from hyperparams in cfg file, NOT from parser

        if anchor_idxs[0] == (num_anchors * 2):  # 6
            stride = 32
        elif anchor_idxs[0] == num_anchors:  # 3
            stride = 16
        else:
            stride = 8

        if cfg.endswith('yolov3-tiny.cfg'):
            stride *= 2

        # Build anchor grids
        num_grids = int(self.img_dim / stride)  # number grid points
        self.grid_x = torch.arange(num_grids).repeat(num_grids, 1).view([1, 1, num_grids, num_grids]).float()
        self.grid_y = torch.arange(num_grids).repeat(num_grids, 1).t().view([1, 1, num_grids, num_grids]).float()
        self.anchor_wh = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])  # scale anchors
        self.anchor_w = self.anchor_wh[:, 0].view((1, num_anchors, 1, 1))
        self.anchor_h = self.anchor_wh[:, 1].view((1, num_anchors, 1, 1))
        self.weights = yolo_util.class_weights()

        self.loss_means = torch.ones(6)
        self.yolo_layer = anchor_idxs[0] / num_anchors  # 2, 1, 0
        self.stride = stride

        if ONNX_EXPORT:  # use fully populated and reshaped tensors
            self.anchor_w = self.anchor_w.repeat((1, 1, num_grids, num_grids)).view(1, -1, 1)
            self.anchor_h = self.anchor_h.repeat((1, 1, num_grids, num_grids)).view(1, -1, 1)
            self.grid_x = self.grid_x.repeat(1, num_anchors, 1, 1).view(1, -1, 1)
            self.grid_y = self.grid_y.repeat(1, num_anchors, 1, 1).view(1, -1, 1)
            self.grid_xy = torch.cat((self.grid_x, self.grid_y), 2)
            self.anchor_wh = torch.cat((self.anchor_w, self.anchor_h), 2) / num_grids

    def forward(self, p, targets=None, batch_report=False, var=None):
        FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
        bs = p.shape[0]  # batch size
        num_grids = p.shape[2]  # number of grid points

        if p.is_cuda and not self.weights.is_cuda:
            self.grid_x, self.grid_y = self.grid_x.cuda(), self.grid_y.cuda()
            self.anchor_wh, self.anchor_w, self.anchor_h =\
                self.anchor_wh.cuda(), self.anchor_w.cuda(), self.anchor_h.cuda()
            self.weights, self.loss_means = self.weights.cuda(), self.loss_means.cuda()

        # p.view(12, 255, 13, 13) -- > (12, 3, 13, 13, 80)  # (bs, anchors, grid, grid, classes + xywh)
        # prediction
        p = p.view(bs, self.num_anchors, self.bbox_attrs, num_grids, num_grids).permute(0, 1, 3, 4, 2).contiguous()

        # Training
        if targets is not None:
            mse_loss = nn.MSELoss()
            bce_w_logits_loss = nn.BCEWithLogitsLoss()
            cross_entropy_loss = nn.CrossEntropyLoss()

            # Get outputs
            x = torch.sigmoid(p[..., 0])  # Center x
            y = torch.sigmoid(p[..., 1])  # Center y
            p_conf = p[..., 4]  # Conf
            p_cls = p[..., 5:]  # Class

            # Width and height (yolo method)
            w = p[..., 2]  # Width
            h = p[..., 3]  # Height
            width = torch.exp(w.data) * self.anchor_w
            height = torch.exp(h.data) * self.anchor_h

            # Width and height (power method)
            # w = torch.sigmoid(p[..., 2])  # Width
            # h = torch.sigmoid(p[..., 3])  # Height
            # width = ((w.data * 2) ** 2) * self.anchor_w
            # height = ((h.data * 2) ** 2) * self.anchor_h

            p_boxes = None
            if batch_report:
                # Predicted boxes: add offset and scale with anchors (in grid space, i.e. 0-13)
                gx = x.data + self.grid_x[:, :, :num_grids, :num_grids]
                gy = y.data + self.grid_y[:, :, :num_grids, :num_grids]
                p_boxes = torch.stack((gx - width / 2,
                                       gy - height / 2,
                                       gx + width / 2,
                                       gy + height / 2), 4)  # x1y1x2y2

            tx, ty, tw, th, mask, tcls, num_tps, num_fps, num_fns, target_categories = \
                yolo_util.build_targets(p_boxes, p_conf, p_cls, targets, self.anchor_wh, self.num_anchors,
                                        self.num_classes, num_grids, batch_report)

            tcls = tcls[mask]
            if x.is_cuda:
                tx, ty, tw, th, mask, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), mask.cuda(), tcls.cuda()

            # Compute losses
            num_targets = sum([len(x) for x in targets])  # number of targets
            num_masks = mask.sum().float()  # number of anchors (assigned to targets)
            batch_size = len(targets)  # batch size
            k = num_masks / batch_size
            if num_masks > 0:
                lx = k * mse_loss(x[mask], tx[mask])
                ly = k * mse_loss(y[mask], ty[mask])
                lw = k * mse_loss(w[mask], tw[mask])
                lh = k * mse_loss(h[mask], th[mask])

                lcls = (k / 4) * cross_entropy_loss(p_cls[mask], torch.argmax(tcls, 1))
                # lcls = (k * 10) * bce_w_logits_loss(p_cls[mask], tcls.float())
            else:
                lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])

            lconf = (k * 64) * bce_w_logits_loss(p_conf, mask.float())

            # Sum loss components
            balance_losses_flag = False
            if balance_losses_flag:
                k = 1 / self.loss_means.clone()
                loss = (lx * k[0] + ly * k[1] + lw * k[2] + lh * k[3] + lconf * k[4] + lcls * k[5]) / k.mean()

                self.loss_means = self.loss_means * 0.99 + \
                                  FT([lx.data, ly.data, lw.data, lh.data, lconf.data, lcls.data]) * 0.01
            else:
                loss = lx + ly + lw + lh + lconf + lcls

            # Sum False Positives from unassigned anchors
            num_extra_fps = torch.zeros(self.num_classes)
            if batch_report:
                i = torch.sigmoid(p_conf[~mask]) > 0.5
                if i.sum() > 0:
                    fp_classes = torch.argmax(p_cls[~mask][i], 1)
                    num_extra_fps = torch.bincount(fp_classes, minlength=self.num_classes).float().cpu()  # extra FPs

            return loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(),\
                   num_targets, num_tps, num_fps, num_extra_fps, num_fns, target_categories

        if ONNX_EXPORT:
            p = p.view(1, -1, 85)
            xy = torch.sigmoid(p[..., 0:2]) + self.grid_xy  # x, y
            width_height = torch.exp(p[..., 2:4]) * self.anchor_wh  # width, height
            p_conf = torch.sigmoid(p[..., 4:5])  # Conf
            p_cls = p[..., 5:85]

            # Broadcasting only supported on first dimension in CoreML. See onnx-coreml/_operators.py
            # p_cls = F.softmax(p_cls, 2) * p_conf  # SSD-like conf
            p_cls = torch.exp(p_cls).permute(2, 1, 0)
            p_cls = p_cls / p_cls.sum(0).unsqueeze(0) * p_conf.permute(2, 1, 0)  # F.softmax() equivalent
            p_cls = p_cls.permute(2, 1, 0)

            return torch.cat((xy / num_grids, width_height, p_conf, p_cls), 2).squeeze().t()

        p[..., 0] = torch.sigmoid(p[..., 0]) + self.grid_x  # x
        p[..., 1] = torch.sigmoid(p[..., 1]) + self.grid_y  # y
        p[..., 2] = torch.exp(p[..., 2]) * self.anchor_w  # width
        p[..., 3] = torch.exp(p[..., 3]) * self.anchor_h  # height
        p[..., 4] = torch.sigmoid(p[..., 4])  # p_conf
        p[..., :4] *= self.stride

        # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
        return p.view(bs, -1, 5 + self.num_classes)


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_path, img_size=416):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_config(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'nT', 'TP', 'FP', 'FPe', 'FN', 'TC']

    def forward(self, x, targets=None, batch_report=False, var=0):
        self.losses = defaultdict(float)
        is_training = targets is not None
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets, batch_report, var)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            if batch_report:
                self.losses['TC'] /= 3  # target category
                metrics = torch.zeros(3, len(self.losses['FPe']))  # TP, FP, FN

                ui = np.unique(self.losses['TC'])[1:]
                for i in ui:
                    j = self.losses['TC'] == float(i)
                    metrics[0, i] = (self.losses['TP'][j] > 0).sum().float()  # TP
                    metrics[1, i] = (self.losses['FP'][j] > 0).sum().float()  # FP
                    metrics[2, i] = (self.losses['FN'][j] == 3).sum().float()  # FN
                metrics[1] += self.losses['FPe']

                self.losses['TP'] = metrics[0].sum()
                self.losses['FP'] = metrics[1].sum()
                self.losses['FN'] = metrics[2].sum()
                self.losses['metrics'] = metrics
            else:
                self.losses['TP'] = 0
                self.losses['FP'] = 0
                self.losses['FN'] = 0

            self.losses['nT'] /= 3
            self.losses['TC'] = 0

        if ONNX_EXPORT:
            # Produce a single-layer *.onnx model (upsample ops not working in PyTorch 1.0 export yet)
            output = output[0]  # first layer reshaped to 85 x 507
            return output[5:85].t(), output[:4].t()  # ONNX scores, boxes

        return sum(output) if is_training else torch.cat(output, 1)


def load_weights(self, weights_path, cutoff=-1):
    # Parses and loads the weights stored in 'weights_path'
    # @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)

    if weights_path.endswith('darknet53.conv.74'):
        cutoff = 75
    elif weights_path.endswith('yolov3-tiny.conv.15'):
        cutoff = 16

    # Open the weights file
    fp = open(weights_path, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

    # Needed to write header when saving weights
    self.header_info = header

    self.seen = header[3]
    weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
    fp.close()

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w


"""
    @:param path    - path of the new weights file
    @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
"""


def save_weights(self, path, cutoff=-1):
    fp = open(path, 'wb')
    self.header_info[3] = self.seen
    self.header_info.tofile(fp)

    # Iterate through layers
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            # If batch norm, load bn first
            if module_def['batch_normalize']:
                bn_layer = module[1]
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
            # Load conv bias
            else:
                conv_layer.bias.data.cpu().numpy().tofile(fp)
            # Load conv weights
            conv_layer.weight.data.cpu().numpy().tofile(fp)
    fp.close()


def create_deep_shortcut_block(first_seq, first_params, second_params, depth):
    seq = first_seq
    for _ in range(depth):
        seq = ShortcutBlock(seq, create_conv_seq(*first_params), create_conv_seq(*second_params))
    return seq


class FirstRouteBlock(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        shortcut12 = ShortcutBlock(
            create_conv_seq(256, 512, 3, 2, 1), create_conv_seq(512, 256, 1, 1, 0), create_conv_seq(256, 512, 3, 1, 1)
        )
        self.shortcut19 = create_deep_shortcut_block(shortcut12, (512, 256, 1, 1, 0), (256, 512, 3, 1, 1), depth=7)
        shortcut20 = ShortcutBlock(
            create_conv_seq(512, 1024, 3, 2, 1), create_conv_seq(1024, 512, 1, 1, 0), create_conv_seq(512, 1024, 3, 1, 1)
        )
        self.shortcut23 = create_deep_shortcut_block(shortcut20, (1024, 512, 1, 1, 0), (512, 1024, 3, 1, 1), depth=3)
        tmp_module_list = list()
        for i in range(5):
            tmp_params = (1024, 512, 1, 1, 0) if i % 2 == 0 else (512, 1024, 3, 1, 1)
            tmp_module_list.append(create_conv_seq(*tmp_params))

        self.seq_of_conv_seqs1 = nn.Sequential(*tmp_module_list)

        self.seq4yolo_layer1 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1), nn.Conv2d(1024, 255, kernel_size=1, stride=1, bias=False)
        )
        anchor_idxs = [int(x) for x in [6, 7, 8]]
        anchors = [float(x) for x in [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]]
        anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in anchor_idxs]
        self.yolo_layer1 = YOLOLayer(anchors, 80, img_size, anchor_idxs, cfg='yolov3.cfg')
        self.upsample_seq1 = nn.Sequential(
            create_conv_seq(512, 256, 1, 1, 0),
            Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, targets, batch_report, var):
        z1 = self.shortcut19(x)
        z2 = self.shortcut23(z1)
        z3 = self.seq_of_conv_seqs1(z2)
        z4 = self.upsample_seq1(z3)
        z5 = self.seq4yolo_layer1(z3)
        z6 = self.yolo_layer1(z5, targets, batch_report, var)
        return [torch.cat([z1, z4], 1), z6]


class SecondRouteBlock(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        shortcut4 = ShortcutBlock(
            create_conv_seq(128, 256, 3, 2, 1), create_conv_seq(256, 128, 1, 1, 0), create_conv_seq(128, 256, 3, 1, 1)
        )
        self.shortcut11 = create_deep_shortcut_block(shortcut4, (256, 128, 1, 1, 0), (128, 256, 3, 1, 1), depth=7)
        self.route1 = FirstRouteBlock(img_size)
        tmp_module_list = list()
        for i in range(5):
            tmp_params = (768, 256, 1, 1, 0) if i == 0 else (512, 256, 1, 1, 0) if i % 2 == 0 else (256, 512, 3, 1, 1)
            tmp_module_list.append(create_conv_seq(*tmp_params))

        self.seq_of_conv_seqs2 = nn.Sequential(*tmp_module_list)
        self.seq4yolo_layer2 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Conv2d(512, 255, kernel_size=1, stride=1, bias=False)
        )
        anchor_idxs = [int(x) for x in [3, 4, 5]]
        anchors = [float(x) for x in [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]]
        anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in anchor_idxs]
        self.yolo_layer2 = YOLOLayer(anchors, 80, img_size, anchor_idxs, cfg='yolov3.cfg')
        self.upsample_seq2 = nn.Sequential(
            create_conv_seq(256, 128, 1, 1, 0),
            Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, targets, batch_report, var):
        z1 = self.shortcut11(x)
        z2, yolo_output1 = self.route1(z1, targets, batch_report, var)
        z3 = self.seq_of_conv_seqs2(z2)
        z4 = self.upsample_seq2(z3)
        z5 = self.seq4yolo_layer2(z3)
        yolo_output2 = self.yolo_layer2(z5, targets, batch_report, var)
        return [torch.cat([z1, z4], 1), yolo_output1, yolo_output2]


class YoloV3(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, img_size=416, conf_threshold=0.7, nms_threshold=0.45):
        super().__init__()
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.loss_names = ['loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'nT', 'TP', 'FP', 'FPe', 'FN', 'TC']
        self.loss_dict = None
        self.first_conv_seq = create_conv_seq(3, 32, 3, 1, 1)
        self.shortcut1 = ShortcutBlock(
            create_conv_seq(32, 64, 3, 2, 1), create_conv_seq(64, 32, 1, 1, 0), create_conv_seq(32, 64, 3, 1, 1)
        )
        shortcut2 = ShortcutBlock(
            create_conv_seq(64, 128, 3, 2, 1), create_conv_seq(128, 64, 1, 1, 0), create_conv_seq(64, 128, 3, 1, 1)
        )
        self.shortcut3 = ShortcutBlock(shortcut2, create_conv_seq(128, 64, 1, 1, 0), create_conv_seq(64, 128, 3, 1, 1))
        self.route2 = SecondRouteBlock(img_size)
        tmp_module_list = list()
        for i in range(6):
            tmp_params = (384, 128, 1, 1, 0) if i == 0 else (256, 128, 1, 1, 0) if i % 2 == 0 else (128, 256, 3, 1, 1)
            tmp_module_list.append(create_conv_seq(*tmp_params))

        self.seq_of_conv_seqs3 = nn.Sequential(*tmp_module_list)
        self.last_conv = nn.Conv2d(256, 255, kernel_size=1, stride=1)
        anchor_idxs = [int(x) for x in [0, 1, 2]]
        anchors = [float(x) for x in [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]]
        anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in anchor_idxs]
        self.yolo_layer3 = YOLOLayer(anchors, 80, img_size, anchor_idxs, cfg='yolov3.cfg')

    def forward(self, x, targets=None, batch_report=False, var=0):
        self.loss_dict = defaultdict(float)
        is_training = targets is not None
        z = self.first_conv_seq(x)
        z = self.shortcut1(z)
        z = self.shortcut3(z)
        z, yolo_output1, yolo_output2 = self.route2(z, targets, batch_report, var)
        z = self.seq_of_conv_seqs3(z)
        z = self.last_conv(z)
        yolo_output3 = self.yolo_layer3(z, targets, batch_report, var)
        output_list = list()
        for output in [yolo_output1, yolo_output2, yolo_output3]:
            loss, *misc_losses = output
            output_list.append(loss)
            for name, misc_loss in zip(self.loss_names, misc_losses):
                self.loss_dict[name] += misc_loss

        if is_training:
            if batch_report:
                self.loss_dict['TC'] /= 3  # target category
                metrics = torch.zeros(3, len(self.loss_dict['FPe']))  # TP, FP, FN
                ui = np.unique(self.loss_dict['TC'])[1:]
                for i in ui:
                    j = self.loss_dict['TC'] == float(i)
                    metrics[0, i] = (self.loss_dict['TP'][j] > 0).sum().float()  # TP
                    metrics[1, i] = (self.loss_dict['FP'][j] > 0).sum().float()  # FP
                    metrics[2, i] = (self.loss_dict['FN'][j] == 3).sum().float()  # FN

                metrics[1] += self.loss_dict['FPe']
                self.loss_dict['TP'] = metrics[0].sum()
                self.loss_dict['FP'] = metrics[1].sum()
                self.loss_dict['FN'] = metrics[2].sum()
                self.loss_dict['metrics'] = metrics
            else:
                self.loss_dict['TP'] = 0
                self.loss_dict['FP'] = 0
                self.loss_dict['FN'] = 0

            self.loss_dict['nT'] /= 3
            self.loss_dict['TC'] = 0

        if ONNX_EXPORT:
            # Produce a single-layer *.onnx model (upsample ops not working in PyTorch 1.0 export yet)
            output = output_list[0]  # first layer reshaped to 85 x 507
            return output[5:85].t(), output[:4].t()  # ONNX scores, boxes

        return sum(output_list) if is_training\
            else non_max_suppression(torch.cat(output_list, 1), 80, self.conf_threshold, self.nms_threshold)
