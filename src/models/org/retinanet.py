import math
import time

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock, Bottleneck

from models.lib.nms.pth_nms import pth_nms
from structure import losses
from structure.anchors import Anchors
from structure.transformers import BBoxTransform, ClipBoxes

MODEL_URL_DICT = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def nms(dets, thresh):
    # Dispatch to either CPU or GPU NMS implementations. Accept dets as tensor
    return pth_nms(dets, thresh)


class PyramidFeatures(nn.Module):
    def __init__(self, c3_size, c4_size, c5_size, feature_size=256):
        super().__init__()

        # upsample C5 to get P5 from the FPN paper
        self.p5_1 = nn.Conv2d(c5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.p5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.p4_1 = nn.Conv2d(c4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.p4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.p3_1 = nn.Conv2d(c3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.p3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.p6 = nn.Conv2d(c5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.p7_1 = nn.ReLU()
        self.p7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):

        c3, c4, c5 = inputs

        p5_x = self.p5_1(c5)
        p5_upsampled_x = self.p5_upsampled(p5_x)
        p5_x = self.p5_2(p5_x)

        p4_x = self.p4_1(c4)
        p4_x = p5_upsampled_x + p4_x
        p4_upsampled_x = self.p4_upsampled(p4_x)
        p4_x = self.p4_2(p4_x)

        p3_x = self.p3_1(c3)
        p3_x = p3_x + p4_upsampled_x
        p3_x = self.p3_2(p3_x)

        p6_x = self.p6(c5)

        p7_x = self.p7_1(p6_x)
        p7_x = self.p7_2(p7_x)

        return [p3_x, p4_x, p5_x, p6_x, p7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super().__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors*4, kernel_size=3, padding=1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.prior = prior

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):

        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


class RetinaNet(nn.Module):
    def __init__(self, backbone, num_classes, block, layers):
        super().__init__()
        self.backbone = backbone

        if block == BasicBlock:
            fpn_sizes = [self.backbone.layer2[layers[1]-1].conv2.out_channels,
                         self.backbone.layer3[layers[2]-1].conv2.out_channels,
                         self.backbone.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.backbone.layer2[layers[1]-1].conv3.out_channels,
                         self.backbone.layer3[layers[2]-1].conv3.out_channels,
                         self.backbone.layer4[layers[3]-1].conv3.out_channels]

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressor = RegressionModel(256)
        self.classifier = ClassificationModel(256, num_classes=num_classes)
        self.anchors = Anchors()
        self.regress_boxes = BBoxTransform()
        self.clip_boxes = ClipBoxes()
        self.focal_loss = losses.FocalLoss()
        self.timestamps_dict = {'start': [], 'end': []}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.classifier.output.weight.data.fill_(0)
        self.classifier.output.bias.data.fill_(-math.log((1.0 - self.classifier.prior) / self.classifier.prior))
        self.regressor.output.weight.data.fill_(0)
        self.regressor.output.bias.data.fill_(0)
        self.freeze_bn()

    def freeze_bn(self):
        # Freeze BatchNorm layers
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def detect(self, x2, x3, x4, anchors, annotations, batch_shape):
        features = self.fpn([x2, x3, x4])
        regression = torch.cat([self.regressor(feature) for feature in features], dim=1)
        classification = torch.cat([self.classifier(feature) for feature in features], dim=1)

        if self.training:
            return self.focal_loss(classification, regression, anchors, annotations)

        transformed_anchors = self.regress_boxes(anchors, regression)
        transformed_anchors = self.clip_boxes(transformed_anchors, batch_shape)
        scores = torch.max(classification, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > 0.05)[0, :, 0]

        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return [torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1, 4).cuda()]

        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]
        anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], 0.5)
        nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
        self.timestamps_dict['end'].append(time.perf_counter())
        return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
            annotations = None

        self.timestamps_dict['start'].append(time.perf_counter())
        x2, x3, x4 = self.backbone(img_batch)
        anchors = self.anchors(img_batch)
        batch_shape = img_batch.shape
        return self.detect(x2, x3, x4, anchors, annotations, batch_shape)


def retinanet18(num_classes, pretrained=False, **kwargs):
    """Constructs a RetinaNet-18 based on a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    backbone = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    retinanet = RetinaNet(backbone, num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        retinanet.backbone.load_state_dict(model_zoo.load_url(MODEL_URL_DICT['resnet18'], model_dir='.'), strict=False)
    return retinanet


def retinanet34(num_classes, pretrained=False, **kwargs):
    """Constructs a RetinaNet-34 based on a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    backbone = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    retinanet = RetinaNet(backbone, num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        retinanet.backbone.load_state_dict(model_zoo.load_url(MODEL_URL_DICT['resnet34'], model_dir='.'), strict=False)
    return retinanet


def retinanet50(num_classes, pretrained=False, **kwargs):
    """Constructs a RetinaNet-50 based on a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    backbone = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    retinanet = RetinaNet(backbone, num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        retinanet.backbone.load_state_dict(model_zoo.load_url(MODEL_URL_DICT['resnet50'], model_dir='.'), strict=False)
    return retinanet


def retinanet101(num_classes, pretrained=False, **kwargs):
    """Constructs a RetinaNet-101 based on a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    backbone = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    retinanet = RetinaNet(backbone, num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        retinanet.backbone.load_state_dict(model_zoo.load_url(MODEL_URL_DICT['resnet101'], model_dir='.'), strict=False)
    return retinanet


def retinanet152(num_classes, pretrained=False, **kwargs):
    """Constructs a RetinaNet-152 based on a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    backbone = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    retinanet = RetinaNet(backbone, num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        retinanet.backbone.load_state_dict(model_zoo.load_url(MODEL_URL_DICT['resnet152'], model_dir='.'), strict=False)
    return retinanet


def get_retinanet(depth, **kwargs):
    if depth == 18:
        return retinanet18(**kwargs)
    elif depth == 34:
        return retinanet34(**kwargs)
    elif depth == 50:
        return retinanet50(**kwargs)
    elif depth == 101:
        return retinanet101(**kwargs)
    elif depth == 152:
        return retinanet152(**kwargs)
    else:
        raise ValueError('depth `{}` is not expected'.format(depth))
