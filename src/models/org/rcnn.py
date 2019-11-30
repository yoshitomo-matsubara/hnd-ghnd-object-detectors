import random
from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_boxes, resize_keypoints
from torchvision.models.utils import load_state_dict_from_url
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import misc as misc_nn_ops

from models import custom
from models.ext import get_ext_fpn_backbone
from models.ext.backbone import ExtIntermediateLayerGetter
from models.mimic.resnet_layer import get_mimic_layers


class CustomRCNNTransform(GeneralizedRCNNTransform):
    def __init__(self, min_size, max_size, image_mean, image_std):
        super().__init__(min_size, max_size, image_mean, image_std)

    def resize(self, image, target, fixed_size=None):
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))
        if fixed_size is not None:
            size = fixed_size
        elif self.training:
            size = random.choice(self.min_size)
        else:
            # FIXME assume for now that testing uses the largest scale
            size = self.min_size[-1]
        scale_factor = size / min_size
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "masks" in target:
            mask = target["masks"]
            mask = misc_nn_ops.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
            target["masks"] = mask

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        return image, target

    def forward(self, images, targets=None, fixed_sizes=None):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else targets
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)
            image, target = self.resize(image, target, fixed_sizes[i] if fixed_sizes is not None else None)
            images[i] = image
            if targets is not None:
                targets[i] = target

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_list = ImageList(images, image_sizes)
        return image_list, targets


class CustomRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transform):
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.ext_training = False
        self.distill_backbone_only = False

    def forward(self, images, targets=None, fixed_sizes=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets, fixed_sizes)
        features = self.backbone(images.tensors)
        if self.distill_backbone_only:
            return features

        loss_dict = dict()
        if isinstance(self.backbone.body, ExtIntermediateLayerGetter):
            features, ext_logits = features
            if self.ext_training:
                return ext_logits
            elif not self.training and features is None:
                pred_dict = {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.int64),
                             'scores': torch.empty(0), 'keypoints': torch.empty(0, 17, 3),
                             'keypoints_scores': torch.empty(0, 17)}
                return [pred_dict]

        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        if self.training:
            loss_dict.update(detector_losses)
            loss_dict.update(proposal_losses)
            return loss_dict
        return detections


class FasterRCNN(CustomRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]

        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform = CustomRCNNTransform(min_size, max_size, image_mean, image_std)
        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


class MaskRCNN(FasterRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # Mask parameters
                 mask_roi_pool=None, mask_head=None, mask_predictor=None):

        assert isinstance(mask_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if mask_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")

        out_channels = backbone.out_channels

        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=14,
                sampling_ratio=2)

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                               mask_dim_reduced, num_classes)

        super(MaskRCNN, self).__init__(
            backbone, num_classes,
            # transform parameters
            min_size, max_size,
            image_mean, image_std,
            # RPN-specific parameters
            rpn_anchor_generator, rpn_head,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            # Box parameters
            box_roi_pool, box_head, box_predictor,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights)

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor


class KeypointRCNN(FasterRCNN):
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=None, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # keypoint parameters
                 keypoint_roi_pool=None, keypoint_head=None, keypoint_predictor=None,
                 num_keypoints=17):

        assert isinstance(keypoint_roi_pool, (MultiScaleRoIAlign, type(None)))
        if min_size is None:
            min_size = (640, 672, 704, 736, 768, 800)

        if num_classes is not None:
            if keypoint_predictor is not None:
                raise ValueError("num_classes should be None when keypoint_predictor is specified")

        out_channels = backbone.out_channels

        if keypoint_roi_pool is None:
            keypoint_roi_pool = MultiScaleRoIAlign(
                featmap_names=[0, 1, 2, 3],
                output_size=14,
                sampling_ratio=2)

        if keypoint_head is None:
            keypoint_layers = tuple(512 for _ in range(8))
            keypoint_head = KeypointRCNNHeads(out_channels, keypoint_layers)

        if keypoint_predictor is None:
            keypoint_dim_reduced = 512  # == keypoint_layers[-1]
            keypoint_predictor = KeypointRCNNPredictor(keypoint_dim_reduced, num_keypoints)

        super(KeypointRCNN, self).__init__(
            backbone, num_classes,
            # transform parameters
            min_size, max_size,
            image_mean, image_std,
            # RPN-specific parameters
            rpn_anchor_generator, rpn_head,
            rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            # Box parameters
            box_roi_pool, box_head, box_predictor,
            box_score_thresh, box_nms_thresh, box_detections_per_img,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights)

        self.roi_heads.keypoint_roi_pool = keypoint_roi_pool
        self.roi_heads.keypoint_head = keypoint_head
        self.roi_heads.keypoint_predictor = keypoint_predictor


MODEL_URL_DICT = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
    'keypointrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth'
}

MODEL_CLASS_DICT = {
    'faster_rcnn': (FasterRCNN, 'fasterrcnn_resnet50_fpn_coco'),
    'mask_rcnn': (MaskRCNN, 'maskrcnn_resnet50_fpn_coco'),
    'keypoint_rcnn': (KeypointRCNN, 'keypointrcnn_resnet50_fpn_coco')
}


def get_base_backbone(backbone_name, backbone_params_config):
    pretrained = backbone_params_config['pretrained']
    if backbone_name.startswith('resnet'):
        return resnet.__dict__[backbone_name](pretrained=pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    elif backbone_name.startswith('custom_resnet'):
        layer1, layer2, layer3, layer4 = get_mimic_layers(backbone_name, backbone_params_config)
        return custom.resnet.__dict__[backbone_name](pretrained=pretrained, norm_layer=misc_nn_ops.FrozenBatchNorm2d,
                                                     layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4)
    raise ValueError('backbone_name `{}` is not expected'.format(backbone_name))


def get_fpn_backbone(backbone, freeze_layers):
    if freeze_layers:
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)


def get_model_config(model_name):
    if model_name in MODEL_CLASS_DICT:
        return MODEL_CLASS_DICT[model_name]
    raise KeyError('model_name `{}` is not expected'.format(model_name))


def get_model(model_name, pretrained, num_classes=91, backbone_config=None,
              custom_backbone=None, strict=True, progress=True, **kwargs):
    backbone_name = backbone_config['name']
    backbone_params_config = backbone_config['params']
    if pretrained:
        backbone_params_config['pretrained'] = False

    if custom_backbone is None:
        base_backbone = get_base_backbone(backbone_name, backbone_params_config)
        ext_config = backbone_config.get('ext_config', None)
        freeze_layers = backbone_params_config['freeze_layers']
        if ext_config is not None:
            backbone = get_ext_fpn_backbone(base_backbone, ext_config, freeze_layers)
            strict = False
        else:
            backbone = get_fpn_backbone(base_backbone, freeze_layers)
    else:
        backbone = custom_backbone

    model_class, pretrained_key = get_model_config(model_name)
    model = model_class(backbone, num_classes, **kwargs)
    if pretrained and backbone_name.endswith('resnet50'):
        if backbone_name != 'resnet50':
            strict = False

        state_dict = load_state_dict_from_url(MODEL_URL_DICT[pretrained_key], progress=progress)
        model.load_state_dict(state_dict, strict=strict)
    return model
