from collections import OrderedDict

import torch
from torch import nn
from torch.jit.annotations import List, Optional, Dict
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, concat_box_prediction_layers

from models.ext.backbone import ExtIntermediateLayerGetter, ExtBackboneWithFPN
from structure.transformer import Compose, Quantizer, Dequantizer


class RcnnHead(nn.Module):
    def __init__(self, rcnn_model, bottleneck_transformer=None):
        super().__init__()
        backbone = rcnn_model.backbone
        self.transform = rcnn_model.transform
        self.layer0 = nn.Sequential(backbone.body.conv1, backbone.body.bn1, backbone.body.relu, backbone.body.maxpool)
        self.layer1_encoder = backbone.body.layer1.encoder
        self.bottleneck_transformer = bottleneck_transformer
        del backbone.body.conv1, backbone.body.bn1, backbone.body.relu, backbone.body.maxpool

    def forward(self, images, targets=None):
        # Keep transform inside the head just to make input of forward function simple
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        z = self.layer0(images.tensors)
        z = self.layer1_encoder(z)
        if self.layer1_encoder.ext_classifier is not None:
            z, ext_z = z
            if z is None:
                # Stop inference since it is decided that there is no object we are interested in
                return None

        if self.bottleneck_transformer is not None:
            z, _ = self.bottleneck_transformer(z, targets)
        return z, images.tensors.shape, images.image_sizes, original_image_sizes


class ModifiedIntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __constants__ = ['layers']
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, layer1_output):
        out = OrderedDict()
        out[0] = layer1_output
        x = layer1_output
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class ModifiedAnchorGenerator(AnchorGenerator):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    def __init__(
        self,
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
    ):
        super().__init__(sizes, aspect_ratios)
        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)
        assert len(sizes) == len(aspect_ratios)

    def forward(self, image_sizes, tensors_shape, feature_maps):
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        image_size = tensors_shape[-2:]
        strides = tuple((image_size[0] / g[0], image_size[1] / g[1]) for g in grid_sizes)
        self.set_cell_anchors(feature_maps[0].device)
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        for i, (image_height, image_width) in enumerate(image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors


class ModifiedRegionProposalNetwork(RegionProposalNetwork):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self,
                 anchor_generator,
                 head,
                 #
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 #
                 pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__(anchor_generator, head, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction,
                         pre_nms_top_n, post_nms_top_n, nms_thresh)

    def forward(self, image_sizes, tensors_shape, features, targets=None):
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(image_sizes, tensors_shape, features)

        num_images = len(anchors)
        num_anchors_per_level = [o[0].numel() for o in objectness]
        objectness, pred_bbox_deltas = \
            concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses


class RcnnTail(nn.Module):
    def __init__(self, rcnn_model, bottleneck_transformer=None):
        super().__init__()
        self.bottleneck_transformer = bottleneck_transformer
        self.layer1_decoder = rcnn_model.backbone.body.layer1.decoder
        del rcnn_model.backbone.body.layer1
        backbone = rcnn_model.backbone
        if isinstance(backbone, ExtBackboneWithFPN):
            backbone.split = True

        backbone.body.return_layers.pop('layer1')
        backbone.body = ModifiedIntermediateLayerGetter(backbone.body, backbone.body.return_layers)
        self.sub_backbone = backbone
        # Anchor Generator and RPN do not use tensors of images, thus they are modified so that we can split RCNN
        rpn = rcnn_model.rpn
        anchor_generator = ModifiedAnchorGenerator(rpn.anchor_generator.sizes, rpn.anchor_generator.aspect_ratios)
        self.rpn =\
            ModifiedRegionProposalNetwork(anchor_generator, rpn.head, rpn.proposal_matcher.high_threshold,
                                          rpn.proposal_matcher.low_threshold, rpn.fg_bg_sampler.batch_size_per_image,
                                          rpn.fg_bg_sampler.positive_fraction, rpn._pre_nms_top_n, rpn._post_nms_top_n,
                                          rpn.nms_thresh)
        self.roi_heads = rcnn_model.roi_heads
        self.transform = rcnn_model.transform

    def forward(self, z, tensors_shape, image_sizes, original_image_sizes, targets=None):
        if self.bottleneck_transformer is not None:
            z, _ = self.bottleneck_transformer(z, targets)

        layer1_feature = self.layer1_decoder(z)
        features = self.sub_backbone(layer1_feature)
        loss_dict = dict()
        if isinstance(self.sub_backbone.body, ExtIntermediateLayerGetter):
            sub_features, ext_logits = features
            if not self.training and sub_features is None:
                height, width = image_sizes[0]
                pred_dict = {'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.int64),
                             'scores': torch.empty(0), 'masks': torch.zeros(100, 3, height, width),
                             'keypoints': torch.empty(0, 17, 3), 'keypoints_scores': torch.empty(0, 17)}
                return [pred_dict]

        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])

        proposals, proposal_losses = self.rpn(image_sizes, tensors_shape, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, image_sizes, targets)
        detections = self.transform.postprocess(detections, image_sizes, original_image_sizes)
        if self.training:
            loss_dict.update(detector_losses)
            loss_dict.update(proposal_losses)
            return loss_dict
        return detections


def split_rcnn_model(model, quantization):
    encoder_transformer = None if quantization is None else Compose([Quantizer(num_bits=quantization)])
    decoder_transformer = None if quantization is None else Compose([Dequantizer(num_bits=quantization)])
    head_model = RcnnHead(model, bottleneck_transformer=encoder_transformer)
    tail_model = RcnnTail(model, bottleneck_transformer=decoder_transformer)
    del model
    return head_model, tail_model
