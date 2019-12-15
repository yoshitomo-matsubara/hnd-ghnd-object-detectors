from collections import OrderedDict

from torch import nn
from torch.jit.annotations import Dict
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from models.mimic.base import BottleneckBase4Ext, ExtEncoder
from myutils.pytorch import module_util


def has_only_empty_bbox(target):
    return all(any(o <= 1 for o in box[2:]) for box in target['boxes'])


def count_visible_keypoints(target):
    return sum(sum(1 for row in kp if row[2] > 0) for kp in target['keypoints'])


def check_if_valid_target(target, min_keypoints_per_image=10):
    # if it's empty, there is no annotation
    if len(target) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if has_only_empty_bbox(target):
        return False
    # keypoints task have a slight different criteria for considering
    # if an annotation is valid
    if 'keypoints' not in target:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if count_visible_keypoints(target) >= min_keypoints_per_image:
        return True
    return False


def check_if_includes_ext(module):
    return isinstance(module, BottleneckBase4Ext) and isinstance(module.encoder, ExtEncoder)


class ExtIntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __constants__ = ['layers']
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers, ext_config):
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
        self.threshold = ext_config['threshold']
        self.ext_training = False

    def get_ext_classifier(self):
        for _, module in self.items():
            if check_if_includes_ext(module):
                return module.get_ext_classifier()
        return None

    def forward(self, x):
        out = OrderedDict()
        ext_x = None
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
                if check_if_includes_ext(module):
                    x, ext_x = x
                    out[out_name] = x
                    if x is None:
                        return None, ext_x
                    elif self.ext_training:
                        return x, ext_x
        return out, ext_x


class ExtBackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, ext_config):
        super().__init__()
        if ext_config.get('backbone_frozen', False):
            module_util.freeze_module_params(backbone)

        self.body = ExtIntermediateLayerGetter(backbone, return_layers=return_layers, ext_config=ext_config)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels
        self.split = False

    def forward(self, x):
        if self.split:
            z = self.body(x)
            return self.fpn(z)

        z, ext_z = self.body(x)
        if (not self.training and z is None) or self.body.ext_training:
            return None, ext_z
        elif self.training:
            return z, ext_z
        return self.fpn(z), ext_z
