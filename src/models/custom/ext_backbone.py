from collections import OrderedDict

from torch import nn
from torch.jit.annotations import Dict
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

from models.custom.ext_classifier import get_ext_classifier
from myutils.pytorch import module_util


# Train this class in ways like with/without freeze `model` parameters
# May need to add if statement (like isinstance(features, tuple)) right after features = self.backbone(images.tensors) in CustomRCNN
class ExtIntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __constants__ = ['layers']
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers, ext_idx):
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
        self.ext_layer_name = list(self.return_layers.keys())[ext_idx]
        self.ext_classifier = get_ext_classifier(model)

    def forward(self, x):
        out = OrderedDict()
        z = None
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
                if name == self.ext_layer_name:
                    z = self.ext_classifier(x)
                    # if argmax(z) is negative label, return out, False to stop inference
                    # if not self.training and z.argmax(1) == 0:
                    #     return True or False
        return out, z


class ExtBackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, ext_idx, backbone_frozen=False):
        super().__init__()
        if backbone_frozen:
            module_util.freeze_module_params(backbone)

        self.body = ExtIntermediateLayerGetter(backbone, return_layers=return_layers, ext_idx=ext_idx)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = out_channels

    def forward(self, x):
        z, ext_z = self.body(x)
        if not self.training and ext_z is None:
            return None, None
        return self.fpn(z), ext_z
