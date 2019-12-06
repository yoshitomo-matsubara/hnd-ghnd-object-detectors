from torch import nn

from models.ext.classifier import Ext4ResNet
from models.mimic.base import BottleneckBase4Ext, ExtEncoder


class Bottleneck4SmallResNet(BottleneckBase4Ext):
    def __init__(self, bottleneck_channel, ext_config, bottleneck_transformer):
        encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False)
        )
        decoder = nn.Sequential(
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        encoder = ExtEncoder(encoder, Ext4ResNet(64) if ext_config is not None else None, ext_config)
        super().__init__(encoder=encoder, decoder=decoder, bottleneck_transformer=bottleneck_transformer)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()


class Bottleneck4LargeResNet(BottleneckBase4Ext):
    def __init__(self, bottleneck_channel, ext_config, bottleneck_transformer):
        encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False)
        )
        decoder = nn.Sequential(
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        encoder = ExtEncoder(encoder, Ext4ResNet(64) if ext_config is not None else None, ext_config)
        super().__init__(encoder=encoder, decoder=decoder, bottleneck_transformer=bottleneck_transformer)

    def get_ext_classifier(self):
        return self.encoder.get_ext_classifier()


def get_mimic_layers(backbone_name, backbone_config, bottleneck_transformer=None):
    layer1, layer2, layer3, layer4 = None, None, None, None
    backbone_params_config = backbone_config['params']
    layer1_config = backbone_params_config.get('layer1', None)
    if layer1_config is not None:
        layer1_name = layer1_config['name']
        ext_config = backbone_config.get('ext_config', None)
        if layer1_name == 'Bottleneck4SmallResNet' and backbone_name in {'custom_resnet18', 'custom_resnet34'}:
            layer1 = Bottleneck4LargeResNet(layer1_config['bottleneck_channel'], ext_config, bottleneck_transformer)
        elif layer1_name == 'Bottleneck4LargeResNet'\
                and backbone_name in {'custom_resnet50', 'custom_resnet101', 'custom_resnet152'}:
            layer1 = Bottleneck4LargeResNet(layer1_config['bottleneck_channel'], ext_config, bottleneck_transformer)
        else:
            raise ValueError('layer1_name `{}` is not expected'.format(layer1_name))
    return layer1, layer2, layer3, layer4
