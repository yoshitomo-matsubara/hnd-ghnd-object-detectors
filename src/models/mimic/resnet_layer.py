from torch import nn


class Bottleneck4ResNet50(nn.Module):
    def __init__(self, bottleneck_channel):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, padding=1, bias=False)
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 64, kernel_size=2, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=2, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def get_mimic_layers(backbone_name, backbone_params_config):
    layer1, layer2, layer3, layer4 = None, None, None, None
    layer1_config = backbone_params_config.get('layer1', None)
    if layer1_config is not None:
        layer1_name = layer1_config['name']
        if layer1_name == 'Bottleneck4ResNet50' and backbone_name == 'custom_resnet50':
            layer1 = Bottleneck4ResNet50(layer1_config['bottleneck_channel'])
    return layer1, layer2, layer3, layer4

