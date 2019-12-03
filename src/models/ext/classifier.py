from torch import nn
from torchvision.models.resnet import ResNet

from models.custom.resnet import CustomResNet


class BaseExtClassifier(nn.Module):
    def __init__(self, ext_idx):
        super().__init__()
        self.ext_idx = ext_idx

    def forward(self, *args):
        raise NotImplementedError('forward function is not implemented')


class Ext4ResNet(BaseExtClassifier):
    def __init__(self, inplanes):
        super().__init__(ext_idx=0)
        # inplanes = 512 for ResNets-18 and -34, and 2048 for all the other ResNets
        input_channel = inplanes // 8
        self.extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((64, 64)),
            nn.Conv2d(input_channel, input_channel // 4, kernel_size=4, stride=2),
            nn.BatchNorm2d(input_channel // 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(input_channel // 4, input_channel // 8, kernel_size=4, stride=2),
            nn.BatchNorm2d(input_channel // 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.linear = nn.Linear(input_channel // 8 * 4 * 4, 2)

    def forward(self, x):
        z = self.extractor(x)
        z = self.linear(z.flatten(1))
        return z if self.training else z.softmax(dim=1)


def get_ext_classifier(backbone):
    if isinstance(backbone, (ResNet, CustomResNet)):
        return Ext4ResNet(backbone)
    raise ValueError('type of backbone `{}` is not expected'.format(type(backbone)))
