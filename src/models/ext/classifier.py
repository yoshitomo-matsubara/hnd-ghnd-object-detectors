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
    def __init__(self, bottleneck_channel):
        super().__init__(ext_idx=0)
        self.extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((64, 64)),
            nn.Conv2d(bottleneck_channel, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.linear = nn.Linear(32 * 4 * 4, 2)

    def forward(self, x):
        z = self.extractor(x)
        z = self.linear(z.flatten(1))
        return z if self.training else z.softmax(dim=1)


def get_ext_classifier(backbone):
    if isinstance(backbone, (ResNet, CustomResNet)):
        return Ext4ResNet(backbone)
    raise ValueError('type of backbone `{}` is not expected'.format(type(backbone)))
