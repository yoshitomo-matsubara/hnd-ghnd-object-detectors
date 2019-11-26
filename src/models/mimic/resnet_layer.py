from torch import nn


class Bottleneck4ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 3, kernel_size=2, padding=1, bias=False)
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=2, bias=False),
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
