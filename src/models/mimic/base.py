from torch import nn


class ExtEncoder(nn.Module):
    def __init__(self, encoder, ext_classifier, threshold):
        super().__init__()
        self.encoder = encoder
        self.ext_classifier = ext_classifier
        self.threshold = threshold

    def forward(self, x):
        z = self.encoder(x)
        if self.ext_classifier is None:
            return z

        ext_z = self.ext_classifier(z)
        if not self.training and ext_z.shape[0] == 1 and ext_z[0][1] < self.threshold:
            return None, ext_z
        return z, ext_z

    def get_ext_classifier(self):
        return self.ext_classifier


class BottleneckBase4Ext(nn.Module):
    def __init__(self, encoder, decoder, bottleneck_transformer=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck_transformer = bottleneck_transformer
        self.uses_ext_encoder = isinstance(encoder, ExtEncoder)

    def forward_ext(self, z):
        z, ext_z = z
        if z is None:
            return z, ext_z
        elif self.bottleneck_transformer is not None:
            z = self.bottleneck_transformer(z)
        return self.decoder(z), ext_z

    def forward(self, x):
        z = self.encoder(x)
        if self.uses_ext_encoder:
            return self.forward_ext(z)
        elif self.bottleneck_transformer is not None:
            z = self.bottleneck_transformer(z)
        return self.decoder(z)

    def get_ext_classifier(self):
        raise NotImplementedError('get_ext_classifier function is not implemented')
