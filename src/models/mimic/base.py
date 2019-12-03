from torch import nn


class ExtDecoder(nn.Module):
    def __init__(self, ext_classifier, decoder, threshold):
        super().__init__()
        self.ext_classifier = ext_classifier
        self.decoder = decoder
        self.threshold = threshold

    def forward(self, x):
        if self.ext_classifier is None:
            return self.decoder(x)

        ext_z = self.ext_classifier(x)
        if not self.training and ext_z.shape[0] == 1 and ext_z[0][1] < self.threshold:
            return None, ext_z
        return self.decoder(x), ext_z

    def get_ext_classifier(self):
        return self.ext_classifier


class BottleneckBase4Ext(nn.Module):
    def __init__(self, encoder, decoder, transformer=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transformer = transformer

    def forward(self, x):
        z = self.encoder(x)
        if self.transformer is not None:
            z = self.transformer(z)
        return self.decoder(z)

    def get_ext_classifier(self):
        raise NotImplementedError('get_ext_classifier function is not implemented')
