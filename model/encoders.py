import timm
import torch.nn as nn


class TimmEncoder(nn.Module):
    def __init__(self, name, pretrained=True, in_channels=3, depth=5, output_stride=32):
        super().__init__()
        self.model = timm.create_model(model_name=name,
                                       in_chans=in_channels,
                                       features_only=True,
                                       pretrained=pretrained,
                                       output_stride=output_stride,
                                       out_indices=tuple(range(depth)))

        self._in_channels = in_channels
        self._out_channels = [in_channels,] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x):
        features = self.model(x)
        features = [x,] + features
        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)
