import torch
from torch import nn

from config import CFG
from model.encoders import TimmEncoder
from model.decoder import Decoder
from model.utils import init_decoder, init_head, check_input_shape


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None):
        conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        if activation is None:
            activation = nn.Identity()
        elif activation == "sigmoid":
            activation = nn.Sigmoid()
        else:
            raise NotImplementedError(
                f"{activation} activation is not implemented!"
            )
        super().__init__(conv2d, activation)


class UNet(nn.Module):
    def __init__(
        self,
        encoder_name: str = "efficientnet_b0",
        encoder_depth: int = 5,
        encoder_pretrained: bool = True,
        encoder_in_channels: int = 3,
        decoder_use_batchnorm: bool = True,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        activation = None,
        n_classes: int = 1,
    ):
        super().__init__()

        self.encoder = TimmEncoder(
            encoder_name,
            pretrained=encoder_pretrained,
            in_channels=encoder_in_channels,
            depth=encoder_depth
        )

        self.decoder = Decoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
        )
        self.head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=n_classes,
            activation=activation,
            kernel_size=3,
        )

        init_decoder(self.decoder)
        init_head(self.head)

        check_input_shape(CFG.img_size[0], CFG.img_size[1], self.encoder)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.head(decoder_output)

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)
        return x