import os
import pickle
import numpy as np
import torch
import torch.nn as nn

from config import CFG


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def check_input_shape(h, w, encoder):
    output_stride = encoder.output_stride
    if h % output_stride != 0 or w % output_stride != 0:
        raise RuntimeError(
            f"Wrong input shape ({h}, {w}). "
            f"Expected image height and width divisible by {output_stride}."
        )


def init_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def init_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def save_model(model, epoch, optimizer, loss, val_loss, filename):
    save_dir = os.path.join("experiments", CFG.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # Save training config as pickle object
        pickle.dump(CFG, open(os.path.join(save_dir, 'config.config'), 'wb'))

    path = os.path.join(save_dir, filename)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'loss': loss if loss is not None else None,
        'val_loss': val_loss if val_loss is not None else None,
        'epoch': epoch if epoch is not None else None,
        'img_size': CFG.img_size,
        'num_classes': CFG.num_classes
    }, path)


def load_model(model, path, optimizer=None):
    checkpoint = torch.load(path, map_location=CFG.device)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()
    return model, optimizer, epoch, loss
