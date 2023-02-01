import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from openvino.runtime import Core

from config import CFG


def setup_criterion():
    BCELoss = nn.BCEWithLogitsLoss()

    def criterion(y_pred, y_true):
        return BCELoss(y_pred, y_true)

    return criterion


def setup_optimizer(model):
    if CFG.optimizer == "Adam":
        optim.Adam(model.parameters(),
                   lr=CFG.lr,
                   betas=(0.9, 0.999),
                   eps=1e-08,
                   weight_decay=CFG.wd)
    elif CFG.optimizer == "Adamax":
        return optim.Adamax(model.parameters(),
                            lr=CFG.lr,
                            betas=(0.9, 0.999),
                            eps=1e-08,
                            weight_decay=CFG.wd)
    elif CFG.optimizer == "RMSProp":
        return optim.RMSprop(model.parameters(),
                             lr=CFG.lr,
                             alpha=0.99,
                             eps=1e-08,
                             weight_decay=CFG.wd,
                             momentum=CFG.momentum)
    elif CFG.optimizer == "SGD":
        return optim.SGD(model.parameters(),
                         lr=CFG.lr,
                         momentum=CFG.momentum,
                         weight_decay=CFG.wd)
    else:
        raise NotImplementedError(f"Optimizer '{CFG.optimizer}' is not implemented!")


def setup_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max,
                                                   eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0,
                                                             eta_min=CFG.min_lr)
    elif CFG.scheduler == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG.min_lr,)
    elif CFG.scheduler == 'ExponentialLR':
        return lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG.scheduler == None:
        return None
    else:
        raise NotImplementedError(f"Scheduer '{CFG.scheduler}' is not implemented!")


def init_openvino_session(path):
    ie = Core()
    model = ie.read_model(model=path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    # Get input and output layers.
    output_layer = compiled_model.output(0)
    return compiled_model, output_layer
