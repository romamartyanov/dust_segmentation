import torch


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1, 0))
    return dice


def recall_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=1e-5):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)

    tp = (y_pred * y_true).sum(dim=dim)  # TP
    fn = ((1 - y_pred) * (1 - y_true)).sum(dim=dim)  # FN

    recall = ((tp + epsilon) / (tp + fn + epsilon)).mean(dim=(1, 0))
    return recall


def precision_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)

    tp = torch.sum(y_pred * y_true, dim=dim)  # TP
    fp = torch.sum(y_pred * (1 - y_true), dim=dim)  # FP

    precision = ((tp + epsilon) / (tp + fp + epsilon)).mean(dim=(1, 0))
    return precision


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)

    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)

    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1, 0))
    return iou