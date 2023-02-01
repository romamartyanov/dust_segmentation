import torch


class CFG:
    seed          = 42
    debug         = False # set debug=False for Full Training
    exp_name      = 'test_exp'
    comment       = 'Initial experiment. Unet-efficientnet-b1-224x224'
    model_name    = 'Unet'
    backbone      = 'efficientnet_b1'
    train_data    = 'test_data_tiff__cut/train'
    valid_data    = 'test_data_tiff__cut/valid'
    train_bs      = 8
    valid_bs      = train_bs
    big_img_size  = [2016, 2016]
    img_size      = [224, 224]
    epochs        = 25
    lr            = 1e-3
    optimizer     = 'Adamax'
    scheduler     = 'CosineAnnealingLR'
    min_lr        = 1e-6
    T_max         = int(30000/train_bs*epochs)+10
    T_0           = 25
    warmup_epochs = 0
    wd            = 1e-6
    momentum      = 0
    n_accumulate  = max(1, 32//train_bs)
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes   = 1
    class_labels  = {
        "dust": [250, 50, 83]
    }


if len(CFG.class_labels) != CFG.num_classes:
    raise ValueError("len(CFG.classes) != CFG.num_classes")
if CFG.big_img_size[0] % CFG.img_size[0] != 0 and CFG.big_img_size[1] % CFG.img_size[1] != 0:
        raise ValueError("Image size is not divisible by piece size!")