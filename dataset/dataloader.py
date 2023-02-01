import os
from torch.utils.data import DataLoader

from config import CFG
from dataset.dataset import DatasetBuilder


def prepare_loader(dataset_path, transforms, loader_type="train", debug=False):
    if not os.path.exists(dataset_path):
        raise ValueError(f"Directory '{dataset_path}' is not exists!")

    if loader_type == "train":
        bs = CFG.train_bs
        shuffle = True
        drop_last = True
    elif loader_type == "valid":
        bs = CFG.valid_bs
        shuffle = False
        drop_last = False
    else:
        raise ValueError

    dataset = DatasetBuilder(dataset_path, transforms=transforms)
    loader = DataLoader(dataset, batch_size=bs if not debug else 1,
                        num_workers=4, shuffle=shuffle, pin_memory=True, drop_last=drop_last)
    return loader
