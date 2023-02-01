import os
import random
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from config import CFG


def seed_everything(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('\n> EVERTHING IS SEEDED\n')


def save_metrics_plot(history, epochs):
    x = np.linspace(1, epochs, epochs).astype(int)

    f = plt.figure(figsize=(15, 40))
    for i, (key, value) in enumerate(history.items()):
        ax = f.add_subplot(len(history)+1, 1, i+1)
        ax.plot(x, value)
        ax.set_title(key)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    save_dir = os.path.join("experiments", CFG.exp_name)
    plt.savefig(os.path.join(save_dir, 'metrics.pdf'), bbox_inches='tight')


def grid_image(img, num_pieces_ver, num_pieces_hor, channels, piece_size):
    # Cut the image into pieces using slicing
    pieces = np.vsplit(img, num_pieces_ver)
    pieces = np.concatenate([np.hsplit(row_pieces, num_pieces_hor) for row_pieces in pieces])
    # Reshape pieces into a 5-dim numpy array
    pieces = np.reshape(pieces, (num_pieces_ver, num_pieces_hor, piece_size, piece_size, channels))
    pieces = np.transpose(pieces, (0, 1, 4, 2, 3))
    # It really works. Can be checked in dataset_preparation.py
    return pieces


def ungrid_image(image_pieces, num_pieces_ver, num_pieces_hor, channels, piece_size):
    # Reshape the array into a 3-dim original image
    image_pieces = np.transpose(image_pieces, (0, 3, 1, 4, 2))
    image = image_pieces.reshape(num_pieces_ver*piece_size, num_pieces_hor*piece_size, channels)
    # It really works. Can be checked in dataset_preparation.py
    return image