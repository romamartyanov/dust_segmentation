import os
import argparse
import cv2
import numpy as np

from config import CFG
from utils import grid_image


def parse_args():
    """Input arguments parsing."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to input dataset.', required=True)
    parser.add_argument('--export_dir', type=str, help='Export directory.', required=True)
    parser.add_argument('--big_size', default=tuple(CFG.big_img_size), type=tuple,
                        help='Size of resized original image.')
    parser.add_argument('--piece_size', default=CFG.img_size[0], type=int,
                        help='Size of image pieces.')
    return parser.parse_args()


def save_grid_images(img_path, export_dir, height_b, width_b, piece_size):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img_ext = ".".join(os.path.splitext(os.path.basename(img_path))[1:])

    num_pieces_hor = width_b // piece_size
    num_pieces_ver = height_b // piece_size

    img = cv2.imread(img_path)
    img = cv2.resize(img, (height_b, width_b))
    channels = img.shape[2]
    # Cut image into pieces
    image_pieces = grid_image(img, num_pieces_ver, num_pieces_hor, channels, piece_size)

    # Save cutted image pieces
    for i in range(num_pieces_ver):
        for j in range(num_pieces_hor):
            export_path = os.path.join(export_dir, f"{img_name}_{i}_{j}{img_ext}")
            image_piece = np.transpose(image_pieces[i][j], (1, 2, 0))
            cv2.imwrite(export_path, image_piece)


if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.dataset_path
    export_dir = args.export_dir

    # dataset_path = "test_data_tiff/train"
    # export_dir = "test_data_tiff__cut/train/"
    # dataset_path = "test_data_tiff/valid"
    # export_dir = "test_data_tiff__cut/valid/"

    # Get the height and width of the resized image
    height, width = args.big_size
    piece_size = args.piece_size

    # Check is this size divisible by piece size
    if height % piece_size != 0 or width % piece_size != 0:
        raise ValueError("Image size is not divisible by piece size!")

    for root, _, files in os.walk(dataset_path):
        sub_root = root.replace(dataset_path, "")
        sub_root = sub_root[1:] if len(sub_root) and sub_root[0] == '/' else sub_root

        for file in files:
            if file == ".DS_Store":
                continue

            temp_export_dir = os.path.join(export_dir, sub_root)
            if not os.path.exists(temp_export_dir):
                os.makedirs(temp_export_dir)

            file_path = os.path.join(root, file)
            filename = os.path.splitext(os.path.basename(file_path))[0]

            save_grid_images(file_path, temp_export_dir, height, width, piece_size)
