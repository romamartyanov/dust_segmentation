import os
import argparse
import time

import cv2
import torch
import numpy as np

from config import CFG
from utils import seed_everything, grid_image, ungrid_image
from helpers import init_openvino_session
from model_convert import ModelOptimization
from model.unet import UNet
from model.utils import load_model, sigmoid

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    """Input arguments parsing."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default="test_data_tiff/train/image/",
                        help='Path to test dataset.')
    parser.add_argument('--model_path', type=str, default="experiments/test_exp/Unet_efficientnet_b1_best_epoch.pth",
                        help='Path to model weights. If None - dummy weights will be used.')
    parser.add_argument('--export_dir', default="saved_masks/", type=str,
                        help='Export directory.')
    parser.add_argument('--bechmark', default=True, type=bool,
                        help="Run performance test. Without data 'bechmark' mode will use synthetic data.")
    return parser.parse_args()


def load_inference_model_pytorch(model, args):
    model_path = args.model_path
    if model_path is not None:
        model, _, _, _ = load_model(model, model_path)
        print("Loaded weights:", model_path)

    else:
        save_dir = os.path.join("experiments", "bechmark")
        model_path = os.path.join(save_dir, "model.pth")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save({
            'model_state_dict': model.state_dict(),
        }, model_path)
        print("Used dummy weights. It saved to:", model_path)

    model.eval()
    return model, model_path


def one_image_processing(img, compiled_model, output_layer,
                         piece_size, height_b, width_b,
                         num_pieces_hor, num_pieces_ver, thr=0.5):
    channels = img.shape[2]

    # Resize image to divisible size
    img = cv2.resize(img, (height_b, width_b))
    image_pieces = grid_image(img, num_pieces_ver, num_pieces_hor, channels, piece_size)

    mask_pieces = np.zeros(
        (num_pieces_ver, num_pieces_hor, CFG.num_classes, piece_size, piece_size)
    )
    for i in range(num_pieces_ver):
        img_batch = image_pieces[i]
        pred_i = compiled_model([img_batch])[output_layer]
        mask_pieces[i] = sigmoid(pred_i)

    mask = ungrid_image(mask_pieces, num_pieces_ver, num_pieces_hor, CFG.num_classes, piece_size)

    mask = (mask >= thr)
    mask = (mask * 255).astype(np.uint8)
    return mask


def inference_bechmark(compiled_model, output_layer, timer=30):
    laps = []

    height_b, width_b = CFG.big_img_size
    piece_size = CFG.img_size[0]
    # Calculate the number of pieces in the horizontal and vertical direction
    num_pieces_hor = width_b // piece_size
    num_pieces_ver = height_b // piece_size

    # Start timer for benchmark
    t_end = time.time() + timer
    while time.time() < t_end:
        # randn takes about 3 seconds
        img = np.random.randn(4912, 7360, 3)

        img_start = time.time()
        _ = one_image_processing(
            img, compiled_model, output_layer,
            piece_size, height_b, width_b,
            num_pieces_hor, num_pieces_ver
        )
        img_end = time.time()
        laps.append(img_end - img_start)
    laps = np.array(laps)

    print(
        "Results:\n"
        f"Timer: {timer} sec.\n"
        f"Number of processed images: {laps.size}\n"
        f"Mean time for pipeline: {np.mean(laps)}\n"
    )

    return laps



def inference(ir_model, output_layer, dataset_path, export_dir):
    height_b, width_b = CFG.big_img_size
    piece_size = CFG.img_size[0]

    # Calculate the number of pieces in the horizontal and vertical direction
    num_pieces_hor = width_b // piece_size
    num_pieces_ver = height_b // piece_size

    for root, _, files in os.walk(dataset_path):
        sub_root = root.replace(dataset_path, "")
        sub_root = sub_root[1:] if len(sub_root) and sub_root[0] == '/' else sub_root

        for file in files:
            if file == ".DS_Store":
                continue

            temp_export_dir = os.path.join(export_dir, sub_root)
            if not os.path.exists(temp_export_dir):
                os.makedirs(temp_export_dir)

            img_path = os.path.join(root, file)
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            mask = one_image_processing(
                img, ir_model, output_layer,
                piece_size, height_b, width_b,
                num_pieces_hor, num_pieces_ver
            )

            export_path = os.path.join(export_dir, f"{img_name}_mask.png")
            cv2.imwrite(export_path, mask)
            exit()


def main():
    args = parse_args()
    if not args.bechmark and (args.dataset_path is None or args.export_dir is None):
        raise ValueError(
            "Please, provide data for testing in 'dataset_path' argument and "
            "path for segmentation masks in 'export_dir' argument."
            "Or choose 'bechmark' mode. Without data 'bechmark' mode will use synthetic data."
        )

    seed_everything(CFG.seed)
    model = UNet(
        encoder_name=CFG.backbone,
        encoder_pretrained=True,
        n_classes=CFG.num_classes,
        activation=None,
    )
    model.to(CFG.device)

    model, model_path = load_inference_model_pytorch(model, args)

    mo = ModelOptimization(model, model_path)
    ir_path = mo.onnx2openvino(mode="FP16")

    ir_model, output_layer = init_openvino_session(ir_path)

    if args.bechmark:
        inference_bechmark(ir_model, output_layer)
    else:
        inference(ir_model, output_layer, args.dataset_path, args.export_dir)


if __name__ == "__main__":
    main()