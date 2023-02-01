import os
import glob
import cv2
import numpy as np

from config import CFG


def get_images_in_directory(dataset_dir):
    # is required to add "/" at the end, if there is none
    images_dir = os.path.join(dataset_dir, 'image', '')
    masks_dir = os.path.join(dataset_dir, 'mask', '')

    images, masks = [], []
    for img_path in glob.glob(f'{images_dir}*.tiff'):
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(masks_dir, f"{image_name}.png")
        if os.path.exists(mask_path):
            images.append(img_path)
            masks.append(mask_path)
        else:
            print(f"Warning: mask {mask_path} is not exists!")

    return images, masks


def load_image(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    img = np.clip(img, 0, 1)
    return img


def load_mask(path):
    img_msk = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    msk = np.zeros([*img_msk.shape[:2], CFG.num_classes])

    for i, color in enumerate(CFG.class_labels.values()):
        color = np.array(color)
        msk[:, :, i] = cv2.inRange(img_msk, color, color)

    msk = msk.astype(np.float32) / 255
    msk = np.clip(msk, 0, 1)
    return msk