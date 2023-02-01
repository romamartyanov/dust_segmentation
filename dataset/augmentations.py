from config import CFG

import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def pre_transforms(image_size=CFG.img_size):
    return [
        albu.Resize(*image_size, interpolation=cv2.INTER_LINEAR, p=1)
    ]


def flip_transforms():
    return [
        albu.VerticalFlip(p=0.5),
        albu.HorizontalFlip(p=0.5)
    ]


def spatial_transforms():
    distorsion = [
        albu.GridDistortion(always_apply=False,
                            p=0.25,
                            num_steps=6,
                            distort_limit=(-0.06, 0.06)),
        albu.CoarseDropout(max_holes=8,
                           max_height=CFG.img_size[0]//20,
                           max_width=CFG.img_size[1]//20,
                           min_holes=5, fill_value=0,
                           mask_fill_value=0,
                           p=0.25),
    ]
    return distorsion


def pixel_transforms():
    return [
        albu.RandomBrightnessContrast(always_apply=False, p=0.25,
                                      brightness_limit=0.15,
                                      contrast_limit=0.15,
                                      brightness_by_max=True)
    ]


def rotate_transforms():
    return [
        albu.ShiftScaleRotate(
            shift_limit=0.06, scale_limit=0.05, rotate_limit=10, p=0.5)
    ]


def post_transforms():
    # Convert image to torch.Tensor format
    return [ToTensorV2(transpose_mask=True)]


def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


def get_train_transforms():
    return compose([
        pre_transforms(),
        flip_transforms(),
        spatial_transforms(),
        pixel_transforms(),
        rotate_transforms(),
        post_transforms()
    ])


def get_valid_transforms():
    return compose([
        pre_transforms(),
        post_transforms()
    ])
