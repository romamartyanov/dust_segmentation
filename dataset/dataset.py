from torch.utils.data import Dataset

from dataset.utils import get_images_in_directory, load_image, load_mask


class DatasetBuilder(Dataset):
    def __init__(self, dataset_dir, label=True, transforms=None):
        self.label = label
        self.img_paths, self.msk_paths = get_images_in_directory(dataset_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.msk_paths)

    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        img = []
        img = load_image(img_path)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_mask(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            return img, msk

        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            return img