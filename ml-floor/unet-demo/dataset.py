import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '_mask.gif'))
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask


def plot_group(image, mask):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')

    plt.show()


def test():
    TRAIN_IMG_DIR = "./data/train_image"
    TRAIN_MASK_DIR = "./data/train_mask"
    VAL_IMG_DIR = "./data/val_image"
    VAL_MASK_DIR = "./data/val_mask"

    dataset_train = CarvanaDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    dataset_val = CarvanaDataset(VAL_IMG_DIR, VAL_MASK_DIR)
    print(len(dataset_train), len(dataset_val))

    plot_group(*dataset_train[221])


if __name__ == '__main__':
    test()
