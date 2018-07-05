import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
# from source.datasets.folder import ImageFolder

class TestDataset(Dataset):
    def __init__(self, image_path, transform):
        self.image_path = image_path
        self.transform = transform
        self.list = self.get_list(image_path)
        self.num_data = len(self.list)

    def get_list(self, image_path):
        return os.listdir(image_path)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_path, self.list[index]))
        return self.transform(image)

    def __len__(self):
        return self.num_data

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images



    def __len__(self):
        return len(self.imgs)

def get_loader(image_path, metadata_path, crop_size, image_size, batch_size, dataset='ImageNet', mode='train'):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    elif mode == 'test' or mode == 'evaluate' or mode == 'vis':
        transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif mode == 'demo':
        transform = transforms.Compose([
            transforms.Scale((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if dataset == 'ImageNet':
        dataset = ImageFolder(os.path.join(image_path, mode), transform)
    print(len(dataset))
    shuffle = False
    if mode == 'train':
        shuffle = True
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
