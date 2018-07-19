import torch.utils.data as data

from PIL import Image
import numpy as np
import random

import os
import os.path


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, ages, extensions):
    persons = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            person = []
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    age = int(fname.split('_')[-1].split('.')[0])
                    item = (path, age_to_class(age, ages))
                    person.append(item)

        persons.append(person)

    return persons

def age_to_class(age, ages):
    for i, _age in enumerate(ages):
        if age <= _age:
            return i
    return len(ages)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class AgePairDataFolder(data.Dataset):

    def __init__(self, root, ages=[10, 18, 30, 40, 50, 60], loader=default_loader, extensions=IMG_EXTENSIONS, transform=None, target_transform=None):
        self.ages = ages
        persons, person_to_idx = find_classes(root)
        samples = make_dataset(root, person_to_idx, ages, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def get_p_example(self, person):
        index = random.randrange(len(person))
        path, target = person[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, target)

    def get_n_example(self, person):
        index = random.randrange(len(person))
        path, target = person[index]
        target = (target + random.randrange(1, len(self.ages)+1)) % (len(self.ages)+1)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, target)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        personA = self.samples[index]
        _index = (index + random.randrange(1, len(self.samples))) % len(self.samples)
        personB = self.samples[_index]
        pExample1, pExample2 = self.get_p_example(personA), self.get_p_example(personA)
        nExample1, nExample2 = self.get_p_example(personA), self.get_p_example(personB)

        return pExample1[0], pExample2[0], pExample1[1], pExample2[1], nExample1[0], nExample2[0], nExample1[1], nExample2[1]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str