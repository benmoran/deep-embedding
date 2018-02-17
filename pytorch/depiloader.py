
# coding: utf-8

import os.path
import pathlib
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import nibabel


# In[108]:


class DepiDataset(Dataset):
    LABELS = {'CON': np.array([0], 'uint8'),
              'ASD': np.array([1], 'uint8')}
    
    @staticmethod
    def find_files(rootdir, substring):
        for root, dirs, files in os.walk(rootdir):
            for filename in files:
                if filename.endswith(".gz"):
                    if substring is None or substring in filename:
                        yield os.path.join(root, filename)

    def __init__(self, datadir, substring=None, transform=None):
        self.filenames = list(self.find_files(datadir, substring))
        self.labels = [os.path.basename(filename)[9:12] for filename in self.filenames]
        self.images = np.stack([nibabel.load(filename).get_data() for filename in self.filenames])
        self.transform = transform
    
    def __len__(self):
        return len(self.filenames)
    
    def get_label(self, num):
        path = pathlib.Path(self.filenames[num])
        for part in path.parts:
            if part.startswith("subject_"):
                return self.LABELS[part[8:11]]
        raise ValueError("Couldn't find label for {}".format(self.filenames[num]))

    def get_image(self, num):
        return self.images[num]

    def __getitem__(self, num):
        result = self.get_image(num), self.get_label(num)
        if self.transform is not None:
            result = self.transform(result)
        return result


# Creating some transformations.


# I don't think we'll need this one, 
# but just to show the idea of a parametric transform:
class Threshold(object):
    def __init__(self, threshold_fraction=0.001):
        self.threshold_fraction = threshold_fraction
        
    def __call__(self, sample):
        ndarray, label = sample
        return np.where(np.abs(ndarray)>self.threshold_fraction, 
                        ndarray, 0), label

class Normalize(object):
    "Normalize the voxel data to have max of 1.0"
    def __call__(self, sample):
        ndarray, label = sample
        return ndarray / ndarray.max(), label


class AddChannel(object):
    # Insert a leading "Channels" dimension of size 1
    def __call__(self, sample):
        tensor, label = sample
        tensor = tensor.contiguous()
        return tensor.view(torch.Size([1]) + tensor.shape), label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample
 
        return torch.from_numpy(image), torch.from_numpy(labels)


if __name__ == '__main__':
    dataset = DepiDataset("../../depi", "4mm", 
                          transform=transforms.Compose([Threshold(), 
                                                        Normalize(),
                                                        ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=4)

