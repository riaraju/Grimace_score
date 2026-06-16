from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5BBDataset(Dataset):
    def __init__(self, dataset_file: Path, transforms, indexes: List[int], device="cpu", num_channels=3, epoch_size=None):
        self.dataset_file = dataset_file
        self.transforms = transforms
        self.indexes = indexes
        self.device = device
        self.num_channels = num_channels
        self.epoch_size = epoch_size if epoch_size is not None else len(indexes)

        # The hdf5 file can not be pickled, which is necessary for the torch dataloader workers to spawn,
        # so we do not initialize it here and open it in __getitem__ instead once the worker has spawned.
        self.hdf5_file = None

    @classmethod
    def from_set_name(cls, dataset_file: Path, set_name: str, transforms, device="cpu", num_channels=3, epoch_size=None):
        with h5py.File(dataset_file, 'r') as file:
            set_range_index = list(file.attrs["set_names"]).index(set_name)
            set_range = file.attrs["set_ranges"][set_range_index]
            indexes = list(range(set_range[0], set_range[1]))

        return cls(dataset_file, transforms, indexes, device, num_channels, epoch_size)

    @classmethod
    def from_entire_file(cls, dataset_file: Path, transforms, device="cpu", num_channels=3, epoch_size=None):
        with h5py.File(dataset_file, 'r') as file:
            indexes = list(range(file.attrs["num_samples"]))
        return cls(dataset_file, transforms, indexes, device, num_channels, epoch_size)

    def __len__(self):
        return self.epoch_size

    def index(self, idx: int):
        return self.indexes[idx]

    def __getitem__(self, idx):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.dataset_file, 'r')

        image_index = self.index(idx)
        image = self.hdf5_file['images'][image_index]
        image = image.astype(np.float32)
        image /= 255.0
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image)
        image = image.to(self.device)
        if self.num_channels != 1:
            image = torch.cat([image] * self.num_channels, dim=0)

        bb = self.hdf5_file['bounding_boxes'][image_index]
        bb = bb.astype(np.int32)
        bb = torch.tensor(bb)
        bb = bb.to(self.device)

        label = self.hdf5_file['labels'][image_index]

        label = label.astype(np.float32)
        label = torch.tensor(label)
        label = label.to(self.device)

        image = self.transforms((image, bb))
        image = image.to(self.device)

        return image, label
