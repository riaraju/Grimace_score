from pathlib import Path

import h5py

from dataset.hdf5_bb_dataset import HDF5BBDataset


class KFoldHDF5BBDataset:
    def __init__(self, dataset_file: Path, num_folds: int, train_transforms, val_transforms,
                 device="cpu", num_channels=3):
        self.dataset_file = dataset_file
        self.num_folds = num_folds
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.device = device
        self.num_channels = num_channels

    def get_fold_datasets(self, val_fold_index: int):
        with h5py.File(self.dataset_file, 'r') as file:
            train_indexes = []
            val_indexes = []
            for fold_index in range(self.num_folds):
                fold_index_range = file.attrs["fold_ranges"][fold_index]
                fold_indexes = list(range(fold_index_range[0], fold_index_range[1]))
                if fold_index == val_fold_index:
                    val_indexes += fold_indexes
                else:
                    train_indexes += fold_indexes

            train_dataset = HDF5BBDataset(self.dataset_file, self.train_transforms, train_indexes, self.device, self.num_channels)
            val_dataset = HDF5BBDataset(self.dataset_file, self.val_transforms, val_indexes, self.device, self.num_channels)

        return train_dataset, val_dataset

    def get_train_dataset(self, val_fold_index: int):
        with h5py.File(self.dataset_file, 'r') as file:
            train_indexes = []
            val_indexes = []
            for fold_index in range(self.num_folds):
                fold_index_range = file.attrs["fold_ranges"][fold_index]
                fold_indexes = list(range(fold_index_range[0], fold_index_range[1]))
                if fold_index == val_fold_index:
                    val_indexes += fold_indexes
                else:
                    train_indexes += fold_indexes

            train_dataset = HDF5BBDataset(self.dataset_file, self.train_transforms, train_indexes, self.device, self.num_channels)
        return train_dataset

    def get_val_dataset(self, val_fold_index: int):
        with h5py.File(self.dataset_file, 'r') as file:
            train_indexes = []
            val_indexes = []
            for fold_index in range(self.num_folds):
                fold_index_range = file.attrs["fold_ranges"][fold_index]
                fold_indexes = list(range(fold_index_range[0], fold_index_range[1]))
                if fold_index == val_fold_index:
                    val_indexes += fold_indexes
                else:
                    train_indexes += fold_indexes

            val_dataset = HDF5BBDataset(self.dataset_file, self.val_transforms, val_indexes, self.device, self.num_channels)
        return val_dataset
