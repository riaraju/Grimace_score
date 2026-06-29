from pathlib import Path

import torch

from torchvision.transforms import v2

from criterion.grimace_cross_entropy import GrimaceCrossEntropyLoss
from training.setup import setup
from training.trainer import Trainer
from model.grimace_full_model import GrimaceFullModel
from training.grimace_full_training import GrimaceFullTraining
from transforms.random_choice import RandomChoice

from dataset.kfold_hdf5_bb_dataset import KFoldHDF5BBDataset
from transforms.bounding_box_padding import BoundingBoxPadding
from transforms.crop_bounding_box import CropBoundingBox
from transforms.random_bounding_box_padding import RandomBoundingBoxPadding
from transforms.square_bounding_box import SquareBoundingBox

if __name__ == "__main__":
    experiment_folder = Path("experiments/GrimaceFullLoaoBB")
    dataset_file = Path("datasets/LoaoBBSurround1.h5")
    val_dataset_file = Path("datasets/LoaoPaddedBB.h5")

    means = [0.5919]
    stds = [0.1457]

    train_transforms = v2.Compose([
        RandomChoice([
            RandomBoundingBoxPadding(-0.1, 0.1),
            BoundingBoxPadding(0.05),
        ], weights=[1, 1]),
        SquareBoundingBox(),
        CropBoundingBox(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=(0.85, 1 / 0.85), contrast=(0.95, 1 / 0.95)),
        RandomChoice([
            v2.RandomRotation(10, interpolation=v2.InterpolationMode.NEAREST),
            v2.RandomRotation(10, interpolation=v2.InterpolationMode.BILINEAR),
            v2.Identity()
        ], weights=[1, 1, 2]),
        v2.Normalize(mean=means, std=stds),
        RandomChoice([
            v2.Resize((224, 224), interpolation=v2.InterpolationMode.NEAREST),
            v2.Resize((224, 224), interpolation=v2.InterpolationMode.BILINEAR),
        ], weights=[1, 1])
    ])

    val_transforms = v2.Compose([
        BoundingBoxPadding(0.05),
        SquareBoundingBox(),
        CropBoundingBox(),
        v2.Normalize(mean=means, std=stds),
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BILINEAR),
    ])

    kfold_dataset = KFoldHDF5BBDataset(dataset_file, 31, train_transforms, val_transforms, device="cuda", num_channels=3)
    kfold_val_dataset = KFoldHDF5BBDataset(val_dataset_file, 31, train_transforms, val_transforms, device="cuda", num_channels=3)

    for fold_index in range(0, kfold_dataset.num_folds):
        setup(seed=1)

        fold_folder = experiment_folder / f"fold_{fold_index}"

        model = GrimaceFullModel(base_model_name="vit_b_16", hidden_layer_sizes=[2048, 1024], dropouts=[0.4, 0.2], input_size=(224, 224), num_channels=3)

        train_dataset = kfold_dataset.get_train_dataset(fold_index)
        val_dataset = kfold_val_dataset.get_val_dataset(fold_index)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

        optimizer = torch.optim.SGD([
            {"params": model.base_model.parameters(), "lr": 0.0001, "weight_decay": 0.0},
            {"params": model.hidden_layers.parameters(), "lr": 0.001, "weight_decay": 0.0},
            {"params": model.au_heads.parameters(), "lr": 0.001, "weight_decay": 0.0}
        ], momentum=0.95)
        criterion = GrimaceCrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        device = torch.device("cuda")

        trainer = Trainer(model, optimizer, criterion, device)

        training = GrimaceFullTraining(trainer, scheduler, fold_folder)

        training.train(train_dataloader, val_dataloader, 20)
