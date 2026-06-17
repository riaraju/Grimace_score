from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from training.persistence import save_model
from training.trainer import Trainer


class GrimaceFullTraining:
    def __init__(self, training: Trainer, scheduler: Any, save_folder: Path):
        self.training = training
        self.scheduler = scheduler
        self.save_folder = save_folder

        self.best_weights = None
        self.best_corr_weights = None
        self.last_weights = None

        self.best_stats = None
        self.last_stats = None

        self.log = []

        self.epoch = 0

    def train_no_val(self, train_dataloader: DataLoader, epochs: int):
        for epoch in range(self.epoch, self.epoch+epochs):
            train_loss, train_predictions, train_labels = self.training.train_epoch(train_dataloader)

            if self.scheduler is not None:
                self.scheduler.step()

            self.last_stats = {
                "epoch": int(epoch),
                "train_loss": float(train_loss)
            }

            self.last_weights = self.training.model.state_dict()

            self.log.append(deepcopy(self.last_stats))

            save_model(self.save_folder, self.last_weights, self.last_weights, self.last_stats, self.last_stats, self.log)

            print(f"Last Epoch {epoch} - Train loss: {train_loss:.4f}")
            print()

        self.epoch = self.epoch + epochs

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, epochs: int):
        if val_dataloader is None:
            return self.train_no_val(train_dataloader, epochs)

        with h5py.File(val_dataloader.dataset.dataset_file, 'r') as h5f:
            video_names = [h5f["video_names"][i] for i in val_dataloader.dataset.indexes]

        for epoch in range(self.epoch, self.epoch+epochs):
            train_loss, train_predictions, train_labels = self.training.train_epoch(train_dataloader)
            val_loss, val_predictions, val_labels = self.training.validate_epoch(val_dataloader)

            if self.scheduler is not None:
                self.scheduler.step()

            au_val_predictions = val_predictions.reshape(val_predictions.shape[0], 5, 4)
            au_val_labels = val_labels.reshape(val_labels.shape[0], 5, 4)

            au_val_pred_cat = np.argmax(au_val_predictions, axis=2)
            au_val_labels_cat = np.argmax(au_val_labels, axis=2)

            val_acc = np.sum((au_val_pred_cat == au_val_labels_cat).astype(int)) / au_val_labels_cat.size
            val_eye_acc = np.sum((au_val_pred_cat[:, 0] == au_val_labels_cat[:, 0]).astype(int)) / au_val_labels_cat[:, 0].size
            val_nose_acc = np.sum((au_val_pred_cat[:, 1] == au_val_labels_cat[:, 1]).astype(int)) / au_val_labels_cat[:, 1].size
            val_cheek_acc = np.sum((au_val_pred_cat[:, 2] == au_val_labels_cat[:, 2]).astype(int)) / au_val_labels_cat[:, 2].size
            val_ear_acc = np.sum((au_val_pred_cat[:, 3] == au_val_labels_cat[:, 3]).astype(int)) / au_val_labels_cat[:, 3].size
            val_whisker_acc = np.sum((au_val_pred_cat[:, 4] == au_val_labels_cat[:, 4]).astype(int)) / au_val_labels_cat[:, 4].size

            au_val_pred_cat = au_val_pred_cat.astype(float)
            au_val_labels_cat = au_val_labels_cat.astype(float)

            au_val_pred_cat[au_val_pred_cat == 3] = np.nan
            au_val_labels_cat[au_val_labels_cat == 3] = np.nan

            avg_val_predictions = np.nanmean(au_val_pred_cat, axis=1)
            avg_val_labels = np.nanmean(au_val_labels_cat, axis=1)

            val_image_mae = np.abs((avg_val_predictions - avg_val_labels)).mean()
            val_image_corr = np.corrcoef(avg_val_labels, avg_val_predictions)[0, 1]

            avg_val_pred_cat = np.round(avg_val_predictions).astype(int)
            avg_val_labels_cat = np.round(avg_val_labels).astype(int)

            val_image_acc = np.sum((avg_val_pred_cat == avg_val_labels_cat).astype(int)) / avg_val_labels_cat.size

            val_video_stats = pd.DataFrame({
                "video_name": video_names,
                "true_rating": avg_val_labels,
                "predicted_rating": avg_val_predictions
            })

            val_video_stats = val_video_stats.groupby("video_name").agg({
                "true_rating": "mean",
                "predicted_rating": "mean"
            }).reset_index()

            val_video_true = val_video_stats["true_rating"].values
            val_video_pred = val_video_stats["predicted_rating"].values

            val_video_mae = np.abs((val_video_true - val_video_pred)).mean()
            val_video_corr = np.corrcoef(val_video_true, val_video_pred)[0, 1]

            self.last_stats = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_eye_acc": float(val_eye_acc),
                "val_nose_acc": float(val_nose_acc),
                "val_cheek_acc": float(val_cheek_acc),
                "val_ear_acc": float(val_ear_acc),
                "val_whisker_acc": float(val_whisker_acc),
                "val_image_mae": float(val_image_mae),
                "val_image_corr": float(val_image_corr),
                "val_image_acc": float(val_image_acc),
                "val_video_mae": float(val_video_mae),
                "val_video_corr": float(val_video_corr)
            }

            self.last_weights = self.training.model.state_dict()

            if self.best_stats is None or val_loss < self.best_stats["val_loss"]:
                self.best_stats = deepcopy(self.last_stats)
                self.best_weights = deepcopy(self.last_weights)

            self.log.append(deepcopy(self.last_stats))

            save_model(self.save_folder, self.best_weights, self.last_weights, self.best_stats, self.last_stats, self.log)

            print(f"Last Epoch {epoch} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Val eye acc: {val_eye_acc:.4f}, Val nose acc: {val_nose_acc:.4f}, Val cheek acc: {val_cheek_acc:.4f}, Val ear acc: {val_ear_acc:.4f}, Val whisker acc: {val_whisker_acc:.4f}, Val video mae: {val_video_mae:.4f}, Val video corr: {val_video_corr:.4f}, Val image mae: {val_image_mae:.4f}, Val image corr: {val_image_corr:.4f}, Val image acc: {val_image_acc:.4f}")
            print(f"Best Epoch {self.best_stats['epoch']} - Train loss: {self.best_stats['train_loss']:.4f}, Val loss: {self.best_stats['val_loss']:.4f}, Val acc: {self.best_stats['val_acc']:.4f}, Val eye acc: {self.best_stats['val_eye_acc']:.4f}, Val nose acc: {self.best_stats['val_nose_acc']:.4f}, Val cheek acc: {self.best_stats['val_cheek_acc']:.4f}, Val ear acc: {self.best_stats['val_ear_acc']:.4f}, Val whisker acc: {self.best_stats['val_whisker_acc']:.4f}, Val video mae: {self.best_stats['val_video_mae']:.4f}, Val video corr: {self.best_stats['val_video_corr']:.4f}, Val image mae: {self.best_stats['val_image_mae']:.4f}, Val image corr: {self.best_stats['val_image_corr']:.4f}, Val image acc: {self.best_stats['val_image_acc']:.4f}")
            print()

        self.epoch = self.epoch + epochs
