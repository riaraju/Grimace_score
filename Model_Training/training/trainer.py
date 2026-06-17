from typing import Tuple, Any, Callable

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: nn.Module, optimizer: Optimizer, criterion: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def get_training_state(self) -> Tuple[Any, dict]:
        return self.model.state_dict(), {"optimizer": self.optimizer.state_dict()}

    def train_batch(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        # Set model to training mode, enabling dropouts
        self.model.train()
        # Zero the gradients
        self.optimizer.zero_grad()
        # Enable gradients
        with torch.set_grad_enabled(True):
            # Forward pass
            outputs = self.model(images)
            # Calculate loss
            loss = self.criterion(outputs, labels)
        # Backward pass
        loss.backward()
        # Update the weights
        self.optimizer.step()

        return loss, outputs.detach()

    def validate_batch(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        # Set model to training mode, disabling dropouts
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

        return loss, outputs

    def epoch(self, dataloader: DataLoader, batch_function: Callable) -> Tuple[float, np.ndarray, np.ndarray]:
        running_epoch_loss = 0
        epoch_predictions = torch.tensor(0)
        epoch_labels = torch.tensor(0)

        for i, (images, labels) in enumerate(dataloader):
            #for image in images:
            #    cv2.imshow("image", image.cpu().numpy().transpose(1, 2, 0))
            #    cv2.waitKey(0)

            if isinstance(images, list):
                images = [image.to(self.device) for image in images]
            else:
                images = images.to(self.device)
            labels = labels.to(self.device)
            batch_loss, outputs = batch_function(images, labels)

            batch_size = outputs.shape[0]
            running_epoch_loss += batch_loss.item() * batch_size

            pred = outputs.cpu().squeeze(dim=1)
            true = labels.cpu().squeeze(dim=1)

            # Append to the list of all predictions and labels in the epoch
            epoch_predictions = pred if i == 0 else torch.cat((epoch_predictions, pred), dim=0)
            epoch_labels = true if i == 0 else torch.cat((epoch_labels, true), dim=0)

        # Compute the average loss over all samples
        epoch_loss = running_epoch_loss / epoch_predictions.shape[0]

        return epoch_loss, epoch_predictions.numpy(), epoch_labels.numpy()

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        return self.epoch(dataloader, self.train_batch)

    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        return self.epoch(dataloader, self.validate_batch)
