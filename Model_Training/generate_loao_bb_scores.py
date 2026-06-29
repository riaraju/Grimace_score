from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from torchvision.transforms import v2

from dataset.kfold_hdf5_bb_dataset import KFoldHDF5BBDataset
from model.grimace_full_model import GrimaceFullModel
from transforms.bounding_box_padding import BoundingBoxPadding
from transforms.crop_bounding_box import CropBoundingBox
from transforms.square_bounding_box import SquareBoundingBox

if __name__ == "__main__":
    experiment_folder = Path("experiments/GrimaceFullLoaoBB")
    val_dataset_file = Path("datasets/GrimaceFullLoaoBB.h5")

    means = [0.5919]
    stds = [0.1457]

    val_transforms = v2.Compose([
        BoundingBoxPadding(0.05),
        SquareBoundingBox(),
        CropBoundingBox(),
        v2.Normalize(mean=means, std=stds),
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BILINEAR),
    ])

    device = torch.device("cuda")

    kfold_val_dataset = KFoldHDF5BBDataset(val_dataset_file, 31, val_transforms, val_transforms, device="cuda", num_channels=3)

    val_predictions = []
    val_labels = []

    video_names = []
    frame_indices = []

    for fold_index in range(0, kfold_val_dataset.num_folds):
        fold_folder = experiment_folder / f"fold_{fold_index}"

        model = GrimaceFullModel(base_model_name="vit_b_16", hidden_layer_sizes=[2048, 1024], dropouts=[0.4, 0.2], input_size=(224, 224), num_channels=3)
        model.load_state_dict(torch.load(fold_folder / "last_weights.pt"))
        model.to(device)
        model.eval()

        val_dataset = kfold_val_dataset.get_val_dataset(fold_index)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

        with h5py.File(val_dataloader.dataset.dataset_file, 'r') as h5f:
            fold_video_names = [h5f["video_names"][i].decode("utf-8") for i in val_dataloader.dataset.indexes]
            fold_frame_indices = [h5f["frame_indices"][i] for i in val_dataloader.dataset.indexes]

            video_names.extend(fold_video_names)
            frame_indices.extend(fold_frame_indices)

        with torch.no_grad():
            for images, labels in val_dataloader:
                outputs = model(images).detach().cpu().squeeze(dim=1).numpy()
                labels = labels.cpu().squeeze(dim=1).numpy()

                pred = outputs.reshape(outputs.shape[0], 5, 4)
                true = labels.reshape(labels.shape[0], 5, 4)

                pred = np.argmax(pred, axis=2)
                true = np.argmax(true, axis=2)

                val_predictions.extend(pred)
                val_labels.extend(true)

    val_predictions = np.array(val_predictions)
    val_labels = np.array(val_labels)

    video_names = video_names[:val_predictions.shape[0]]

    eye_predictions = val_predictions[:, 0]
    nose_predictions = val_predictions[:, 1]
    cheek_predictions = val_predictions[:, 2]
    ear_predictions = val_predictions[:, 3]
    whisker_predictions = val_predictions[:, 4]

    eye_labels = val_labels[:, 0]
    nose_labels = val_labels[:, 1]
    cheek_labels = val_labels[:, 2]
    ear_labels = val_labels[:, 3]
    whisker_labels = val_labels[:, 4]

    eye_acc = np.sum((eye_predictions == eye_labels).astype(int)) / eye_labels.size
    nose_acc = np.sum((nose_predictions == nose_labels).astype(int)) / nose_labels.size
    cheek_acc = np.sum((cheek_predictions == cheek_labels).astype(int)) / cheek_labels.size
    ear_acc = np.sum((ear_predictions == ear_labels).astype(int)) / ear_labels.size
    whisker_acc = np.sum((whisker_predictions == whisker_labels).astype(int)) / whisker_labels.size

    print(f"Eye accuracy: {eye_acc}")
    print(f"Nose accuracy: {nose_acc}")
    print(f"Cheek accuracy: {cheek_acc}")
    print(f"Ear accuracy: {ear_acc}")
    print(f"Whisker accuracy: {whisker_acc}")

    eye_ratability_pred = eye_predictions != 3
    eye_ratability_true = eye_labels != 3

    nose_ratability_pred = nose_predictions != 3
    nose_ratability_true = nose_labels != 3

    cheek_ratability_pred = cheek_predictions != 3
    cheek_ratability_true = cheek_labels != 3

    ear_ratability_pred = ear_predictions != 3
    ear_ratability_true = ear_labels != 3

    whisker_ratability_pred = whisker_predictions != 3
    whisker_ratability_true = whisker_labels != 3

    eye_rat_acc = np.sum((eye_ratability_pred == eye_ratability_true).astype(int)) / eye_ratability_true.size
    nose_rat_acc = np.sum((nose_ratability_pred == nose_ratability_true).astype(int)) / nose_ratability_true.size
    cheek_rat_acc = np.sum((cheek_ratability_pred == cheek_ratability_true).astype(int)) / cheek_ratability_true.size
    ear_rat_acc = np.sum((ear_ratability_pred == ear_ratability_true).astype(int)) / ear_ratability_true.size
    whisker_rat_acc = np.sum((whisker_ratability_pred == whisker_ratability_true).astype(int)) / whisker_ratability_true.size

    eye_rat_tp = np.sum((eye_ratability_pred == 1).astype(int) & (eye_ratability_true == 1).astype(int))
    eye_rat_fp = np.sum((eye_ratability_pred == 1).astype(int) & (eye_ratability_true == 0).astype(int))
    eye_rat_fn = np.sum((eye_ratability_pred == 0).astype(int) & (eye_ratability_true == 1).astype(int))
    eye_rat_precision = eye_rat_tp / (eye_rat_tp + eye_rat_fp)
    eye_rat_recall = eye_rat_tp / (eye_rat_tp + eye_rat_fn)

    nose_rat_tp = np.sum((nose_ratability_pred == 1).astype(int) & (nose_ratability_true == 1).astype(int))
    nose_rat_fp = np.sum((nose_ratability_pred == 1).astype(int) & (nose_ratability_true == 0).astype(int))
    nose_rat_fn = np.sum((nose_ratability_pred == 0).astype(int) & (nose_ratability_true == 1).astype(int))
    nose_rat_precision = nose_rat_tp / (nose_rat_tp + nose_rat_fp)
    nose_rat_recall = nose_rat_tp / (nose_rat_tp + nose_rat_fn)

    cheek_rat_tp = np.sum((cheek_ratability_pred == 1).astype(int) & (cheek_ratability_true == 1).astype(int))
    cheek_rat_fp = np.sum((cheek_ratability_pred == 1).astype(int) & (cheek_ratability_true == 0).astype(int))
    cheek_rat_fn = np.sum((cheek_ratability_pred == 0).astype(int) & (cheek_ratability_true == 1).astype(int))
    cheek_rat_precision = cheek_rat_tp / (cheek_rat_tp + cheek_rat_fp)
    cheek_rat_recall = cheek_rat_tp / (cheek_rat_tp + cheek_rat_fn)

    ear_rat_tp = np.sum((ear_ratability_pred == 1).astype(int) & (ear_ratability_true == 1).astype(int))
    ear_rat_fp = np.sum((ear_ratability_pred == 1).astype(int) & (ear_ratability_true == 0).astype(int))
    ear_rat_fn = np.sum((ear_ratability_pred == 0).astype(int) & (ear_ratability_true == 1).astype(int))
    ear_rat_precision = ear_rat_tp / (ear_rat_tp + ear_rat_fp)
    ear_rat_recall = ear_rat_tp / (ear_rat_tp + ear_rat_fn)

    whisker_rat_tp = np.sum((whisker_ratability_pred == 1).astype(int) & (whisker_ratability_true == 1).astype(int))
    whisker_rat_fp = np.sum((whisker_ratability_pred == 1).astype(int) & (whisker_ratability_true == 0).astype(int))
    whisker_rat_fn = np.sum((whisker_ratability_pred == 0).astype(int) & (whisker_ratability_true == 1).astype(int))
    whisker_rat_precision = whisker_rat_tp / (whisker_rat_tp + whisker_rat_fp)
    whisker_rat_recall = whisker_rat_tp / (whisker_rat_tp + whisker_rat_fn)

    print(f"Eye ratability accuracy: {eye_rat_acc} (Precision: {eye_rat_precision}, Recall: {eye_rat_recall})")
    print(f"Nose ratability accuracy: {nose_rat_acc} (Precision: {nose_rat_precision}, Recall: {nose_rat_recall})")
    print(f"Cheek ratability accuracy: {cheek_rat_acc} (Precision: {cheek_rat_precision}, Recall: {cheek_rat_recall})")
    print(f"Ear ratability accuracy: {ear_rat_acc} (Precision: {ear_rat_precision}, Recall: {ear_rat_recall})")
    print(f"Whisker ratability accuracy: {whisker_rat_acc} (Precision: {whisker_rat_precision}, Recall: {whisker_rat_recall})")

    val_predictions = val_predictions.astype(float)
    val_labels = val_labels.astype(float)

    # Replace value 3 with nan
    val_predictions[val_predictions == 3] = np.nan
    val_labels[val_labels == 3] = np.nan

    val_corr = np.corrcoef(val_labels.flatten(), val_predictions.flatten())[0, 1]
    val_mae = np.abs(val_predictions - val_labels).mean()
    val_mse = ((val_predictions - val_labels) ** 2).mean()

    pred_df = pd.DataFrame({
        "video_name": video_names,
        "frame_index": frame_indices,
        "eye": val_predictions[:, 0],
        "nose": val_predictions[:, 1],
        "cheek": val_predictions[:, 2],
        "ear": val_predictions[:, 3],
        "whisker": val_predictions[:, 4],
    })

    for feature in ["eye", "nose", "cheek", "ear", "whisker"]:
        pred_df[feature] = pred_df[feature].astype(pd.Int32Dtype())

    label_df = pd.DataFrame({
        "video_name": video_names,
        "frame_index": frame_indices,
        "eye": val_labels[:, 0],
        "nose": val_labels[:, 1],
        "cheek": val_labels[:, 2],
        "ear": val_labels[:, 3],
        "whisker": val_labels[:, 4],
    })

    for feature in ["eye", "nose", "cheek", "ear", "whisker"]:
        label_df[feature] = label_df[feature].astype(pd.Int32Dtype())

    print(f"AU Correlation: {val_corr}")
    print(f"AU MAE: {val_mae}")
    print(f"AU MSE: {val_mse}")

    val_predictions = np.nanmean(val_predictions, axis=1)
    val_labels = np.nanmean(val_labels, axis=1)

    val_corr = np.corrcoef(val_labels, val_predictions)[0, 1]
    val_mae = np.abs(val_predictions - val_labels).mean()
    val_mse = ((val_predictions - val_labels) ** 2).mean()

    print(f"Image Correlation: {val_corr}")
    print(f"Image MAE: {val_mae}")
    print(f"Image MSE: {val_mse}")

    pred_df["average"] = val_predictions
    label_df["average"] = val_labels

    pred_df.to_csv("C:/Users/Me/Desktop/new_loao_bb2_predictions_9e.csv", index=False)
    label_df.to_csv("C:/Users/Me/Desktop/new_loao_bb2_labels_9e.csv", index=False)

    pred_df = pred_df.groupby("video_name").agg({
        "average": "mean",
    }).reset_index()

    label_df = label_df.groupby("video_name").agg({
        "average": "mean",
    }).reset_index()

    pred_df.to_csv("C:/Users/Me/Desktop/new_loao_bb2_video_predictions_9e.csv", index=False)
    label_df.to_csv("C:/Users/Me/Desktop/new_loao_bb2_video_labels_9e.csv", index=False)

    val_predictions = pred_df["average"].values
    val_labels = label_df["average"].values

    corr = np.corrcoef(val_labels, val_predictions)[0, 1]
    mae = np.abs(val_labels - val_predictions).mean()
    mse = ((val_labels - val_predictions) ** 2).mean()

    print(f"Video Correlation: {corr}")
    print(f"Video MAE: {mae}")
    print(f"Video MSE: {mse}")
