import json
import csv
from pathlib import Path
from typing import List

import torch


def save_model(save_folder: Path, best_weights: dict, last_weights: dict, best_stats: dict, last_stats: dict, log: List[dict]):
    save_folder.mkdir(parents=True, exist_ok=True)

    best_model_file = save_folder / "best_weights.pt"
    last_model_file = save_folder / "last_weights.pt"

    torch.save(best_weights, best_model_file)
    torch.save(last_weights, last_model_file)

    best_stats_file = save_folder / "best_stats.json"
    last_stats_file = save_folder / "last_stats.json"

    with open(best_stats_file, "w") as f:
        json.dump(best_stats, f)

    with open(last_stats_file, "w") as f:
        json.dump(last_stats, f)

    if len(log) != 0:
        log_file = save_folder / "training_log.csv"
        fieldnames = list(log[0].keys())

        with open(log_file, "w", newline="") as f:
            csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(log)
