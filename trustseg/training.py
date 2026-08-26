from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from trustseg.data import VolumeDataset, collate_volumes, pad_to_factor, unpad
from trustseg.io import read_volume, write_volume
from trustseg.losses import trustseg_loss
from trustseg.metrics import SegmentationMetrics, binary_metrics, mean_metrics
from trustseg.models import UNet3D


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_model(model_config: dict[str, Any], with_bbox: bool = True) -> UNet3D:
    in_channels = int(model_config.get("image_channels", 1)) + int(with_bbox)
    return UNet3D(
        in_channels=in_channels,
        out_channels=1,
        base_channels=int(model_config.get("base_channels", 16)),
        dropout=float(model_config.get("dropout", 0.3)),
    )


def load_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        return checkpoint
    model.load_state_dict(checkpoint)
    return {"model_state": checkpoint}


def _pad_supervision(
    target: torch.Tensor,
    weight: torch.Tensor,
    padding: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_depth, pad_height, pad_width = padding
    pad_values = (0, pad_width, 0, pad_height, 0, pad_depth)
    return F.pad(target, pad_values), F.pad(weight, pad_values)


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    factor: int = 16,
    threshold: float = 0.5,
    prediction_dir: Path | None = None,
) -> tuple[SegmentationMetrics, list[dict[str, Any]]]:
    model.eval()
    case_metrics: list[SegmentationMetrics] = []
    rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            inputs = batch["inputs"].to(device, non_blocking=True)
            padded_inputs, padding = pad_to_factor(inputs, factor=factor)
            probability = torch.sigmoid(unpad(model(padded_inputs), padding)).cpu().numpy()
            target = batch["target"].numpy()

            for index, case_id in enumerate(batch["case_id"]):
                depth, height, width = batch["shape"][index]
                probability_volume = probability[index, 0, :depth, :height, :width]
                target_volume = target[index, 0, :depth, :height, :width]
                prediction = probability_volume >= threshold
                metrics = binary_metrics(prediction, target_volume)
                case_metrics.append(metrics)
                rows.append({"case_id": case_id, **metrics.to_dict()})

                if prediction_dir is not None:
                    _, reference = read_volume(batch["reference_path"][index])
                    write_volume(
                        prediction.astype(np.uint8),
                        prediction_dir / f"{case_id}_pred_mask.nii.gz",
                        reference,
                    )

    return mean_metrics(case_metrics), rows


def train_model(
    model: torch.nn.Module,
    train_dataset: VolumeDataset,
    validation_dataset: VolumeDataset,
    output_dir: Path,
    training_config: dict[str, Any],
    stage: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(training_config.get("seed", 2024))
    seed_everything(seed)

    device = choose_device(str(training_config.get("device", "cuda")))
    model.to(device)
    optimizer = Adam(model.parameters(), lr=float(training_config.get("learning_rate", 1e-4)))

    batch_size = int(training_config.get("batch_size", 1))
    workers = int(training_config.get("workers", 4))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=collate_volumes,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=collate_volumes,
    )

    epochs = int(training_config.get("epochs", 200))
    factor = int(training_config.get("padding_factor", 16))
    threshold = float(training_config.get("threshold", 0.5))
    best_dice = -1.0
    best_path = output_dir / "best_model.pth"
    history_path = output_dir / "history.csv"
    started = time.time()

    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_dice"])
        writer.writeheader()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"{stage} epoch {epoch}/{epochs}")
        for batch in progress:
            inputs = batch["inputs"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)

            padded_inputs, padding = pad_to_factor(inputs, factor=factor)
            target, weight = _pad_supervision(target, weight, padding)
            logits = model(padded_inputs)
            loss, parts = trustseg_loss(logits, target, weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach())
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", dice=f"{parts['dice']:.4f}")

        train_loss = running_loss / max(1, len(train_loader))
        validation, _ = evaluate_model(
            model,
            validation_loader,
            device,
            factor=factor,
            threshold=threshold,
        )

        with history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_dice"])
            writer.writerow(
                {"epoch": epoch, "train_loss": f"{train_loss:.8f}", "val_dice": f"{validation.dice:.8f}"}
            )

        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.5f}, "
            f"val_DSC={validation.dice * 100:.2f}%"
        )
        if validation.dice > best_dice:
            best_dice = validation.dice
            torch.save(
                {
                    "stage": stage,
                    "epoch": epoch,
                    "best_validation_dice": best_dice,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                best_path,
            )

    summary = {
        "stage": stage,
        "epochs": epochs,
        "best_validation_dice": best_dice,
        "duration_seconds": time.time() - started,
        "checkpoint": str(best_path),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    return best_path


def write_evaluation(
    aggregate: SegmentationMetrics,
    cases: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump({"mean": asdict(aggregate), "cases": cases}, stream, indent=2)

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["case_id", "dice", "iou", "hd95", "asd"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)
