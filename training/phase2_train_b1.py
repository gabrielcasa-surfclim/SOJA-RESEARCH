"""
phase2_train_b1.py — Fase 2 B1: RandAugment moderado.

Mesma config do baseline (lr=0.0005, dropout=0.1, onecycle, AdamW)
com augmentation="randaug_moderate" e imagens segmentadas.
"""

import csv
import gc
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import models

# Add training dir to path
sys.path.insert(0, os.path.dirname(__file__))
import prepare

# ── Monkey-patch para usar imagens segmentadas ──────────────────────────────
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SEGMENTED_DIR = os.path.join(PROJECT_ROOT, "data", "images_segmented")

_original_load = prepare._load_split_csv

def _patched_load(split_name):
    records = _original_load(split_name)
    remapped = []
    for path, cls in records:
        seg_path = path.replace("/data/images/", "/data/images_segmented/")
        if os.path.exists(os.path.join(PROJECT_ROOT, seg_path.replace(PROJECT_ROOT + "/", ""))) or os.path.exists(seg_path):
            remapped.append((seg_path, cls))
        else:
            remapped.append((path, cls))
    return remapped

prepare._load_split_csv = _patched_load

# ── Config (IDENTICAL to baseline except augmentation) ──────────────────────
EXPERIMENT_ID = "B1"
MODEL_NAME = "efficientnet_b0"
LR = 0.0005
DROPOUT = 0.1
OPTIMIZER_NAME = "adamw"
SCHEDULER_NAME = "onecycle"
AUGMENTATION = "randaug_moderate"
WEIGHT_DECAY = 0.0001
LABEL_SMOOTHING = 0.05
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 6
BUDGET_SECONDS = 900
SEED = 42

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "best_phase2_B1.pth")
RESULTS_CSV = os.path.join(os.path.dirname(__file__), "phase2_results.csv")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)


def build_model(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, num_classes),
    )
    return model


def train():
    set_seed(SEED)
    device = prepare.get_device()
    print(f"Device: {device}")
    print(f"Experiment: {EXPERIMENT_ID} — RandAugment Moderado")
    print(f"Config: lr={LR} dropout={DROPOUT} {SCHEDULER_NAME} {AUGMENTATION} {OPTIMIZER_NAME}")
    print(f"Budget: {BUDGET_SECONDS}s | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}\n")

    # Data
    train_loader, val_loader, class_names = prepare.get_dataloaders(
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        augmentation_level=AUGMENTATION,
        num_workers=2,
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}\n")

    # Model
    model = build_model(num_classes).to(device)

    # Optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )

    # Loss
    class_weights = prepare.get_class_weights()
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    # Train
    start_time = time.time()
    epochs_completed = 0

    for epoch in range(EPOCHS):
        elapsed = time.time() - start_time
        if elapsed >= BUDGET_SECONDS:
            print(f"\nBudget atingido ({elapsed:.0f}s)")
            break

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if time.time() - start_time >= BUDGET_SECONDS:
                break

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epochs_completed = epoch + 1
        train_acc = correct / max(total, 1)
        avg_loss = running_loss / max(batch_idx + 1, 1)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Acc: {train_acc:.4f} | {elapsed:.0f}s")

    total_time = time.time() - start_time
    print(f"\nTreinamento: {epochs_completed} epochs em {total_time:.1f}s")

    # Evaluate validation
    print("\n--- VALIDAÇÃO ---")
    val_metrics = prepare.evaluate(model, val_loader, device, class_names)

    # Evaluate holdout
    print("\n--- HOLDOUT ---")
    from prepare import _load_split_csv, _ListDataset
    from torchvision import transforms

    holdout_records = _load_split_csv("holdout")
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    holdout_samples = []
    for path, cls in holdout_records:
        if cls in class_to_idx:
            holdout_samples.append((path, class_to_idx[cls]))

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    holdout_ds = _ListDataset(holdout_samples, val_transform)
    holdout_loader = torch.utils.data.DataLoader(holdout_ds, batch_size=BATCH_SIZE, shuffle=False)
    holdout_metrics = prepare.evaluate(model, holdout_loader, device, class_names)

    # Per-class holdout accuracy
    model.eval()
    per_class = {cls: {"correct": 0, "total": 0} for cls in class_names if cls != "Saudável"}
    with torch.no_grad():
        for path, cls in holdout_records:
            if cls not in class_to_idx or cls == "Saudável":
                continue
            from PIL import Image
            img = Image.open(path).convert("RGB")
            img_t = val_transform(img).unsqueeze(0).to(device)
            output = model(img_t)
            pred_idx = output.argmax(1).item()
            per_class[cls]["total"] += 1
            if pred_idx == class_to_idx[cls]:
                per_class[cls]["correct"] += 1

    print("\n--- COMPARATIVO ---")
    print(f"{'Métrica':<20} {'Baseline B0':>12} {'B1 RandAug':>12} {'Delta':>8}")
    print("-" * 55)
    print(f"{'Val Accuracy':<20} {'98.17%':>12} {val_metrics['accuracy']*100:>11.2f}% {(val_metrics['accuracy']-0.9817)*100:>+7.2f}pp")
    print(f"{'Holdout Accuracy':<20} {'70.69%':>12} {holdout_metrics['accuracy']*100:>11.2f}% {(holdout_metrics['accuracy']-0.7069)*100:>+7.2f}pp")
    print(f"{'Holdout F1':<20} {'0.6994':>12} {holdout_metrics['f1']:>12.4f} {holdout_metrics['f1']-0.6994:>+8.4f}")

    baseline_per_class = {"Ferrugem": 0.70, "Mancha-alvo": 0.89, "Mosaico": 0.57, "Olho-de-rã": 0.62, "Oídio": 0.75}

    print(f"\n{'Classe':<15} {'Baseline':>10} {'B1':>10} {'Delta':>8}")
    print("-" * 45)
    for cls in sorted(per_class.keys()):
        pc = per_class[cls]
        if pc["total"] > 0:
            acc = pc["correct"] / pc["total"]
            base = baseline_per_class.get(cls, 0)
            print(f"{cls:<15} {base*100:>9.0f}% {acc*100:>9.0f}% {(acc-base)*100:>+7.0f}pp")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": MODEL_NAME,
        "num_classes": num_classes,
        "class_names": class_names,
        "accuracy": holdout_metrics["accuracy"],
        "val_accuracy": val_metrics["accuracy"],
        "metrics": holdout_metrics,
        "val_metrics": val_metrics,
        "per_class": per_class,
        "hyperparams": {
            "experiment": EXPERIMENT_ID,
            "lr": LR, "dropout": DROPOUT, "optimizer": OPTIMIZER_NAME,
            "scheduler": SCHEDULER_NAME, "augmentation": AUGMENTATION,
            "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTHING,
            "batch_size": BATCH_SIZE, "image_size": IMAGE_SIZE,
            "epochs": epochs_completed, "budget": BUDGET_SECONDS,
        },
    }, CHECKPOINT_PATH)
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")

    # Update results CSV
    per_class_accs = {}
    for cls in per_class:
        pc = per_class[cls]
        per_class_accs[cls] = round(pc["correct"] / pc["total"], 2) if pc["total"] > 0 else 0.0

    config_str = (f"lr={LR} dropout={DROPOUT} {SCHEDULER_NAME} {AUGMENTATION} "
                  f"{OPTIMIZER_NAME} wd={WEIGHT_DECAY} ls={LABEL_SMOOTHING} "
                  f"bs={BATCH_SIZE} img={IMAGE_SIZE} ep={epochs_completed}")

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            EXPERIMENT_ID, "1", f"RandAugment moderado ({AUGMENTATION})",
            f"{val_metrics['accuracy']:.4f}",
            f"{holdout_metrics['accuracy']:.4f}",
            f"{holdout_metrics['f1']:.4f}",
            f"{per_class_accs.get('Ferrugem', 0):.2f}",
            f"{per_class_accs.get('Mancha-alvo', 0):.2f}",
            f"{per_class_accs.get('Oídio', 0):.2f}",
            f"{per_class_accs.get('Olho-de-rã', 0):.2f}",
            f"{per_class_accs.get('Mosaico', 0):.2f}",
            config_str,
            datetime.now().isoformat(timespec="seconds"),
        ])
    print(f"Resultado adicionado a {RESULTS_CSV}")

    # Cleanup
    del model, optimizer, scheduler, criterion
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    print("\nMemória liberada (gc.collect + mps.empty_cache)")

    return holdout_metrics


if __name__ == "__main__":
    train()
