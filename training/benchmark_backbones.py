"""
benchmark_backbones.py — Treina e compara múltiplos backbones no mesmo split.

Treina convnext_base e maxvit_tiny_tf_224, avalia no holdout externo,
e gera relatório comparativo com EfficientNet-B0 como referência.

Uso:
    python3 training/benchmark_backbones.py
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import timm
import torch
import torch.nn as nn
from torchvision import transforms

from prepare import (
    _ListDataset,
    _load_split_csv,
    evaluate,
    get_class_weights,
    get_dataloaders,
    get_device,
)

# ---------------------------------------------------------------------------
# Config — mesmos hiperparâmetros do EfficientNet-B0 baseline
# ---------------------------------------------------------------------------
LEARNING_RATE = 0.0003
IMAGE_SIZE = 224
DROPOUT = 0.2
OPTIMIZER = "adamw"
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
SCHEDULER = "cosine"
AUGMENTATION_LEVEL = "light"
TRAINING_BUDGET_SECONDS = 8 * 60  # 8 min por modelo (convnext precisa)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Modelos a treinar (EfficientNet-B0 é referência, não retreina)
BACKBONES = [
    {
        "name": "convnext_base",
        "params_m": 88.6,
        "initial_batch": 16,
        "min_batch": 4,
        "epochs": 3,
    },
    {
        "name": "maxvit_tiny_tf_224",
        "params_m": 30.9,
        "initial_batch": 16,
        "min_batch": 4,
        "epochs": 3,
    },
]

EFFICIENTNET_BASELINE = {
    "name": "efficientnet_b0",
    "params_m": 5.3,
    "val_accuracy": 0.972,
    "val_f1": 0.965,
    "holdout_accuracy": 0.638,
    "holdout_f1": None,
    "epochs_completed": 3,
    "time_seconds": 300,
    "batch_size": 32,
}

# Histórico de experimentos
HISTORY = [
    {"experiment": "Random split (data leak)", "holdout_pct": 40.7, "note": "leak entre train/val"},
    {"experiment": "Unificação classes EN/PT", "holdout_pct": 42.0, "note": "Target Spot→Mancha-alvo"},
    {"experiment": "Split C (folder group)", "holdout_pct": 47.3, "note": "sem leak, 112 imgs holdout"},
    {"experiment": "50/50 campo no treino", "holdout_pct": 63.8, "note": "58 imgs holdout"},
    {"experiment": "Segmentação HSV (revertida)", "holdout_pct": 60.3, "note": "perdeu tecido doente"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_model_timm(model_name, num_classes, dropout):
    """Cria modelo via timm com pretrained ImageNet."""
    model = timm.create_model(
        model_name, pretrained=True, num_classes=num_classes, drop_rate=dropout
    )
    return model


def load_holdout(class_names, image_size, batch_size):
    """Carrega holdout externo como DataLoader."""
    holdout_records = _load_split_csv("holdout")
    if not holdout_records:
        return None, []

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    samples = []
    for path, cls in holdout_records:
        if cls in class_to_idx:
            # Paths no CSV já são absolutos
            if os.path.isabs(path):
                full_path = path
            else:
                full_path = os.path.join(PROJECT_ROOT, path)
            samples.append((full_path, class_to_idx[cls]))

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = _ListDataset(samples, val_transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, holdout_records


def holdout_per_class_source(model, holdout_records, class_names, device, image_size):
    """Avalia holdout com detalhamento por classe e por fonte."""
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    model.eval()
    results_by_class = defaultdict(lambda: {"correct": 0, "total": 0})
    results_by_source = defaultdict(lambda: {"correct": 0, "total": 0})

    from PIL import Image

    with torch.no_grad():
        for record in holdout_records:
            path, cls = record[0], record[1]
            if cls not in class_to_idx:
                continue
            label_idx = class_to_idx[cls]

            if os.path.isabs(path):
                full_path = path
            else:
                full_path = os.path.join(PROJECT_ROOT, path)

            try:
                img = Image.open(full_path).convert("RGB")
                img_t = val_transform(img).unsqueeze(0).to(device)
                out = model(img_t)
                pred = out.argmax(dim=1).item()
            except Exception:
                continue

            correct = int(pred == label_idx)
            results_by_class[cls]["correct"] += correct
            results_by_class[cls]["total"] += 1

            # Fonte: ler do CSV (col 2) se disponível
            source = record[2] if len(record) > 2 else "unknown"
            results_by_source[source]["correct"] += correct
            results_by_source[source]["total"] += 1

    return dict(results_by_class), dict(results_by_source)


def load_holdout_records_full():
    """Carrega holdout CSV com todas as colunas."""
    csv_path = os.path.join(PROJECT_ROOT, "data", "splits", "holdout.csv")
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append((row["path"], row["class"], row.get("source", ""), row.get("folder", "")))
    return records


# ---------------------------------------------------------------------------
# Train one backbone
# ---------------------------------------------------------------------------
def train_backbone(backbone_cfg, train_loader, val_loader, class_names, class_weights_tensor, device):
    """Treina um backbone e retorna métricas + modelo."""
    model_name = backbone_cfg["name"]
    batch_size = backbone_cfg["initial_batch"]
    epochs = backbone_cfg["epochs"]
    min_batch = backbone_cfg["min_batch"]

    num_classes = len(class_names)
    print(f"\n{'='*70}")
    print(f"  TREINANDO: {model_name} (~{backbone_cfg['params_m']:.1f}M params)")
    print(f"  LR={LEARNING_RATE} | Batch={batch_size} | Epochs={epochs} | Budget={TRAINING_BUDGET_SECONDS}s")
    print(f"{'='*70}\n")

    # Build model
    model = build_model_timm(model_name, num_classes, DROPOUT)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parâmetros: {n_params:.1f}M")

    model = model.to(device)

    # Rebuild data loaders with this batch size if different from default
    from prepare import get_dataloaders as _gdl
    if batch_size != 32:
        tl, vl, _ = _gdl(image_size=IMAGE_SIZE, batch_size=batch_size, augmentation_level=AUGMENTATION_LEVEL)
    else:
        tl, vl = train_loader, val_loader

    # Optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Scheduler
    steps_per_epoch = len(tl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps_per_epoch
    )

    # Loss
    cw = class_weights_tensor.to(device) if class_weights_tensor is not None else None
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)

    # Train loop with OOM fallback
    start_time = time.time()
    epochs_completed = 0
    oom_occurred = False

    for epoch in range(epochs):
        elapsed = time.time() - start_time
        if elapsed >= TRAINING_BUDGET_SECONDS:
            print(f"  Budget atingido após {elapsed:.0f}s")
            break

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        for batch_idx, (images, labels) in enumerate(tl):
            if time.time() - start_time >= TRAINING_BUDGET_SECONDS:
                break

            images = images.to(device)
            labels = labels.to(device)

            try:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "MPS" in str(e):
                    # OOM — reduz batch e recomeça
                    torch.mps.empty_cache() if hasattr(torch, "mps") else None
                    new_batch = max(min_batch, batch_size // 2)
                    if new_batch < batch_size and not oom_occurred:
                        print(f"  OOM! Reduzindo batch {batch_size} → {new_batch}")
                        batch_size = new_batch
                        backbone_cfg["actual_batch"] = batch_size
                        tl, vl, _ = _gdl(
                            image_size=IMAGE_SIZE,
                            batch_size=batch_size,
                            augmentation_level=AUGMENTATION_LEVEL,
                        )
                        # Rebuild scheduler
                        steps_per_epoch = len(tl)
                        params = filter(lambda p: p.requires_grad, model.parameters())
                        optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=epochs * steps_per_epoch
                        )
                        oom_occurred = True
                        break  # restart epoch
                    else:
                        raise
                else:
                    raise

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_count += 1
        else:
            # Epoch completed normally
            epochs_completed = epoch + 1
            train_acc = correct / max(total, 1)
            avg_loss = running_loss / max(batch_count, 1)
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Tempo: {elapsed:.0f}s")
            continue

        # If we broke out of batch loop (OOM restart), redo this epoch
        if oom_occurred:
            oom_occurred = False
            continue

    total_time = time.time() - start_time
    print(f"  Treinamento: {epochs_completed} epochs em {total_time:.1f}s")

    # Evaluate val
    print(f"\n  Avaliando validação...")
    val_metrics = evaluate(model, vl, device, class_names)

    # Save checkpoint
    ckpt_path = os.path.join(SCRIPT_DIR, f"best_{model_name}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "num_classes": num_classes,
        "class_names": class_names,
        "accuracy": val_metrics["accuracy"],
        "metrics": val_metrics,
    }, ckpt_path)
    print(f"  Checkpoint: {ckpt_path}")

    return {
        "model": model,
        "val_metrics": val_metrics,
        "epochs_completed": epochs_completed,
        "time_seconds": total_time,
        "batch_size": batch_size,
        "n_params_m": n_params,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Benchmark: {len(BACKBONES)} backbones + EfficientNet-B0 (referência)")

    # Load data
    train_loader, val_loader, class_names = get_dataloaders(
        image_size=IMAGE_SIZE,
        batch_size=32,
        augmentation_level=AUGMENTATION_LEVEL,
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    class_weights = get_class_weights()

    # Load holdout
    holdout_records_full = load_holdout_records_full()
    holdout_loader, _ = load_holdout(class_names, IMAGE_SIZE, 16)
    print(f"Holdout: {len(holdout_records_full)} imagens\n")

    # Results storage
    all_results = {}

    # Add EfficientNet baseline
    all_results["efficientnet_b0"] = {
        "val_accuracy": EFFICIENTNET_BASELINE["val_accuracy"],
        "val_f1": EFFICIENTNET_BASELINE["val_f1"],
        "holdout_accuracy": EFFICIENTNET_BASELINE["holdout_accuracy"],
        "holdout_f1": EFFICIENTNET_BASELINE.get("holdout_f1"),
        "epochs_completed": EFFICIENTNET_BASELINE["epochs_completed"],
        "time_seconds": EFFICIENTNET_BASELINE["time_seconds"],
        "batch_size": EFFICIENTNET_BASELINE["batch_size"],
        "n_params_m": EFFICIENTNET_BASELINE["params_m"],
        "per_class": None,
        "per_source": None,
    }

    # Train each backbone
    for backbone_cfg in BACKBONES:
        model_name = backbone_cfg["name"]
        result = train_backbone(
            backbone_cfg, train_loader, val_loader, class_names, class_weights, device
        )
        model = result["model"]

        # Evaluate holdout
        print(f"\n  Avaliando holdout externo...")
        if holdout_loader:
            holdout_metrics = evaluate(model, holdout_loader, device, class_names)

            # Per-class and per-source breakdown
            per_class, per_source = holdout_per_class_source(
                model, holdout_records_full, class_names, device, IMAGE_SIZE
            )

            all_results[model_name] = {
                "val_accuracy": result["val_metrics"]["accuracy"],
                "val_f1": result["val_metrics"]["f1"],
                "val_precision": result["val_metrics"]["precision"],
                "val_recall": result["val_metrics"]["recall"],
                "holdout_accuracy": holdout_metrics["accuracy"],
                "holdout_f1": holdout_metrics["f1"],
                "holdout_precision": holdout_metrics["precision"],
                "holdout_recall": holdout_metrics["recall"],
                "epochs_completed": result["epochs_completed"],
                "time_seconds": result["time_seconds"],
                "batch_size": result["batch_size"],
                "n_params_m": result["n_params_m"],
                "per_class": per_class,
                "per_source": per_source,
                "confusion_matrix": holdout_metrics.get("confusion_matrix"),
            }
        else:
            all_results[model_name] = {
                "val_accuracy": result["val_metrics"]["accuracy"],
                "val_f1": result["val_metrics"]["f1"],
                "holdout_accuracy": None,
                "holdout_f1": None,
                "epochs_completed": result["epochs_completed"],
                "time_seconds": result["time_seconds"],
                "batch_size": result["batch_size"],
                "n_params_m": result["n_params_m"],
                "per_class": None,
                "per_source": None,
            }

        # Free memory
        del model
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()

    # -----------------------------------------------------------------------
    # Print final comparison table
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*80}")
    print(f"  COMPARATIVO FINAL — 3 BACKBONES")
    print(f"{'='*80}\n")

    header = f"  {'Modelo':<25} {'Params':>7} {'Val Acc':>8} {'Val F1':>7} {'Hold Acc':>9} {'Hold F1':>8} {'Epochs':>7} {'Tempo':>6} {'Batch':>6}"
    print(header)
    print("  " + "-" * 78)

    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1].get("holdout_accuracy") or 0,
        reverse=True,
    )

    best_holdout_name = sorted_models[0][0] if sorted_models else None

    for name, r in sorted_models:
        val_acc = f"{r['val_accuracy']*100:.1f}%" if r.get("val_accuracy") else "—"
        val_f1 = f"{r['val_f1']:.3f}" if r.get("val_f1") else "—"
        h_acc = f"{r['holdout_accuracy']*100:.1f}%" if r.get("holdout_accuracy") else "—"
        h_f1 = f"{r['holdout_f1']:.3f}" if r.get("holdout_f1") else "—"
        params = f"{r['n_params_m']:.1f}M"
        epochs = str(r["epochs_completed"])
        tempo = f"{r['time_seconds']:.0f}s"
        batch = str(r["batch_size"])
        marker = " *" if name == best_holdout_name else ""
        print(f"  {name:<25} {params:>7} {val_acc:>8} {val_f1:>7} {h_acc:>9} {h_f1:>8} {epochs:>7} {tempo:>6} {batch:>6}{marker}")

    print(f"\n  * = melhor holdout accuracy")

    # Per-class breakdown for trained models
    for name, r in sorted_models:
        if r.get("per_class"):
            print(f"\n  {name} — holdout por classe:")
            for cls, stats in sorted(r["per_class"].items()):
                acc = stats["correct"] / max(stats["total"], 1) * 100
                print(f"    {cls:<20} {stats['correct']:>3}/{stats['total']:<3} ({acc:.0f}%)")

    # Per-source breakdown
    for name, r in sorted_models:
        if r.get("per_source"):
            print(f"\n  {name} — holdout por fonte:")
            for src, stats in sorted(r["per_source"].items()):
                acc = stats["correct"] / max(stats["total"], 1) * 100
                print(f"    {src:<25} {stats['correct']:>3}/{stats['total']:<3} ({acc:.0f}%)")

    # Recommendation
    print(f"\n{'='*80}")
    print(f"  RECOMENDAÇÃO")
    print(f"{'='*80}")

    best = sorted_models[0]
    best_name, best_r = best
    print(f"\n  Melhor holdout: {best_name} ({best_r['holdout_accuracy']*100:.1f}%)")

    # Cost-efficiency
    for name, r in sorted_models:
        h_acc = r.get("holdout_accuracy") or 0
        params = r["n_params_m"]
        efficiency = h_acc / params * 100 if params > 0 else 0
        print(f"  {name:<25} eficiência: {efficiency:.2f} acc%/M-param")

    if best_name == "efficientnet_b0":
        print(f"\n  EfficientNet-B0 continua sendo a melhor escolha:")
        print(f"  - Melhor holdout accuracy")
        print(f"  - 5.3M params (mais leve, ideal pra mobile)")
        print(f"  - Treina 3 epochs no budget de 5 min")
    else:
        eff_holdout = all_results.get("efficientnet_b0", {}).get("holdout_accuracy", 0)
        gain = (best_r["holdout_accuracy"] - eff_holdout) * 100
        print(f"\n  {best_name} superou EfficientNet-B0 em {gain:.1f}pp no holdout.")
        print(f"  Porém tem {best_r['n_params_m']:.1f}M params vs 5.3M (custo maior).")
        if gain < 3:
            print(f"  Ganho marginal (<3pp) — EfficientNet-B0 ainda é recomendado pra produção.")
        else:
            print(f"  Ganho significativo — considerar {best_name} se o tamanho não for problema.")

    # -----------------------------------------------------------------------
    # Save reports
    # -----------------------------------------------------------------------

    # JSON
    json_path = os.path.join(SCRIPT_DIR, "benchmark_results.json")
    json_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "lr": LEARNING_RATE,
            "image_size": IMAGE_SIZE,
            "dropout": DROPOUT,
            "optimizer": OPTIMIZER,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "scheduler": SCHEDULER,
            "augmentation": AUGMENTATION_LEVEL,
            "budget_seconds": TRAINING_BUDGET_SECONDS,
        },
        "history": HISTORY,
        "results": {},
    }
    for name, r in all_results.items():
        entry = {k: v for k, v in r.items() if k not in ("per_class", "per_source")}
        if r.get("per_class"):
            entry["holdout_per_class"] = r["per_class"]
        if r.get("per_source"):
            entry["holdout_per_source"] = r["per_source"]
        json_data["results"][name] = entry

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON: {json_path}")

    # CSV
    csv_path = os.path.join(SCRIPT_DIR, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "params_m", "val_accuracy", "val_f1",
            "holdout_accuracy", "holdout_f1",
            "epochs", "time_s", "batch_size",
        ])
        for name, r in sorted_models:
            writer.writerow([
                name,
                f"{r['n_params_m']:.1f}",
                f"{r['val_accuracy']:.4f}" if r.get("val_accuracy") else "",
                f"{r['val_f1']:.4f}" if r.get("val_f1") else "",
                f"{r['holdout_accuracy']:.4f}" if r.get("holdout_accuracy") else "",
                f"{r['holdout_f1']:.4f}" if r.get("holdout_f1") else "",
                r["epochs_completed"],
                f"{r['time_seconds']:.0f}",
                r["batch_size"],
            ])
    print(f"  CSV:  {csv_path}")

    # Markdown report
    md_path = os.path.join(SCRIPT_DIR, "benchmark_report.md")
    with open(md_path, "w") as f:
        f.write("# Benchmark de Backbones — Soja Research\n\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Histórico de Experimentos\n\n")
        f.write("| Experimento | Holdout | Nota |\n")
        f.write("|---|---|---|\n")
        for h in HISTORY:
            f.write(f"| {h['experiment']} | {h['holdout_pct']}% | {h['note']} |\n")
        f.write("\n")

        f.write("## Comparativo de Backbones\n\n")
        f.write("| Modelo | Params | Val Acc | Val F1 | Holdout Acc | Holdout F1 | Epochs | Tempo | Batch |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for name, r in sorted_models:
            val_acc = f"{r['val_accuracy']*100:.1f}%" if r.get("val_accuracy") else "—"
            val_f1 = f"{r['val_f1']:.3f}" if r.get("val_f1") else "—"
            h_acc = f"{r['holdout_accuracy']*100:.1f}%" if r.get("holdout_accuracy") else "—"
            h_f1 = f"{r['holdout_f1']:.3f}" if r.get("holdout_f1") else "—"
            f.write(f"| {name} | {r['n_params_m']:.1f}M | {val_acc} | {val_f1} | {h_acc} | {h_f1} | {r['epochs_completed']} | {r['time_seconds']:.0f}s | {r['batch_size']} |\n")
        f.write("\n")

        # Per-class details
        for name, r in sorted_models:
            if r.get("per_class"):
                f.write(f"### {name} — Holdout por Classe\n\n")
                f.write("| Classe | Corretas | Total | Acc |\n")
                f.write("|---|---|---|---|\n")
                for cls, stats in sorted(r["per_class"].items()):
                    acc = stats["correct"] / max(stats["total"], 1) * 100
                    f.write(f"| {cls} | {stats['correct']} | {stats['total']} | {acc:.0f}% |\n")
                f.write("\n")

        # Per-source details
        for name, r in sorted_models:
            if r.get("per_source"):
                f.write(f"### {name} — Holdout por Fonte\n\n")
                f.write("| Fonte | Corretas | Total | Acc |\n")
                f.write("|---|---|---|---|\n")
                for src, stats in sorted(r["per_source"].items()):
                    acc = stats["correct"] / max(stats["total"], 1) * 100
                    f.write(f"| {src} | {stats['correct']} | {stats['total']} | {acc:.0f}% |\n")
                f.write("\n")

        f.write("## Recomendação\n\n")
        f.write(f"Melhor holdout: **{best_name}** ({best_r['holdout_accuracy']*100:.1f}%)\n\n")

        f.write("## Config\n\n")
        f.write(f"- LR: {LEARNING_RATE}\n")
        f.write(f"- Image size: {IMAGE_SIZE}\n")
        f.write(f"- Dropout: {DROPOUT}\n")
        f.write(f"- Optimizer: {OPTIMIZER}\n")
        f.write(f"- Scheduler: {SCHEDULER}\n")
        f.write(f"- Augmentation: {AUGMENTATION_LEVEL}\n")
        f.write(f"- Label smoothing: {LABEL_SMOOTHING}\n")
        f.write(f"- Budget: {TRAINING_BUDGET_SECONDS}s por modelo\n")

    print(f"  MD:   {md_path}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
