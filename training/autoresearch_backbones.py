"""
autoresearch_backbones.py — Autoresearch agent pra 3 backbones.

Para cada backbone (efficientnet_b0, convnext_base, maxvit_tiny_tf_224):
  - Roda 5 iterações variando LR, augmentation, dropout, scheduler
  - Salva melhor checkpoint de cada
  - Avalia holdout com breakdown por classe e fonte
  - Gera relatório comparativo final

Usa imagens segmentadas (data/images_segmented/) quando disponíveis.
Roda backbones sequencialmente com gc.collect entre eles.

Uso:
    python3 training/autoresearch_backbones.py
    python3 training/autoresearch_backbones.py --original   # força imagens originais

Tempo estimado: ~4h (3 backbones × 5 iterações × 15 min)
"""

import csv
import gc
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Configura NUM_WORKERS antes de importar prepare
os.environ["SOJA_NUM_WORKERS"] = "2"

import prepare
from prepare import (
    _ListDataset,
    _load_split_csv,
    evaluate,
    get_class_weights,
    get_dataloaders,
    get_device,
)
from train import build_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
IMAGE_SIZE = 224
TRAINING_BUDGET_SECONDS = 15 * 60  # 15 min por experimento
NUM_WORKERS = 2

# Usa imagens segmentadas se disponíveis (a menos que --original)
USE_SEGMENTED = "--original" not in (os.sys.argv[1:] if len(os.sys.argv) > 1 else [])
SEGMENTED_DIR = os.path.join(PROJECT_ROOT, "data", "images_segmented")

BACKBONES = [
    {"name": "efficientnet_b0", "batch_size": 32, "min_batch": 8, "epochs": 6},
    {"name": "convnext_base", "batch_size": 8, "min_batch": 4, "epochs": 6},
    {"name": "maxvit_tiny_tf_224", "batch_size": 12, "min_batch": 4, "epochs": 6},
]

# 5 configs de hiperparâmetros a testar por backbone
HYPERPARAM_GRID = [
    {
        "lr": 0.0003, "dropout": 0.2, "scheduler": "cosine",
        "augmentation": "light", "weight_decay": 1e-4,
        "label_smoothing": 0.1, "optimizer": "adamw",
    },
    {
        "lr": 0.0001, "dropout": 0.3, "scheduler": "cosine",
        "augmentation": "medium", "weight_decay": 1e-3,
        "label_smoothing": 0.1, "optimizer": "adamw",
    },
    {
        "lr": 0.0005, "dropout": 0.1, "scheduler": "onecycle",
        "augmentation": "light", "weight_decay": 1e-4,
        "label_smoothing": 0.05, "optimizer": "adamw",
    },
    {
        "lr": 0.001, "dropout": 0.2, "scheduler": "cosine",
        "augmentation": "medium", "weight_decay": 5e-4,
        "label_smoothing": 0.15, "optimizer": "sgd",
    },
    {
        "lr": 0.0002, "dropout": 0.4, "scheduler": "cosine",
        "augmentation": "heavy", "weight_decay": 1e-3,
        "label_smoothing": 0.1, "optimizer": "adamw",
    },
]

HISTORY = [
    {"experiment": "Random split (data leak)", "holdout_pct": 40.7},
    {"experiment": "Unificação classes EN/PT", "holdout_pct": 42.0},
    {"experiment": "Split C (folder group)", "holdout_pct": 47.3},
    {"experiment": "50/50 campo no treino", "holdout_pct": 63.8},
    {"experiment": "Segmentação HSV v1 (revertida)", "holdout_pct": 60.3},
    {"experiment": "Autoresearch EfficientNet (onecycle)", "holdout_pct": 67.2},
    {"experiment": "Benchmark 3 backbones", "holdout_pct": 67.2, "note": "eff>convnext>maxvit"},
]


# ---------------------------------------------------------------------------
# Holdout helpers
# ---------------------------------------------------------------------------
def load_holdout_records_full():
    csv_path = os.path.join(PROJECT_ROOT, "data", "splits", "holdout.csv")
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append((row["path"], row["class"], row.get("source", ""), row.get("folder", "")))
    return records


def _remap_to_segmented(path):
    """Remapeia path de images/ → images_segmented/ se disponível."""
    if not USE_SEGMENTED or not os.path.isdir(SEGMENTED_DIR):
        return path
    seg_path = path.replace("/data/images/", "/data/images_segmented/")
    return seg_path if os.path.exists(seg_path) else path


def make_holdout_loader(class_names, batch_size):
    holdout_records = _load_split_csv("holdout")
    if not holdout_records:
        return None
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    samples = []
    for path, cls in holdout_records:
        if cls not in class_to_idx:
            continue
        full_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
        full_path = _remap_to_segmented(full_path)
        samples.append((full_path, class_to_idx[cls]))
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = _ListDataset(samples, val_transform)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def holdout_per_class_source(model, holdout_records, class_names, device):
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model.eval()
    by_class = defaultdict(lambda: {"correct": 0, "total": 0})
    by_source = defaultdict(lambda: {"correct": 0, "total": 0})
    with torch.no_grad():
        for record in holdout_records:
            path, cls = record[0], record[1]
            if cls not in class_to_idx:
                continue
            label_idx = class_to_idx[cls]
            full_path = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
            full_path = _remap_to_segmented(full_path)
            try:
                img = Image.open(full_path).convert("RGB")
                img_t = val_transform(img).unsqueeze(0).to(device)
                pred = model(img_t).argmax(dim=1).item()
            except Exception:
                continue
            correct = int(pred == label_idx)
            by_class[cls]["correct"] += correct
            by_class[cls]["total"] += 1
            source = record[2] if len(record) > 2 else "unknown"
            by_source[source]["correct"] += correct
            by_source[source]["total"] += 1
    return dict(by_class), dict(by_source)


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------
def train_single(model_name, hp, batch_size, min_batch, max_epochs,
                 train_loader_cache, class_names, class_weights, device):
    """Treina um modelo com hiperparâmetros específicos. Retorna (model, val_metrics, holdout_metrics, info)."""
    num_classes = len(class_names)

    # Build model
    model = build_model(model_name, num_classes, hp["dropout"])
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    model = model.to(device)

    # Get or make dataloaders
    aug = hp["augmentation"]
    cache_key = (batch_size, aug)
    if cache_key not in train_loader_cache:
        tl, vl, _ = get_dataloaders(
            image_size=IMAGE_SIZE, batch_size=batch_size, augmentation_level=aug,
            num_workers=NUM_WORKERS,
        )
        train_loader_cache[cache_key] = (tl, vl)
    tl, vl = train_loader_cache[cache_key]

    # Optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if hp["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    elif hp["optimizer"] == "adam":
        optimizer = torch.optim.Adam(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    elif hp["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params, lr=hp["lr"], momentum=0.9, weight_decay=hp["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])

    # Scheduler
    steps_per_epoch = len(tl)
    if hp["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs * steps_per_epoch
        )
    elif hp["scheduler"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=hp["lr"], steps_per_epoch=steps_per_epoch, epochs=max_epochs,
        )
    elif hp["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    step_scheduler = hp["scheduler"] == "step"

    # Loss
    cw = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=hp["label_smoothing"])

    # Train loop
    start = time.time()
    epochs_done = 0
    oom_restart = False
    actual_batch = batch_size

    for epoch in range(max_epochs):
        if time.time() - start >= TRAINING_BUDGET_SECONDS:
            break

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        for images, labels in tl:
            if time.time() - start >= TRAINING_BUDGET_SECONDS:
                break
            images = images.to(device)
            labels = labels.to(device)

            try:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if scheduler and not step_scheduler:
                    scheduler.step()
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "MPS" in str(e):
                    if hasattr(torch, "mps"):
                        torch.mps.empty_cache()
                    new_batch = max(min_batch, actual_batch // 2)
                    if new_batch < actual_batch:
                        print(f"      OOM! batch {actual_batch} → {new_batch}")
                        actual_batch = new_batch
                        tl_new, vl_new, _ = get_dataloaders(
                            image_size=IMAGE_SIZE, batch_size=actual_batch,
                            augmentation_level=aug,
                        )
                        tl, vl = tl_new, vl_new
                        train_loader_cache[cache_key] = (tl, vl)
                        # Rebuild optimizer/scheduler with existing model
                        params = filter(lambda p: p.requires_grad, model.parameters())
                        if hp["optimizer"] == "sgd":
                            optimizer = torch.optim.SGD(params, lr=hp["lr"], momentum=0.9, weight_decay=hp["weight_decay"])
                        else:
                            optimizer = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
                        steps_per_epoch = len(tl)
                        if hp["scheduler"] == "cosine":
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                optimizer, T_max=max_epochs * steps_per_epoch
                            )
                        elif hp["scheduler"] == "onecycle":
                            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                optimizer, max_lr=hp["lr"],
                                steps_per_epoch=steps_per_epoch, epochs=max_epochs,
                            )
                        oom_restart = True
                        break
                    else:
                        raise
                else:
                    raise

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_count += 1

        if oom_restart:
            oom_restart = False
            continue  # retry epoch with smaller batch

        if scheduler and step_scheduler:
            scheduler.step()

        epochs_done = epoch + 1
        train_acc = correct / max(total, 1)
        avg_loss = running_loss / max(batch_count, 1)
        elapsed = time.time() - start
        print(f"      Epoch {epoch+1}/{max_epochs} | Loss: {avg_loss:.4f} | "
              f"Acc: {train_acc:.4f} | {elapsed:.0f}s")

    train_time = time.time() - start

    # Eval val
    val_metrics = evaluate(model, vl, device, class_names)

    # Eval holdout
    holdout_loader = make_holdout_loader(class_names, actual_batch)
    holdout_metrics = None
    if holdout_loader:
        holdout_metrics = evaluate(model, holdout_loader, device, class_names)

    info = {
        "epochs_completed": epochs_done,
        "time_seconds": train_time,
        "batch_size": actual_batch,
        "n_params_m": n_params,
        "hyperparams": hp,
    }

    return model, val_metrics, holdout_metrics, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = get_device()
    print(f"Device: {device}")

    # Redireciona prepare.py pra imagens segmentadas se disponíveis
    using_segmented = USE_SEGMENTED and os.path.isdir(SEGMENTED_DIR)
    if using_segmented:
        # Monkey-patch _load_split_csv pra remapear paths
        _original_load = prepare._load_split_csv

        def _patched_load(split_name):
            records = _original_load(split_name)
            remapped = []
            for path, cls in records:
                seg_path = path.replace("/data/images/", "/data/images_segmented/")
                remapped.append((seg_path if os.path.exists(seg_path) else path, cls))
            return remapped

        prepare._load_split_csv = _patched_load
        print(f"Paths remapeados → images_segmented/ (fallback → images/)")

    seg_status = "SEGMENTADAS" if using_segmented else "ORIGINAIS"
    print(f"Imagens: {seg_status}")
    print(f"Autoresearch: {len(BACKBONES)} backbones × {len(HYPERPARAM_GRID)} configs = "
          f"{len(BACKBONES) * len(HYPERPARAM_GRID)} experimentos")
    print(f"Budget: {TRAINING_BUDGET_SECONDS}s ({TRAINING_BUDGET_SECONDS//60} min) por experimento")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    total_est = len(BACKBONES) * len(HYPERPARAM_GRID) * TRAINING_BUDGET_SECONDS / 60
    print(f"Tempo máximo estimado: {total_est:.0f} min ({total_est/60:.1f}h)\n")

    # Load class info
    train_loader, val_loader, class_names = get_dataloaders(
        image_size=IMAGE_SIZE, batch_size=32, augmentation_level="light",
        num_workers=NUM_WORKERS,
    )
    num_classes = len(class_names)
    class_weights = get_class_weights()
    holdout_records_full = load_holdout_records_full()
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Holdout: {len(holdout_records_full)} imagens\n")

    # Results per backbone
    all_results = {}  # backbone_name -> best result dict
    all_experiments = []  # every experiment for the log
    global_start = time.time()

    for backbone in BACKBONES:
        bname = backbone["name"]
        batch_size = backbone["batch_size"]
        min_batch = backbone["min_batch"]
        max_epochs = backbone["epochs"]

        print(f"\n{'#'*80}")
        print(f"  BACKBONE: {bname} (batch={batch_size}, epochs={max_epochs})")
        print(f"{'#'*80}")

        best_holdout_acc = -1.0
        best_result = None
        best_model_state = None
        train_loader_cache = {}

        for i, hp in enumerate(HYPERPARAM_GRID):
            print(f"\n  --- Iteração {i+1}/{len(HYPERPARAM_GRID)} ---")
            print(f"    LR={hp['lr']} dropout={hp['dropout']} sched={hp['scheduler']} "
                  f"aug={hp['augmentation']} optim={hp['optimizer']} "
                  f"wd={hp['weight_decay']} ls={hp['label_smoothing']}")

            try:
                model, val_m, hold_m, info = train_single(
                    bname, hp, batch_size, min_batch, max_epochs,
                    train_loader_cache, class_names, class_weights, device,
                )
            except Exception as e:
                print(f"    ERRO: {e}")
                all_experiments.append({
                    "backbone": bname, "iteration": i + 1,
                    "hyperparams": hp, "error": str(e),
                })
                continue

            h_acc = hold_m["accuracy"] if hold_m else 0.0
            h_f1 = hold_m["f1"] if hold_m else 0.0

            experiment_record = {
                "backbone": bname,
                "iteration": i + 1,
                "hyperparams": hp,
                "val_accuracy": val_m["accuracy"],
                "val_f1": val_m["f1"],
                "holdout_accuracy": h_acc,
                "holdout_f1": h_f1,
                "epochs_completed": info["epochs_completed"],
                "time_seconds": info["time_seconds"],
                "batch_size": info["batch_size"],
            }
            all_experiments.append(experiment_record)

            marker = ""
            if h_acc > best_holdout_acc:
                best_holdout_acc = h_acc
                best_result = {
                    **experiment_record,
                    "val_metrics": val_m,
                    "holdout_metrics": hold_m,
                    "n_params_m": info["n_params_m"],
                }
                best_model_state = model.state_dict().copy()
                marker = " *** NEW BEST ***"

            print(f"    Val: {val_m['accuracy']*100:.1f}% | "
                  f"Holdout: {h_acc*100:.1f}% (F1={h_f1:.3f}) | "
                  f"{info['epochs_completed']}ep {info['time_seconds']:.0f}s{marker}")

            # Free memory
            del model
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()

        # Save best checkpoint
        if best_model_state is not None and best_result is not None:
            ckpt_name = f"best_{bname}.pth" if bname != "maxvit_tiny_tf_224" else "best_maxvit_tiny.pth"
            ckpt_path = os.path.join(SCRIPT_DIR, ckpt_name)
            torch.save({
                "model_state_dict": best_model_state,
                "model_name": bname,
                "num_classes": num_classes,
                "class_names": class_names,
                "accuracy": best_result["val_accuracy"],
                "holdout_accuracy": best_result["holdout_accuracy"],
                "hyperparams": best_result["hyperparams"],
            }, ckpt_path)
            print(f"\n  Melhor {bname}: holdout={best_holdout_acc*100:.1f}% → {ckpt_path}")

            # Per-class/source breakdown for the best model
            # Reload model to evaluate
            best_model = build_model(bname, num_classes, best_result["hyperparams"]["dropout"])
            best_model.load_state_dict(best_model_state)
            best_model = best_model.to(device)
            per_class, per_source = holdout_per_class_source(
                best_model, holdout_records_full, class_names, device,
            )
            best_result["per_class"] = per_class
            best_result["per_source"] = per_source
            del best_model
            if hasattr(torch, "mps"):
                torch.mps.empty_cache()

            all_results[bname] = best_result
        else:
            print(f"\n  {bname}: nenhum resultado válido!")

        # Libera memória entre backbones
        del best_model_state, train_loader_cache
        gc.collect()
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
        print(f"\n  Memória liberada (gc.collect + mps.empty_cache)")

    total_time = time.time() - global_start

    # -----------------------------------------------------------------------
    # Final comparison table
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*90}")
    print(f"  AUTORESEARCH — COMPARATIVO FINAL ({total_time/60:.0f} min total)")
    print(f"{'='*90}\n")

    header = (f"  {'Backbone':<25} {'Params':>7} {'Val Acc':>8} {'Val F1':>7} "
              f"{'Hold Acc':>9} {'Hold F1':>8} {'Epochs':>7} {'Tempo':>6} {'Config':>30}")
    print(header)
    print("  " + "-" * 88)

    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1].get("holdout_accuracy", 0),
        reverse=True,
    )

    for name, r in sorted_models:
        hp = r["hyperparams"]
        config_str = f"lr={hp['lr']} d={hp['dropout']} {hp['scheduler'][:3]} {hp['augmentation'][:3]}"
        val_acc = f"{r['val_accuracy']*100:.1f}%"
        val_f1 = f"{r['val_f1']:.3f}"
        h_acc = f"{r['holdout_accuracy']*100:.1f}%"
        h_f1 = f"{r['holdout_f1']:.3f}"
        params = f"{r['n_params_m']:.1f}M"
        epochs = str(r["epochs_completed"])
        tempo = f"{r['time_seconds']:.0f}s"
        marker = " *" if name == sorted_models[0][0] else ""
        print(f"  {name:<25} {params:>7} {val_acc:>8} {val_f1:>7} "
              f"{h_acc:>9} {h_f1:>8} {epochs:>7} {tempo:>6} {config_str:>30}{marker}")

    print(f"\n  * = melhor holdout accuracy")

    # Per-class
    for name, r in sorted_models:
        if r.get("per_class"):
            print(f"\n  {name} — holdout por classe:")
            for cls in sorted(r["per_class"]):
                s = r["per_class"][cls]
                acc = s["correct"] / max(s["total"], 1) * 100
                print(f"    {cls:<20} {s['correct']:>3}/{s['total']:<3} ({acc:.0f}%)")

    # Per-source
    for name, r in sorted_models:
        if r.get("per_source"):
            print(f"\n  {name} — holdout por fonte:")
            for src in sorted(r["per_source"]):
                s = r["per_source"][src]
                acc = s["correct"] / max(s["total"], 1) * 100
                print(f"    {src:<25} {s['correct']:>3}/{s['total']:<3} ({acc:.0f}%)")

    # All experiments log
    print(f"\n  {'─'*90}")
    print(f"  HISTÓRICO COMPLETO ({len(all_experiments)} experimentos):")
    print(f"  {'─'*90}")
    print(f"  {'#':>3} {'Backbone':<25} {'LR':>8} {'Drop':>5} {'Sched':>8} {'Aug':>6} "
          f"{'Val':>6} {'Hold':>6} {'Ep':>3} {'Tempo':>6}")
    print(f"  {'─'*90}")

    for i, exp in enumerate(all_experiments):
        if "error" in exp:
            print(f"  {i+1:>3} {exp['backbone']:<25} ERRO: {exp['error'][:50]}")
            continue
        hp = exp["hyperparams"]
        print(f"  {i+1:>3} {exp['backbone']:<25} {hp['lr']:>8.4f} {hp['dropout']:>5.2f} "
              f"{hp['scheduler']:>8} {hp['augmentation']:>6} "
              f"{exp['val_accuracy']*100:>5.1f}% {exp['holdout_accuracy']*100:>5.1f}% "
              f"{exp['epochs_completed']:>3} {exp['time_seconds']:>5.0f}s")

    # Recommendation
    print(f"\n{'='*90}")
    print(f"  RECOMENDAÇÃO")
    print(f"{'='*90}")
    if sorted_models:
        best_name, best_r = sorted_models[0]
        print(f"\n  Melhor holdout: {best_name} ({best_r['holdout_accuracy']*100:.1f}%)")
        print(f"  Config: LR={best_r['hyperparams']['lr']} dropout={best_r['hyperparams']['dropout']} "
              f"scheduler={best_r['hyperparams']['scheduler']} "
              f"augmentation={best_r['hyperparams']['augmentation']}")
        for name, r in sorted_models:
            eff = r["holdout_accuracy"] / r["n_params_m"] * 100 if r["n_params_m"] > 0 else 0
            print(f"  {name:<25} eficiência: {eff:.2f} holdout%/M-param")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    json_path = os.path.join(SCRIPT_DIR, "autoresearch_results.json")
    json_data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total_time_minutes": total_time / 60,
        "budget_per_experiment_seconds": TRAINING_BUDGET_SECONDS,
        "image_size": IMAGE_SIZE,
        "history": HISTORY,
        "best_per_backbone": {},
        "all_experiments": [],
    }
    for name, r in all_results.items():
        entry = {
            "val_accuracy": r["val_accuracy"],
            "val_f1": r["val_f1"],
            "holdout_accuracy": r["holdout_accuracy"],
            "holdout_f1": r["holdout_f1"],
            "n_params_m": r["n_params_m"],
            "epochs_completed": r["epochs_completed"],
            "time_seconds": r["time_seconds"],
            "batch_size": r["batch_size"],
            "hyperparams": r["hyperparams"],
        }
        if r.get("per_class"):
            entry["holdout_per_class"] = r["per_class"]
        if r.get("per_source"):
            entry["holdout_per_source"] = r["per_source"]
        json_data["best_per_backbone"][name] = entry

    for exp in all_experiments:
        json_data["all_experiments"].append({
            k: v for k, v in exp.items() if k not in ("val_metrics", "holdout_metrics")
        })

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON: {json_path}")

    # -----------------------------------------------------------------------
    # Save markdown report
    # -----------------------------------------------------------------------
    md_path = os.path.join(SCRIPT_DIR, "autoresearch_report.md")
    with open(md_path, "w") as f:
        f.write("# Autoresearch Backbones — Soja Research\n\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Tempo total: {total_time/60:.0f} min\n\n")

        f.write("## Histórico de Holdout\n\n")
        f.write("| Experimento | Holdout |\n|---|---|\n")
        for h in HISTORY:
            f.write(f"| {h['experiment']} | {h['holdout_pct']}% |\n")
        f.write("\n")

        f.write("## Melhor Config por Backbone\n\n")
        f.write("| Backbone | Params | Val Acc | Val F1 | Holdout Acc | Holdout F1 | Config |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name, r in sorted_models:
            hp = r["hyperparams"]
            cfg = f"lr={hp['lr']} d={hp['dropout']} {hp['scheduler']} {hp['augmentation']}"
            f.write(f"| {name} | {r['n_params_m']:.1f}M | "
                    f"{r['val_accuracy']*100:.1f}% | {r['val_f1']:.3f} | "
                    f"{r['holdout_accuracy']*100:.1f}% | {r['holdout_f1']:.3f} | {cfg} |\n")
        f.write("\n")

        # Per-class
        for name, r in sorted_models:
            if r.get("per_class"):
                f.write(f"### {name} — Holdout por Classe\n\n")
                f.write("| Classe | Corretas | Total | Acc |\n|---|---|---|---|\n")
                for cls in sorted(r["per_class"]):
                    s = r["per_class"][cls]
                    acc = s["correct"] / max(s["total"], 1) * 100
                    f.write(f"| {cls} | {s['correct']} | {s['total']} | {acc:.0f}% |\n")
                f.write("\n")

        # Per-source
        for name, r in sorted_models:
            if r.get("per_source"):
                f.write(f"### {name} — Holdout por Fonte\n\n")
                f.write("| Fonte | Corretas | Total | Acc |\n|---|---|---|---|\n")
                for src in sorted(r["per_source"]):
                    s = r["per_source"][src]
                    acc = s["correct"] / max(s["total"], 1) * 100
                    f.write(f"| {src} | {s['correct']} | {s['total']} | {acc:.0f}% |\n")
                f.write("\n")

        f.write("## Todos os Experimentos\n\n")
        f.write("| # | Backbone | LR | Dropout | Scheduler | Aug | Val Acc | Holdout Acc | Epochs | Tempo |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for i, exp in enumerate(all_experiments):
            if "error" in exp:
                f.write(f"| {i+1} | {exp['backbone']} | — | — | — | — | ERRO | — | — | — |\n")
                continue
            hp = exp["hyperparams"]
            f.write(f"| {i+1} | {exp['backbone']} | {hp['lr']} | {hp['dropout']} | "
                    f"{hp['scheduler']} | {hp['augmentation']} | "
                    f"{exp['val_accuracy']*100:.1f}% | {exp['holdout_accuracy']*100:.1f}% | "
                    f"{exp['epochs_completed']} | {exp['time_seconds']:.0f}s |\n")
        f.write("\n")

        if sorted_models:
            best_name, best_r = sorted_models[0]
            f.write(f"## Recomendação\n\n")
            f.write(f"Melhor holdout: **{best_name}** ({best_r['holdout_accuracy']*100:.1f}%)\n")

    print(f"  MD:   {md_path}")
    print(f"\n{'='*90}\n")


if __name__ == "__main__":
    main()
