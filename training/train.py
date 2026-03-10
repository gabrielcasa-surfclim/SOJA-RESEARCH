"""
train.py — Treinamento de classificador de doenças em soja.

ARQUIVO EDITÁVEL — O agente autônomo modifica os hiperparâmetros abaixo.
Regra: modificar APENAS as variáveis entre os marcadores HYPERPARAMETERS.
"""

import csv
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import models

from prepare import evaluate, get_dataloaders, get_device

# =============================================================================
# ██  HYPERPARAMETERS — O agente autônomo modifica APENAS esta seção  ██
# =============================================================================

MODEL = "efficientnet_b0"           # efficientnet_b0 | efficientnet_b1 | mobilenet_v3_small | mobilenet_v3_large | resnet18 | resnet34
LEARNING_RATE = 0.0003              # 0.0001 — 0.01
BATCH_SIZE = 32                     # 8 | 16 | 32
EPOCHS = 3                          # será limitado pelo budget de tempo
IMAGE_SIZE = 224                    # 224 | 256 | 320
DROPOUT = 0.2                       # 0.0 — 0.5
OPTIMIZER = "adamw"                # adam | adamw | sgd
SCHEDULER = "cosine"               # cosine | step | onecycle | none
FREEZE_BACKBONE = False             # True = só treina classifier head | False = fine-tune completo
AUGMENTATION_LEVEL = "light"        # none | light | medium | heavy
WEIGHT_DECAY = 1e-4                 # 0 — 0.01
LABEL_SMOOTHING = 0.1              # 0.0 — 0.2

# =============================================================================
# ██  FIM DOS HYPERPARAMETERS                                            ██
# =============================================================================

TRAINING_BUDGET_SECONDS = 5 * 60    # 5 minutos de budget
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "results.tsv")
BEST_ACCURACY_FILE = os.path.join(os.path.dirname(__file__), "best_accuracy.txt")
BEST_MODEL_FILE = os.path.join(os.path.dirname(__file__), "best_model.pth")


def build_model(model_name: str, num_classes: int, dropout: float) -> nn.Module:
    """Carrega modelo pré-treinado e substitui classifier head."""

    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        if dropout > 0:
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(model.classifier[1].in_features, num_classes),
            )
        else:
            model.classifier = nn.Sequential(
                nn.Linear(model.classifier[1].in_features, num_classes),
            )

    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )

    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )

    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    return model


def freeze_backbone(model: nn.Module, model_name: str):
    """Congela todas as camadas exceto o classifier head."""
    for param in model.parameters():
        param.requires_grad = False

    # Descongela o head
    if model_name.startswith("efficientnet"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name.startswith("mobilenet"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name.startswith("resnet"):
        for param in model.fc.parameters():
            param.requires_grad = True


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Constrói optimizer baseado nos hiperparâmetros."""
    params = filter(lambda p: p.requires_grad, model.parameters())

    if OPTIMIZER == "adam":
        return torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adamw":
        return torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "sgd":
        return torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    else:
        raise ValueError(f"Optimizer desconhecido: {OPTIMIZER}")


def build_scheduler(optimizer, steps_per_epoch: int):
    """Constrói learning rate scheduler."""
    if SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * steps_per_epoch)
    elif SCHEDULER == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif SCHEDULER == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LEARNING_RATE, steps_per_epoch=steps_per_epoch, epochs=EPOCHS
        )
    elif SCHEDULER == "none":
        return None
    else:
        raise ValueError(f"Scheduler desconhecido: {SCHEDULER}")


def get_best_accuracy() -> float:
    """Lê melhor acurácia anterior do arquivo."""
    if os.path.exists(BEST_ACCURACY_FILE):
        with open(BEST_ACCURACY_FILE) as f:
            try:
                return float(f.read().strip())
            except ValueError:
                return 0.0
    return 0.0


def save_best_accuracy(accuracy: float):
    """Salva nova melhor acurácia."""
    with open(BEST_ACCURACY_FILE, "w") as f:
        f.write(f"{accuracy:.6f}\n")


def log_result(metrics: dict, elapsed: float, epochs_completed: int):
    """Loga resultado no TSV."""
    file_exists = os.path.exists(RESULTS_FILE)

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        if not file_exists:
            writer.writerow([
                "timestamp", "model", "lr", "batch_size", "epochs_completed", "image_size",
                "dropout", "optimizer", "scheduler", "freeze_backbone", "augmentation",
                "weight_decay", "label_smoothing",
                "accuracy", "precision", "recall", "f1", "loss",
                "elapsed_seconds", "improved",
            ])

        previous_best = get_best_accuracy()
        improved = metrics["accuracy"] > previous_best

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            MODEL, LEARNING_RATE, BATCH_SIZE, epochs_completed, IMAGE_SIZE,
            DROPOUT, OPTIMIZER, SCHEDULER, FREEZE_BACKBONE, AUGMENTATION_LEVEL,
            WEIGHT_DECAY, LABEL_SMOOTHING,
            f"{metrics['accuracy']:.6f}",
            f"{metrics['precision']:.6f}",
            f"{metrics['recall']:.6f}",
            f"{metrics['f1']:.6f}",
            f"{metrics['loss']:.6f}",
            f"{elapsed:.1f}",
            improved,
        ])


def train():
    """Loop principal de treinamento."""
    device = get_device()
    print(f"Device: {device}")
    print(f"Modelo: {MODEL} | LR: {LEARNING_RATE} | Batch: {BATCH_SIZE} | "
          f"Epochs: {EPOCHS} | ImgSize: {IMAGE_SIZE}")
    print(f"Dropout: {DROPOUT} | Optimizer: {OPTIMIZER} | Scheduler: {SCHEDULER}")
    print(f"Freeze: {FREEZE_BACKBONE} | Augmentation: {AUGMENTATION_LEVEL}")
    print(f"Weight Decay: {WEIGHT_DECAY} | Label Smoothing: {LABEL_SMOOTHING}")
    print(f"Budget: {TRAINING_BUDGET_SECONDS}s ({TRAINING_BUDGET_SECONDS/60:.0f} min)\n")

    # Carrega dados
    train_loader, val_loader, class_names = get_dataloaders(
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        augmentation_level=AUGMENTATION_LEVEL,
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}\n")

    # Monta modelo
    model = build_model(MODEL, num_classes, DROPOUT)
    if FREEZE_BACKBONE:
        freeze_backbone(model, MODEL)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Backbone congelado: {trainable:,} / {total:,} parâmetros treináveis")
    model = model.to(device)

    # Optimizer, scheduler, loss
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # Treina com budget de tempo
    start_time = time.time()
    epochs_completed = 0

    for epoch in range(EPOCHS):
        elapsed = time.time() - start_time
        if elapsed >= TRAINING_BUDGET_SECONDS:
            print(f"\n⏱  Budget de tempo atingido após {elapsed:.0f}s")
            break

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            # Checa budget a cada batch
            if time.time() - start_time >= TRAINING_BUDGET_SECONDS:
                break

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if scheduler and SCHEDULER != "step":
                scheduler.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if scheduler and SCHEDULER == "step":
            scheduler.step()

        epochs_completed = epoch + 1
        train_acc = correct / max(total, 1)
        avg_loss = running_loss / max(batch_idx + 1, 1)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1:>3}/{EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Tempo: {elapsed:.0f}s / {TRAINING_BUDGET_SECONDS}s")

    total_time = time.time() - start_time
    print(f"\nTreinamento finalizado: {epochs_completed} epochs em {total_time:.1f}s")

    # Avalia no validation set
    print("\nAvaliando no validation set...")
    metrics = evaluate(model, val_loader, device, class_names)

    # Compara com melhor resultado anterior
    previous_best = get_best_accuracy()
    current_accuracy = metrics["accuracy"]

    print(f"Acurácia atual:   {current_accuracy:.4f} ({current_accuracy*100:.1f}%)")
    print(f"Melhor anterior:  {previous_best:.4f} ({previous_best*100:.1f}%)")

    if current_accuracy > previous_best:
        print(f"\n✅ NOVO RECORDE! Melhorou {(current_accuracy - previous_best)*100:.2f}pp")
        save_best_accuracy(current_accuracy)
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_name": MODEL,
            "num_classes": num_classes,
            "class_names": class_names,
            "accuracy": current_accuracy,
            "metrics": metrics,
            "hyperparameters": {
                "model": MODEL, "lr": LEARNING_RATE, "batch_size": BATCH_SIZE,
                "epochs": epochs_completed, "image_size": IMAGE_SIZE, "dropout": DROPOUT,
                "optimizer": OPTIMIZER, "scheduler": SCHEDULER,
                "freeze_backbone": FREEZE_BACKBONE, "augmentation": AUGMENTATION_LEVEL,
                "weight_decay": WEIGHT_DECAY, "label_smoothing": LABEL_SMOOTHING,
            },
        }, BEST_MODEL_FILE)
        print(f"Modelo salvo em {BEST_MODEL_FILE}")
    else:
        print(f"\n❌ Não melhorou. Descartando.")

    # Loga resultado
    log_result(metrics, total_time, epochs_completed)
    print(f"Resultado logado em {RESULTS_FILE}")

    return metrics


if __name__ == "__main__":
    train()
