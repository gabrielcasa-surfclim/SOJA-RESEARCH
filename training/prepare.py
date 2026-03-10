"""
prepare.py — Dataset loading, augmentation e evaluation para classificação de doenças em soja.

ARQUIVO FIXO — O agente autônomo NÃO deve modificar este arquivo.
Qualquer mudança aqui invalida a comparação entre experimentos.
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# =============================================================================
# Mapeamento de pastas → classes normalizadas
# Várias pastas podem mapear pra mesma classe (ex: digipathos_ferrugem + digipathos_ferrugem_crop)
# =============================================================================

FOLDER_TO_CLASS = {
    "digipathos_ferrugem": "Ferrugem",
    "digipathos_ferrugem_crop": "Ferrugem",
    "soybean_rust": "Ferrugem",
    "digipathos_oidio": "Oídio",
    "digipathos_oidio_crop": "Oídio",
    "digipathos_mancha_alvo": "Mancha-alvo",
    "digipathos_mancha_alvo_crop": "Mancha-alvo",
    "digipathos_cercospora": "Cercospora",
    "digipathos_cercospora_crop": "Cercospora",
    "digipathos_antracnose": "Antracnose",
    "digipathos_mosaico": "Mosaico",
    "digipathos_mosaico_crop": "Mosaico",
    "digipathos_saudavel": "Saudável",
    "healthy": "Saudável",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# =============================================================================
# Dataset customizado que agrupa subpastas por classe
# =============================================================================


class SoybeanDiseaseDataset(Dataset):
    """Dataset que carrega imagens de data/images/ e agrupa subpastas por doença."""

    def __init__(self, data_dir: str = DATA_DIR, transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []  # (path, class_idx)
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.classes: List[str] = []

        # Descobre quais pastas existem e mapeia pra classes
        class_samples = defaultdict(list)

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Diretório de dados não encontrado: {data_dir}")

        for folder_name in sorted(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Usa mapeamento conhecido ou normaliza o nome da pasta
            class_name = FOLDER_TO_CLASS.get(folder_name, _normalize_folder_name(folder_name))

            for fname in os.listdir(folder_path):
                ext = os.path.splitext(fname)[1].lower()
                if ext in VALID_EXTENSIONS:
                    class_samples[class_name].append(os.path.join(folder_path, fname))

        # Ordena classes e atribui índices
        self.classes = sorted(class_samples.keys())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        for class_name in self.classes:
            idx = self.class_to_idx[class_name]
            for path in class_samples[class_name]:
                self.samples.append((path, idx))

        print(f"Dataset carregado: {len(self.samples)} imagens, {len(self.classes)} classes")
        for cls in self.classes:
            count = sum(1 for _, c in self.samples if c == self.class_to_idx[cls])
            print(f"  {cls}: {count} imagens")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _normalize_folder_name(name: str) -> str:
    """Normaliza nome de pasta desconhecida pra usar como classe."""
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\bcrop\b", "", name).strip()
    return name.title()


# =============================================================================
# Transforms (augmentation)
# =============================================================================


def get_train_transform(image_size: int = 224, augmentation_level: str = "medium") -> transforms.Compose:
    """Retorna transform de treino com data augmentation configurável."""

    base = [
        transforms.Resize((image_size, image_size)),
    ]

    if augmentation_level == "light":
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    elif augmentation_level == "medium":
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]
    elif augmentation_level == "heavy":
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        ]
    else:
        aug = []

    post = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if augmentation_level == "heavy":
        post.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)))

    return transforms.Compose(base + aug + post)


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """Transform de validação — apenas resize + normalize, sem augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# =============================================================================
# DataLoaders com split estratificado
# =============================================================================


def get_dataloaders(
    image_size: int = 224,
    batch_size: int = 16,
    augmentation_level: str = "medium",
    val_ratio: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Carrega dataset, divide 80/20 estratificado, retorna (train_loader, val_loader, class_names).
    """
    # Carrega dataset sem transform (vamos aplicar depois via subsets)
    full_dataset = SoybeanDiseaseDataset(data_dir=DATA_DIR, transform=None)

    if len(full_dataset) == 0:
        raise RuntimeError("Dataset vazio! Verifique se há imagens em data/images/")

    # Split estratificado
    labels = [label for _, label in full_dataset.samples]
    train_indices, val_indices = _stratified_split(labels, val_ratio, seed)

    print(f"\nSplit: {len(train_indices)} treino, {len(val_indices)} validação")

    # Datasets com transforms diferentes
    train_transform = get_train_transform(image_size, augmentation_level)
    val_transform = get_val_transform(image_size)

    train_dataset = _TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = _TransformSubset(full_dataset, val_indices, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, full_dataset.classes


def _stratified_split(labels: List[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """Split estratificado manual — mantém proporção de classes."""
    rng = np.random.RandomState(seed)

    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    train_indices = []
    val_indices = []

    for label in sorted(label_to_indices.keys()):
        indices = label_to_indices[label]
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    return train_indices, val_indices


class _TransformSubset(Dataset):
    """Subset com transform próprio."""

    def __init__(self, dataset: SoybeanDiseaseDataset, indices: List[int], transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# =============================================================================
# Avaliação
# =============================================================================


def get_device() -> torch.device:
    """Retorna o melhor device disponível (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str] = None,
) -> Dict:
    """
    Avalia modelo no dataloader e retorna métricas completas.

    Returns:
        dict com accuracy, precision, recall, f1, confusion_matrix, loss
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        n_batches += 1

        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    avg_loss = total_loss / max(n_batches, 1)

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "loss": float(avg_loss),
    }

    # Print formatado
    print(f"\n{'='*50}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Loss:      {avg_loss:.4f}")

    if class_names:
        print(f"\n  Confusion Matrix:")
        header = "          " + "  ".join(f"{n[:8]:>8}" for n in class_names)
        print(header)
        for i, row in enumerate(cm):
            name = class_names[i][:8]
            vals = "  ".join(f"{v:>8}" for v in row)
            print(f"  {name:>8}  {vals}")

    print(f"{'='*50}\n")

    return results


if __name__ == "__main__":
    # Teste rápido: carrega dataset e mostra stats
    device = get_device()
    print(f"Device: {device}")
    train_loader, val_loader, classes = get_dataloaders(image_size=224, batch_size=16)
    print(f"Classes: {classes}")
    print(f"Batches treino: {len(train_loader)}, validação: {len(val_loader)}")
