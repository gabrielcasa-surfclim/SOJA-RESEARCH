"""
prepare.py — Dataset loading, augmentation e evaluation para classificação de doenças em soja.

ARQUIVO FIXO — O agente autônomo NÃO deve modificar este arquivo.
Qualquer mudança aqui invalida a comparação entre experimentos.
"""

import csv
import json
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
    # Digipathos (EMBRAPA)
    "digipathos_ferrugem": "Ferrugem",
    "digipathos_ferrugem_crop": "Ferrugem",
    "soybean_rust": "Ferrugem",
    "digipathos_oidio": "Oídio",
    "digipathos_oidio_crop": "Oídio",
    "digipathos_mancha_alvo": "Mancha-alvo",
    "digipathos_mancha_alvo_crop": "Mancha-alvo",
    "target_spot": "Mancha-alvo",                      # ASDID — era "Target Spot"
    "digipathos_cercospora": "Olho-de-rã",             # Cercospora sojina = Frogeye
    "digipathos_cercospora_crop": "Olho-de-rã",
    "frogeye": "Olho-de-rã",                           # ASDID — era "Frogeye"
    "digipathos_antracnose": "Antracnose",
    "digipathos_mosaico": "Mosaico",
    "digipathos_mosaico_crop": "Mosaico",
    "digipathos_saudavel": "Saudável",
    "healthy": "Saudável",
    # doencasdeplantas.com.br
    "doencasdeplantas_ferrugem_asiatica": "Ferrugem",
    "doencasdeplantas_oidio": "Oídio",
    "doencasdeplantas_mancha_alvo": "Mancha-alvo",
    "doencasdeplantas_mancha_olho_de_ra": "Olho-de-rã",
    "doencasdeplantas_antracnose": "Antracnose",
    "doencasdeplantas_virose_do_mosaico_comum": "Mosaico",
    "doencasdeplantas_virose_do_mosaico_rugoso": "Mosaico",
    "doencasdeplantas_cercosporiose": "Olho-de-rã",
    # soybeanresearchinfo.com (SRIN)
    "srin_anthracnose": "Antracnose",
    "srin_cercospora_leaf_blight": "Olho-de-rã",
    "srin_frogeye_leaf_spot": "Olho-de-rã",
    "srin_powdery_mildew": "Oídio",
    "srin_viruses": "Mosaico",
}

_DATA_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(_DATA_ROOT, "data", "images")
SPLITS_DIR = os.path.join(_DATA_ROOT, "data", "splits")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

MIN_SAMPLES = 50  # Classes com menos imagens que isso são ignoradas no treino


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

        # Filtra classes com poucas imagens
        skipped = {cls: len(paths) for cls, paths in class_samples.items() if len(paths) < MIN_SAMPLES}
        if skipped:
            print(f"Classes ignoradas (< {MIN_SAMPLES} imagens):")
            for cls, count in sorted(skipped.items()):
                print(f"  {cls}: {count} imagens (ignorada)")
            for cls in skipped:
                del class_samples[cls]

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


def _load_split_csv(split_name: str) -> List[Tuple[str, str]]:
    """Carrega um CSV de split e retorna lista de (path, class)."""
    csv_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    records = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append((row["path"], row["class"]))
    return records


def _load_class_weights() -> Dict[str, float]:
    """Carrega class_weights.json do diretório de splits."""
    weights_path = os.path.join(SPLITS_DIR, "class_weights.json")
    with open(weights_path, "r") as f:
        return json.load(f)


def _splits_available() -> bool:
    """Verifica se os CSVs de split existem."""
    return all(
        os.path.exists(os.path.join(SPLITS_DIR, f))
        for f in ["train.csv", "val.csv", "class_weights.json"]
    )


class _ListDataset(Dataset):
    """Dataset a partir de lista de (path, label_idx)."""

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(
    image_size: int = 224,
    batch_size: int = 16,
    augmentation_level: str = "medium",
    val_ratio: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Carrega dataset e retorna (train_loader, val_loader, class_names).

    Se data/splits/ existir (gerado por create_splits.py), usa os splits CSV
    com separação por fonte. Caso contrário, faz split aleatório estratificado.
    """
    train_transform = get_train_transform(image_size, augmentation_level)
    val_transform = get_val_transform(image_size)

    if _splits_available():
        return _get_dataloaders_from_splits(
            train_transform, val_transform, batch_size, num_workers
        )

    # Fallback: split aleatório estratificado (comportamento antigo)
    full_dataset = SoybeanDiseaseDataset(data_dir=DATA_DIR, transform=None)

    if len(full_dataset) == 0:
        raise RuntimeError("Dataset vazio! Verifique se há imagens em data/images/")

    labels = [label for _, label in full_dataset.samples]
    train_indices, val_indices = _stratified_split(labels, val_ratio, seed)

    print(f"\nSplit: {len(train_indices)} treino, {len(val_indices)} validação (random)")

    train_dataset = _TransformSubset(full_dataset, train_indices, train_transform)
    val_dataset = _TransformSubset(full_dataset, val_indices, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, full_dataset.classes


def _get_dataloaders_from_splits(
    train_transform, val_transform, batch_size: int, num_workers: int,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Carrega dataloaders a partir dos CSVs de split."""
    print("Usando splits de data/splits/ (source-based)")

    train_records = _load_split_csv("train")
    val_records = _load_split_csv("val")

    # Descobre classes a partir dos dados
    all_classes = sorted(set(cls for _, cls in train_records + val_records))
    class_to_idx = {name: idx for idx, name in enumerate(all_classes)}

    # Filtra arquivos que existem
    train_samples = []
    for path, cls in train_records:
        if os.path.exists(path) and cls in class_to_idx:
            train_samples.append((path, class_to_idx[cls]))

    val_samples = []
    for path, cls in val_records:
        if os.path.exists(path) and cls in class_to_idx:
            val_samples.append((path, class_to_idx[cls]))

    print(f"\nSplit: {len(train_samples)} treino, {len(val_samples)} validação (source-based)")
    for cls in all_classes:
        idx = class_to_idx[cls]
        t = sum(1 for _, c in train_samples if c == idx)
        v = sum(1 for _, c in val_samples if c == idx)
        print(f"  {cls}: train={t}, val={v}")

    train_dataset = _ListDataset(train_samples, train_transform)
    val_dataset = _ListDataset(val_samples, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, all_classes


def get_class_weights() -> torch.Tensor | None:
    """Retorna tensor de class weights para CrossEntropyLoss, ou None se não disponível."""
    if not _splits_available():
        return None

    weights_dict = _load_class_weights()
    # Ordena pela mesma ordem das classes
    train_records = _load_split_csv("train")
    all_classes = sorted(set(cls for _, cls in train_records))

    weights = [weights_dict.get(cls, 1.0) for cls in all_classes]
    return torch.tensor(weights, dtype=torch.float32)


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
