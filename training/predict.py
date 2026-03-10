"""
predict.py — Classifica uma imagem de folha de soja usando o modelo treinado.

Uso:
    python3 training/predict.py <caminho_ou_url_da_imagem>

Exemplos:
    python3 training/predict.py foto_folha.jpg
    python3 training/predict.py https://example.com/ferrugem.jpg
"""

import os
import sys
import subprocess
import tempfile

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train import build_model

BEST_MODEL_FILE = os.path.join(os.path.dirname(__file__), "best_model.pth")


def load_model(checkpoint_path: str, device: torch.device):
    """Carrega modelo a partir do checkpoint salvo."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_name = checkpoint["model_name"]
    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("hyperparameters", {}).get("image_size", 224)
    accuracy = checkpoint.get("accuracy", 0.0)

    # Reconstrói modelo com mesma arquitetura (dropout=0 na inferência não importa)
    dropout = checkpoint.get("hyperparameters", {}).get("dropout", 0.2)
    model = build_model(model_name, num_classes, dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names, image_size, accuracy


def get_transform(image_size: int) -> transforms.Compose:
    """Mesmo transform de validação do prepare.py."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def download_image(url: str) -> str:
    """Baixa imagem de URL usando curl e retorna caminho do arquivo temporário."""
    suffix = os.path.splitext(url.split("?")[0])[-1] or ".jpg"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    print(f"Baixando imagem de: {url}")
    result = subprocess.run(
        ["curl", "-sL", "-o", tmp_path, url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        os.unlink(tmp_path)
        raise RuntimeError(f"Erro ao baixar imagem: {result.stderr}")

    # Verifica se o arquivo tem conteúdo
    if os.path.getsize(tmp_path) == 0:
        os.unlink(tmp_path)
        raise RuntimeError("Arquivo baixado está vazio")

    print(f"Salvo em: {tmp_path}")
    return tmp_path


def predict(image_path: str):
    """Classifica uma imagem e mostra top-3 classes com probabilidade."""
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Carrega modelo
    if not os.path.exists(BEST_MODEL_FILE):
        print(f"Erro: modelo não encontrado em {BEST_MODEL_FILE}")
        print("Rode o treinamento primeiro: python3 training/train.py")
        sys.exit(1)

    model, class_names, image_size, best_accuracy = load_model(BEST_MODEL_FILE, device)
    print(f"Modelo carregado: {len(class_names)} classes, acurácia val: {best_accuracy*100:.1f}%")
    print(f"Classes: {class_names}\n")

    # Baixa se for URL
    tmp_file = None
    if is_url(image_path):
        tmp_file = download_image(image_path)
        image_path = tmp_file

    try:
        # Carrega e transforma imagem
        image = Image.open(image_path).convert("RGB")
        print(f"Imagem: {image.size[0]}x{image.size[1]}px")

        transform = get_transform(image_size)
        tensor = transform(image).unsqueeze(0).to(device)

        # Predição
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)[0]

        # Top-3
        top3_probs, top3_indices = torch.topk(probs, min(3, len(class_names)))

        print(f"\n{'='*40}")
        print(f"  DIAGNÓSTICO")
        print(f"{'='*40}")
        for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
            name = class_names[idx.item()]
            p = prob.item() * 100
            bar = "█" * int(p / 5) + "░" * (20 - int(p / 5))
            marker = " ◄" if i == 0 else ""
            print(f"  {i+1}. {name:<20} {p:>5.1f}%  {bar}{marker}")
        print(f"{'='*40}")

    finally:
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python3 training/predict.py <caminho_ou_url_da_imagem>")
        sys.exit(1)

    predict(sys.argv[1])
