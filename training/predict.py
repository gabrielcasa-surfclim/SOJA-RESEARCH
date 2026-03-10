"""
predict.py — Classifica uma imagem de folha de soja usando o modelo treinado.

Uso CLI:
    python3 training/predict.py <caminho_ou_url_da_imagem>

Uso como módulo:
    from predict import predict_image
    result = predict_image("/path/to/image.jpg")
    # ou
    result = predict_image_bytes(image_bytes)
"""

import base64
import io
import json
import os
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train import build_model

BEST_MODEL_FILE = os.path.join(os.path.dirname(__file__), "best_model.pth")

# Singleton — modelo carregado uma vez e reutilizado
_model_cache = {}


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_model():
    """Carrega modelo uma vez e mantém em cache."""
    if _model_cache:
        return _model_cache

    device = _get_device()

    if not os.path.exists(BEST_MODEL_FILE):
        raise FileNotFoundError(f"Modelo não encontrado: {BEST_MODEL_FILE}")

    checkpoint = torch.load(BEST_MODEL_FILE, map_location=device, weights_only=False)

    model_name = checkpoint["model_name"]
    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("hyperparameters", {}).get("image_size", 224)
    accuracy = checkpoint.get("accuracy", 0.0)
    dropout = checkpoint.get("hyperparameters", {}).get("dropout", 0.2)

    model = build_model(model_name, num_classes, dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _model_cache.update({
        "model": model,
        "class_names": class_names,
        "transform": transform,
        "device": device,
        "accuracy": accuracy,
    })

    return _model_cache


def _classify(image: Image.Image) -> dict:
    """Classifica uma PIL Image e retorna resultado estruturado."""
    cache = _ensure_model()
    model = cache["model"]
    class_names = cache["class_names"]
    transform = cache["transform"]
    device = cache["device"]

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]

    top3_probs, top3_indices = torch.topk(probs, min(3, len(class_names)))

    top3 = []
    for prob, idx in zip(top3_probs, top3_indices):
        top3.append({
            "class": class_names[idx.item()],
            "confidence": round(prob.item(), 4),
        })

    return {
        "disease": top3[0]["class"],
        "confidence": top3[0]["confidence"],
        "top3": top3,
        "model_accuracy": cache["accuracy"],
    }


def predict_image(image_path: str) -> dict:
    """Classifica uma imagem a partir de um caminho de arquivo."""
    image = Image.open(image_path).convert("RGB")
    return _classify(image)


def predict_image_bytes(image_bytes: bytes) -> dict:
    """Classifica uma imagem a partir de bytes (ex: base64 decodificado)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _classify(image)


def predict_image_base64(b64_string: str) -> dict:
    """Classifica uma imagem a partir de string base64."""
    # Remove header data:image/...;base64, se presente
    if "," in b64_string[:100]:
        b64_string = b64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_string)
    return predict_image_bytes(image_bytes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _download_image(url: str) -> str:
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

    if os.path.getsize(tmp_path) == 0:
        os.unlink(tmp_path)
        raise RuntimeError("Arquivo baixado está vazio")

    return tmp_path


def _cli_predict(image_path: str):
    """CLI: classifica e mostra resultado formatado."""
    cache = _ensure_model()
    print(f"Modelo carregado: {len(cache['class_names'])} classes, acurácia val: {cache['accuracy']*100:.1f}%")
    print(f"Classes: {cache['class_names']}\n")

    tmp_file = None
    if _is_url(image_path):
        tmp_file = _download_image(image_path)
        image_path = tmp_file

    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Imagem: {image.size[0]}x{image.size[1]}px")

        result = _classify(image)

        print(f"\n{'='*40}")
        print(f"  DIAGNÓSTICO")
        print(f"{'='*40}")
        for i, entry in enumerate(result["top3"]):
            p = entry["confidence"] * 100
            bar = "█" * int(p / 5) + "░" * (20 - int(p / 5))
            marker = " ◄" if i == 0 else ""
            print(f"  {i+1}. {entry['class']:<20} {p:>5.1f}%  {bar}{marker}")
        print(f"{'='*40}")

    finally:
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)


def _cli_server():
    """CLI: modo servidor — lê JSON do stdin, responde JSON no stdout."""
    cache = _ensure_model()
    print(json.dumps({
        "status": "ready",
        "classes": cache["class_names"],
        "accuracy": cache["accuracy"],
    }), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            b64 = req.get("image", "")
            result = predict_image_base64(b64)
            print(json.dumps(result), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--server":
        _cli_server()
    elif len(sys.argv) == 2:
        _cli_predict(sys.argv[1])
    else:
        print("Uso: python3 training/predict.py <caminho_ou_url>")
        print("      python3 training/predict.py --server  (modo JSON stdin/stdout)")
        sys.exit(1)
