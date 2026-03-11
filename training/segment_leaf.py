"""
segment_leaf.py — Segmentação clássica de folhas por cor (HSV).

Pipeline:
  1. Converte BGR → HSV
  2. Máscara do verde (folha) com range largo
  3. Morphological close + open pra limpar ruído
  4. Encontra maior contorno (folha principal)
  5. Bounding box → crop com margem
  6. Aplica máscara: fundo preto, folha preservada

Processa data/images/ → data/images_segmented/ mantendo estrutura de pastas.

Uso:
    python3 training/segment_leaf.py                    # processa tudo
    python3 training/segment_leaf.py --test              # testa 5 imagens
    python3 training/segment_leaf.py --folder NAME       # processa 1 pasta
"""

import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images_segmented")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Tamanho máximo do lado maior pra processamento (reduz antes de segmentar)
MAX_PROCESSING_SIZE = 1024


# ─────────────────────────────────────────────────────────────────────
# Segmentação
# ─────────────────────────────────────────────────────────────────────

def segment_leaf(img_bgr):
    """Segmenta folha de soja a partir de imagem BGR.

    Returns:
        (cropped_bgr, mask, stats) ou (None, None, stats) se falhou.
        stats: dict com métricas da segmentação.
    """
    h_orig, w_orig = img_bgr.shape[:2]
    stats = {"original_size": (w_orig, h_orig)}

    # Redimensiona pra processamento se muito grande
    scale = 1.0
    if max(h_orig, w_orig) > MAX_PROCESSING_SIZE:
        scale = MAX_PROCESSING_SIZE / max(h_orig, w_orig)
        img_work = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_work = img_bgr.copy()

    h, w = img_work.shape[:2]

    # BGR → HSV
    hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)

    # Máscara verde ampla: captura folhas verdes, amareladas e marrons (doentes)
    # H: 15-95 (verde amplo + amarelo-verde + marrom-verde)
    # S: 20+ (exclui cinza/branco do fundo)
    # V: 20+ (exclui preto profundo)
    mask_green = cv2.inRange(hsv, (15, 20, 20), (95, 255, 255))

    # Máscara amarelo-marrom: folhas doentes que perdem verde
    # H: 5-25 (amarelo-laranja-marrom), S: 30+, V: 40+
    mask_yellow = cv2.inRange(hsv, (5, 30, 40), (25, 255, 255))

    # Máscara marrom escuro: ferrugem, necrose
    # H: 0-15, S: 30+, V: 20-180
    mask_brown = cv2.inRange(hsv, (0, 30, 20), (15, 255, 180))

    # Combina todas as máscaras
    mask = mask_green | mask_yellow | mask_brown

    # Morphological ops pra limpar
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Close: preenche buracos dentro da folha
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    # Open: remove ruído pequeno do fundo
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Encontra contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        stats["status"] = "no_contours"
        return None, None, stats

    # Maior contorno = folha principal
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    total_area = h * w
    area_ratio = area / total_area

    stats["leaf_area_ratio"] = round(area_ratio, 3)

    # Se a folha ocupa menos de 3% da imagem, provavelmente falhou
    if area_ratio < 0.03:
        stats["status"] = "too_small"
        return None, None, stats

    # Cria máscara limpa só com o maior contorno
    clean_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # Convex hull pra preencher concavidades da folha
    hull = cv2.convexHull(largest)
    hull_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

    # Usa intersecção entre hull e máscara de cor (evita incluir fundo)
    # Mas expande um pouco a máscara original pra não cortar bordas
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    expanded_mask = cv2.dilate(clean_mask, dilate_kernel, iterations=2)
    final_mask = expanded_mask

    # Bounding box com margem
    x, y, bw, bh = cv2.boundingRect(largest)
    margin = int(max(bw, bh) * 0.05)  # 5% de margem
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)

    # Escala de volta pra tamanho original se redimensionou
    if scale != 1.0:
        # Redimensiona máscara pro tamanho original
        final_mask = cv2.resize(final_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        # Recalcula bbox no tamanho original
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_orig, x2)
        y2 = min(h_orig, y2)

    # Aplica máscara: fundo preto
    result = img_bgr.copy()
    result[final_mask == 0] = 0

    # Crop
    cropped = result[y1:y2, x1:x2]
    crop_mask = final_mask[y1:y2, x1:x2]

    stats["status"] = "ok"
    stats["crop_size"] = (x2 - x1, y2 - y1)
    stats["mask_coverage"] = round(np.sum(crop_mask > 0) / (crop_mask.shape[0] * crop_mask.shape[1]), 3)

    return cropped, crop_mask, stats


# ─────────────────────────────────────────────────────────────────────
# Processamento em lote
# ─────────────────────────────────────────────────────────────────────

def process_folder(folder_name, verbose=False):
    """Processa todas as imagens de uma pasta."""
    src_dir = os.path.join(DATA_DIR, folder_name)
    dst_dir = os.path.join(OUT_DIR, folder_name)

    if not os.path.isdir(src_dir):
        print(f"  Pasta não encontrada: {src_dir}")
        return {"ok": 0, "fail": 0, "skip": 0}

    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(src_dir)
                   if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS)

    counts = {"ok": 0, "fail": 0, "skip": 0}

    for fname in files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        # Skip se já existe
        if os.path.exists(dst_path):
            counts["skip"] += 1
            continue

        img = cv2.imread(src_path)
        if img is None:
            counts["fail"] += 1
            if verbose:
                print(f"    Erro ao ler: {fname}")
            continue

        cropped, mask, stats = segment_leaf(img)

        if cropped is not None and cropped.size > 0:
            cv2.imwrite(dst_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts["ok"] += 1
            if verbose:
                print(f"    {fname}: {stats['original_size']} → {stats['crop_size']}  "
                      f"area={stats['leaf_area_ratio']:.1%}  cover={stats['mask_coverage']:.1%}")
        else:
            # Fallback: salva original (não conseguiu segmentar)
            cv2.imwrite(dst_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            counts["fail"] += 1
            if verbose:
                print(f"    {fname}: FALLBACK ({stats.get('status', '?')}) — salvo original")

    return counts


def process_all():
    """Processa todas as pastas."""
    folders = sorted(f for f in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, f)))

    print(f"\n{'='*70}")
    print(f"  SEGMENTAÇÃO DE FOLHAS — {len(folders)} pastas")
    print(f"  {DATA_DIR} → {OUT_DIR}")
    print(f"{'='*70}\n")

    total = {"ok": 0, "fail": 0, "skip": 0}
    start = time.time()

    for i, folder in enumerate(folders):
        src_dir = os.path.join(DATA_DIR, folder)
        n_files = len([f for f in os.listdir(src_dir)
                       if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS])

        counts = process_folder(folder)
        total["ok"] += counts["ok"]
        total["fail"] += counts["fail"]
        total["skip"] += counts["skip"]

        status = f"ok={counts['ok']} fail={counts['fail']}"
        if counts["skip"] > 0:
            status += f" skip={counts['skip']}"
        elapsed = time.time() - start
        print(f"  [{i+1:>2}/{len(folders)}] {folder:<45} {n_files:>5} imgs  {status}  ({elapsed:.0f}s)")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"  RESULTADO")
    print(f"{'='*70}")
    print(f"  Segmentadas: {total['ok']}")
    print(f"  Fallback:    {total['fail']} (salvas como original)")
    print(f"  Skipped:     {total['skip']} (já existiam)")
    print(f"  Tempo:       {elapsed:.1f}s")
    print(f"  Output:      {OUT_DIR}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────
# Teste com 5 imagens
# ─────────────────────────────────────────────────────────────────────

def test_samples():
    """Testa segmentação com 5 imagens de classes diferentes."""
    samples = [
        ("digipathos_ferrugem", "Ferrugem (digipathos)"),
        ("soybean_rust", "Ferrugem (asdid)"),
        ("doencasdeplantas_oidio", "Oídio (campo)"),
        ("digipathos_mosaico", "Mosaico (digipathos)"),
        ("frogeye", "Olho-de-rã (asdid)"),
    ]

    print(f"\n{'='*70}")
    print(f"  TESTE DE SEGMENTAÇÃO — 5 amostras")
    print(f"{'='*70}\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    test_dir = os.path.join(OUT_DIR, "_test")
    os.makedirs(test_dir, exist_ok=True)

    for folder, label in samples:
        src_dir = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(src_dir):
            print(f"  {label:<30} pasta não encontrada: {folder}")
            continue

        files = sorted(f for f in os.listdir(src_dir)
                       if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS)
        if not files:
            print(f"  {label:<30} sem imagens")
            continue

        # Pega a primeira imagem
        fname = files[0]
        src_path = os.path.join(src_dir, fname)

        img = cv2.imread(src_path)
        if img is None:
            print(f"  {label:<30} erro ao ler: {fname}")
            continue

        cropped, mask, stats = segment_leaf(img)

        # Salva original e segmentada lado a lado pra comparação
        out_name = f"{folder}__{fname}"
        orig_path = os.path.join(test_dir, f"orig__{out_name}")
        seg_path = os.path.join(test_dir, f"seg__{out_name}")

        # Redimensiona original pra comparação visual
        h, w = img.shape[:2]
        preview_h = 400
        preview_w = int(w * preview_h / h)
        orig_preview = cv2.resize(img, (preview_w, preview_h))
        cv2.imwrite(orig_path, orig_preview, [cv2.IMWRITE_JPEG_QUALITY, 90])

        if cropped is not None and cropped.size > 0:
            ch, cw = cropped.shape[:2]
            seg_preview_h = 400
            seg_preview_w = int(cw * seg_preview_h / ch)
            seg_preview = cv2.resize(cropped, (seg_preview_w, seg_preview_h))
            cv2.imwrite(seg_path, seg_preview, [cv2.IMWRITE_JPEG_QUALITY, 90])

            print(f"  {label:<30} {stats['original_size'][0]:>4}x{stats['original_size'][1]:<4} → "
                  f"{stats['crop_size'][0]:>4}x{stats['crop_size'][1]:<4}  "
                  f"area={stats['leaf_area_ratio']:>5.1%}  "
                  f"cover={stats['mask_coverage']:>5.1%}  "
                  f"status={stats['status']}")
        else:
            cv2.imwrite(seg_path, orig_preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  {label:<30} FALHOU: {stats.get('status', '?')} — "
                  f"area={stats.get('leaf_area_ratio', 0):.1%}")

    print(f"\n  Arquivos de teste salvos em: {test_dir}/")
    print(f"  Abra orig__* e seg__* pra comparar visualmente.\n")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if "--test" in args:
        test_samples()
    elif "--folder" in args:
        idx = args.index("--folder")
        if idx + 1 < len(args):
            folder = args[idx + 1]
            print(f"\nProcessando pasta: {folder}")
            counts = process_folder(folder, verbose=True)
            print(f"\nResultado: ok={counts['ok']} fail={counts['fail']} skip={counts['skip']}")
        else:
            print("Uso: python3 segment_leaf.py --folder NOME_DA_PASTA")
    else:
        process_all()


if __name__ == "__main__":
    main()
