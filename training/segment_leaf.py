"""
segment_leaf.py — Segmentação de folhas com preservação de tecido doente.

Pipeline v2 (corrige problema da v1 que removia manchas doentes):
  1. HSV color detection para encontrar a REGIÃO da folha
  2. Morphological close agressivo pra fechar buracos
  3. Maior contorno → convex hull → PREENCHE TUDO dentro
  4. GrabCut refinamento (opcional, melhora bordas)
  5. Crop com margem + fundo preto

Diferença da v1: a máscara final é o contorno PREENCHIDO, não a
máscara de cor. Isso preserva manchas marrons, necrose, ferrugem etc.

Processa data/images/ → data/images_segmented/ mantendo estrutura de pastas.

Uso:
    python3 training/segment_leaf.py --test             # testa 5 imagens
    python3 training/segment_leaf.py                    # processa tudo
    python3 training/segment_leaf.py --folder NAME      # processa 1 pasta
    python3 training/segment_leaf.py --force             # reprocessa tudo
"""

import gc
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "images_segmented")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

MAX_PROCESSING_SIZE = 800  # menor pra ser mais rápido
NUM_WORKERS = 2
GRABCUT_ITERS = 3  # iterações do GrabCut (0 = desativa)


# ─────────────────────────────────────────────────────────────────────
# Segmentação v2
# ─────────────────────────────────────────────────────────────────────

def segment_leaf(img_bgr, use_grabcut=True):
    """Segmenta folha preservando TODO o tecido (inclusive doente).

    Pipeline:
      1. HSV para encontrar pixels de folha (verde + amarelo + marrom)
      2. Morphological close grande para unir regiões
      3. Maior contorno → convex hull
      4. Hull preenchido = máscara final (preserva manchas doentes)
      5. GrabCut opcionalmente refina bordas

    Returns:
        (cropped_bgr, mask, stats) ou (None, None, stats) se falhou.
    """
    h_orig, w_orig = img_bgr.shape[:2]
    stats = {"original_size": (w_orig, h_orig)}

    # Redimensiona pra processamento
    scale = 1.0
    if max(h_orig, w_orig) > MAX_PROCESSING_SIZE:
        scale = MAX_PROCESSING_SIZE / max(h_orig, w_orig)
        img_work = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        img_work = img_bgr.copy()

    h, w = img_work.shape[:2]

    # --- Etapa 1: detectar região da folha via cor ---
    hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)

    # Range amplo: qualquer coisa com saturação que não seja branco/cinza/preto
    # Folhas verdes: H 20-90, S 25+
    mask_green = cv2.inRange(hsv, (20, 25, 25), (90, 255, 255))
    # Amarelo-verde (doença inicial): H 10-25
    mask_yellow = cv2.inRange(hsv, (10, 30, 40), (25, 255, 255))
    # Marrom (ferrugem, necrose): H 0-15, S 30+
    mask_brown = cv2.inRange(hsv, (0, 30, 25), (15, 255, 200))

    color_mask = mask_green | mask_yellow | mask_brown

    # --- Etapa 2: morphological ops agressivos ---
    # Close grande: conecta regiões da mesma folha separadas por manchas
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, k_close, iterations=3)
    # Open: remove ruído de fundo
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, k_open, iterations=1)

    # --- Etapa 3: contorno → hull preenchido ---
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        stats["status"] = "no_contours"
        return None, None, stats

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    area_ratio = area / (h * w)
    stats["leaf_area_ratio"] = round(area_ratio, 3)

    if area_ratio < 0.03:
        stats["status"] = "too_small"
        return None, None, stats

    # Convex hull preenchido = máscara que cobre TODA a folha
    hull = cv2.convexHull(largest)
    hull_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

    # --- Etapa 4: GrabCut para refinar bordas (opcional) ---
    final_mask = hull_mask

    if use_grabcut and GRABCUT_ITERS > 0 and area_ratio < 0.85:
        try:
            # Inicializa GrabCut com a hull mask
            gc_mask = np.where(hull_mask > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)

            # Pixels da color_mask original são definitely foreground
            gc_mask[color_mask > 0] = cv2.GC_FGD

            # Bordas externas são definitely background
            border = 5
            gc_mask[:border, :] = cv2.GC_BGD
            gc_mask[-border:, :] = cv2.GC_BGD
            gc_mask[:, :border] = cv2.GC_BGD
            gc_mask[:, -border:] = cv2.GC_BGD

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            cv2.grabCut(img_work, gc_mask, None, bgd_model, fgd_model,
                        GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)

            final_mask = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
            ).astype(np.uint8)

            # Se GrabCut removeu demais (< 50% da hull), volta pro hull
            gc_area = np.sum(final_mask > 0)
            hull_area = np.sum(hull_mask > 0)
            if gc_area < hull_area * 0.5:
                final_mask = hull_mask
                stats["grabcut"] = "reverted"
            else:
                stats["grabcut"] = "ok"
        except Exception:
            final_mask = hull_mask
            stats["grabcut"] = "failed"
    else:
        stats["grabcut"] = "skipped"

    # Dilata levemente pra não cortar bordas da folha
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.dilate(final_mask, k_dilate, iterations=1)

    # --- Etapa 5: bounding box + crop ---
    x, y, bw, bh = cv2.boundingRect(
        cv2.findNonZero(final_mask) if np.any(final_mask) else np.array([[0, 0]])
    )
    margin = int(max(bw, bh) * 0.03)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)

    # Escala de volta
    if scale != 1.0:
        final_mask = cv2.resize(final_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        x1 = max(0, int(x1 / scale))
        y1 = max(0, int(y1 / scale))
        x2 = min(w_orig, int(x2 / scale))
        y2 = min(h_orig, int(y2 / scale))

    # Aplica máscara + crop
    result = img_bgr.copy()
    result[final_mask == 0] = 0
    cropped = result[y1:y2, x1:x2]
    crop_mask = final_mask[y1:y2, x1:x2]

    stats["status"] = "ok"
    stats["crop_size"] = (x2 - x1, y2 - y1)
    stats["mask_coverage"] = round(
        np.sum(crop_mask > 0) / max(crop_mask.shape[0] * crop_mask.shape[1], 1), 3
    )

    return cropped, crop_mask, stats


# ─────────────────────────────────────────────────────────────────────
# Worker para multiprocessing
# ─────────────────────────────────────────────────────────────────────

def _process_one(args):
    """Processa uma única imagem (chamado via ProcessPoolExecutor)."""
    src_path, dst_path = args
    try:
        img = cv2.imread(src_path)
        if img is None:
            return "fail", None

        cropped, _, stats = segment_leaf(img)

        if cropped is not None and cropped.size > 0:
            cv2.imwrite(dst_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return "ok", stats
        else:
            # Fallback: salva original
            cv2.imwrite(dst_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return "fail", stats
    except Exception as e:
        try:
            img = cv2.imread(src_path)
            if img is not None:
                cv2.imwrite(dst_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception:
            pass
        return "fail", {"status": f"error: {e}"}


# ─────────────────────────────────────────────────────────────────────
# Processamento em lote
# ─────────────────────────────────────────────────────────────────────

def process_folder(folder_name, verbose=False, force=False):
    """Processa todas as imagens de uma pasta."""
    src_dir = os.path.join(DATA_DIR, folder_name)
    dst_dir = os.path.join(OUT_DIR, folder_name)

    if not os.path.isdir(src_dir):
        print(f"  Pasta não encontrada: {src_dir}")
        return {"ok": 0, "fail": 0, "skip": 0}

    os.makedirs(dst_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(src_dir)
                   if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS)

    # Monta lista de jobs
    jobs = []
    skip_count = 0
    for fname in files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        if not force and os.path.exists(dst_path):
            skip_count += 1
            continue
        jobs.append((src_path, dst_path))

    counts = {"ok": 0, "fail": 0, "skip": skip_count}

    if not jobs:
        return counts

    # Processa com pool de workers
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for status, stats in pool.map(_process_one, jobs):
            counts[status] = counts.get(status, 0) + 1
            if verbose and stats:
                print(f"    {stats.get('status', '?')}  "
                      f"area={stats.get('leaf_area_ratio', 0):.1%}  "
                      f"gc={stats.get('grabcut', '?')}")

    return counts


def process_all(force=False):
    """Processa todas as pastas."""
    folders = sorted(f for f in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, f)))

    print(f"\n{'='*70}")
    print(f"  SEGMENTAÇÃO v2 — {len(folders)} pastas, {NUM_WORKERS} workers")
    print(f"  {DATA_DIR} → {OUT_DIR}")
    print(f"  GrabCut: {'on' if GRABCUT_ITERS > 0 else 'off'} ({GRABCUT_ITERS} iter)")
    if force:
        print(f"  FORCE: reprocessando tudo")
    print(f"{'='*70}\n")

    total = {"ok": 0, "fail": 0, "skip": 0}
    start = time.time()

    for i, folder in enumerate(folders):
        src_dir = os.path.join(DATA_DIR, folder)
        n_files = len([f for f in os.listdir(src_dir)
                       if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS])

        counts = process_folder(folder, force=force)
        total["ok"] += counts["ok"]
        total["fail"] += counts["fail"]
        total["skip"] += counts["skip"]

        status = f"ok={counts['ok']} fail={counts['fail']}"
        if counts["skip"] > 0:
            status += f" skip={counts['skip']}"
        elapsed = time.time() - start
        print(f"  [{i+1:>2}/{len(folders)}] {folder:<45} {n_files:>5} imgs  {status}  ({elapsed:.0f}s)")

        # Libera memória periodicamente
        if (i + 1) % 10 == 0:
            gc.collect()

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"  RESULTADO")
    print(f"{'='*70}")
    print(f"  Segmentadas: {total['ok']}")
    print(f"  Fallback:    {total['fail']} (salvas como original)")
    print(f"  Skipped:     {total['skip']} (já existiam)")
    print(f"  Tempo:       {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output:      {OUT_DIR}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────
# Teste com 5 imagens
# ─────────────────────────────────────────────────────────────────────

def test_samples():
    """Testa segmentação com 5 imagens de classes diferentes."""
    samples = [
        ("digipathos_ferrugem", "Ferrugem (lab)"),
        ("doencasdeplantas_ferrugem_asiatica", "Ferrugem (campo)"),
        ("digipathos_oidio", "Oídio (lab)"),
        ("digipathos_mosaico", "Mosaico (lab)"),
        ("frogeye", "Olho-de-rã (asdid)"),
    ]

    print(f"\n{'='*70}")
    print(f"  TESTE DE SEGMENTAÇÃO v2 — 5 amostras")
    print(f"  Hull preenchido + GrabCut refinamento")
    print(f"{'='*70}\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    test_dir = os.path.join(OUT_DIR, "_test_v2")
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

        # Pega a 3a imagem (evita corner cases da 1a)
        fname = files[min(2, len(files) - 1)]
        src_path = os.path.join(src_dir, fname)

        img = cv2.imread(src_path)
        if img is None:
            print(f"  {label:<30} erro ao ler: {fname}")
            continue

        cropped, mask, stats = segment_leaf(img)

        out_name = f"{folder}__{fname}"
        orig_path = os.path.join(test_dir, f"orig__{out_name}")
        seg_path = os.path.join(test_dir, f"seg__{out_name}")

        # Salva original
        oh, ow = img.shape[:2]
        preview_h = 400
        preview_w = int(ow * preview_h / oh)
        cv2.imwrite(orig_path, cv2.resize(img, (preview_w, preview_h)),
                     [cv2.IMWRITE_JPEG_QUALITY, 90])

        if cropped is not None and cropped.size > 0:
            ch, cw = cropped.shape[:2]
            seg_h = 400
            seg_w = int(cw * seg_h / ch)
            cv2.imwrite(seg_path, cv2.resize(cropped, (seg_w, seg_h)),
                         [cv2.IMWRITE_JPEG_QUALITY, 90])

            print(f"  {label:<30} {stats['original_size'][0]:>4}x{stats['original_size'][1]:<4} → "
                  f"{stats['crop_size'][0]:>4}x{stats['crop_size'][1]:<4}  "
                  f"area={stats['leaf_area_ratio']:>5.1%}  "
                  f"cover={stats['mask_coverage']:>5.1%}  "
                  f"gc={stats.get('grabcut', '?')}")
        else:
            cv2.imwrite(seg_path,
                         cv2.resize(img, (preview_w, preview_h)),
                         [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  {label:<30} FALHOU: {stats.get('status', '?')} "
                  f"area={stats.get('leaf_area_ratio', 0):.1%}")

    print(f"\n  Arquivos salvos em: {test_dir}/")
    print(f"  Compare orig__* vs seg__* visualmente.\n")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    force = "--force" in args

    if "--test" in args:
        test_samples()
    elif "--folder" in args:
        idx = args.index("--folder")
        if idx + 1 < len(args):
            folder = args[idx + 1]
            print(f"\nProcessando pasta: {folder}")
            counts = process_folder(folder, verbose=True, force=force)
            print(f"\nResultado: ok={counts['ok']} fail={counts['fail']} skip={counts['skip']}")
        else:
            print("Uso: python3 segment_leaf.py --folder NOME_DA_PASTA")
    else:
        process_all(force=force)


if __name__ == "__main__":
    main()
