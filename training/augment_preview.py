"""
augment_preview.py — Preview visual dos 4 perfis de augmentation.

Pega 1 imagem de cada classe do holdout, aplica 4 perfis, salva grid.
"""

import csv
import io
import os
import random
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "augment_preview_grid")
IMG_SIZE = 224
SEED = 42


def load_one_per_class():
    """Pick one holdout image per class (middle of list for representativeness)."""
    by_class = defaultdict(list)
    with open(os.path.join(SPLITS_DIR, "holdout.csv")) as f:
        for row in csv.DictReader(f):
            by_class[row["class"]].append(row["path"])

    selected = {}
    for cls in sorted(by_class.keys()):
        paths = by_class[cls]
        selected[cls] = paths[len(paths) // 2]
    return selected


def load_same_class_images(cls):
    """Load a few train images of the same class for CutMix source."""
    paths = []
    with open(os.path.join(SPLITS_DIR, "train.csv")) as f:
        for row in csv.DictReader(f):
            if row["class"] == cls:
                paths.append(row["path"])
            if len(paths) >= 20:
                break
    return paths


def load_and_resize(path):
    """Load image with PIL, resize to IMG_SIZE."""
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return img


# ─── Augmentation profiles ───────────────────────────────────────────────────


def aug_light(img):
    """LIGHT (current baseline): flip, rotate leve, brightness leve."""
    rng = random.Random()

    # Random horizontal flip
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotation ±10°
    angle = rng.uniform(-10, 10)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    # Brightness ±10%
    from PIL import ImageEnhance
    factor = rng.uniform(0.9, 1.1)
    img = ImageEnhance.Brightness(img).enhance(factor)

    return img


def aug_b1(img):
    """B1 (RandAugment moderado)."""
    from PIL import ImageEnhance
    rng = random.Random()

    # Horizontal flip
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Rotation ±15°
    angle = rng.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    # Brightness 0.8-1.2
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))

    # Contrast 0.8-1.2
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))

    # Blur leve (50% chance)
    if rng.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))

    # JPEG compression 70-90
    buf = io.BytesIO()
    quality = rng.randint(70, 90)
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    # Hue/Saturation shift via HSV
    arr = np.array(img).astype(np.float32)
    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-5, 5)) % 180  # hue ±0.03 (mapped to 0-180)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.85, 1.15), 0, 255)  # saturation
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    img = Image.fromarray(rgb)

    # Random erasing leve (10-15% da área)
    if rng.random() > 0.5:
        arr = np.array(img)
        h, w = arr.shape[:2]
        eh = rng.randint(int(h * 0.08), int(h * 0.15))
        ew = rng.randint(int(w * 0.08), int(w * 0.15))
        ey = rng.randint(0, h - eh)
        ex = rng.randint(0, w - ew)
        arr[ey:ey + eh, ex:ex + ew] = rng.randint(0, 255)
        img = Image.fromarray(arr)

    return img


def aug_b2(img):
    """B2 (RandAugment agressivo)."""
    from PIL import ImageEnhance
    rng = random.Random()

    # Horizontal + vertical flip
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Rotation ±30°
    angle = rng.uniform(-30, 30)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

    # Brightness 0.7-1.3
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.7, 1.3))

    # Contrast 0.7-1.3
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))

    # Blur moderado
    if rng.random() > 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))

    # JPEG compression 50-85
    buf = io.BytesIO()
    quality = rng.randint(50, 85)
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    # Hue/Saturation agressivo
    arr = np.array(img).astype(np.float32)
    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-9, 9)) % 180  # hue ±0.05
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.7, 1.3), 0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    img = Image.fromarray(rgb)

    # Random erasing moderado (15-25%)
    if rng.random() > 0.3:
        arr = np.array(img)
        h, w = arr.shape[:2]
        eh = rng.randint(int(h * 0.12), int(h * 0.25))
        ew = rng.randint(int(w * 0.12), int(w * 0.25))
        ey = rng.randint(0, h - eh)
        ex = rng.randint(0, w - ew)
        arr[ey:ey + eh, ex:ex + ew] = rng.randint(0, 255)
        img = Image.fromarray(arr)

    return img


def aug_b3_cutmix(img, cls):
    """B3 (CutMix): recorta pedaço de outra imagem da mesma classe e cola."""
    rng = random.Random()

    # Load a random same-class image
    donor_paths = load_same_class_images(cls)
    if not donor_paths:
        return img

    donor_path = rng.choice(donor_paths)
    try:
        donor = load_and_resize(donor_path)
    except Exception:
        return img

    arr_base = np.array(img)
    arr_donor = np.array(donor)
    h, w = arr_base.shape[:2]

    # CutMix: cut 25-40% of area from donor
    lam = rng.uniform(0.25, 0.40)
    cut_h = int(h * np.sqrt(lam))
    cut_w = int(w * np.sqrt(lam))

    cy = rng.randint(0, h)
    cx = rng.randint(0, w)

    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)

    arr_base[y1:y2, x1:x2] = arr_donor[y1:y2, x1:x2]

    return Image.fromarray(arr_base)


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  AUGMENTATION PREVIEW")
    print("=" * 60)

    selected = load_one_per_class()
    profiles = [
        ("LIGHT", aug_light),
        ("B1_moderate", aug_b1),
        ("B2_aggressive", aug_b2),
        ("B3_cutmix", None),  # special handling
    ]

    generated = []

    for cls, path in sorted(selected.items()):
        print(f"\n  [{cls}] {os.path.basename(path)}")

        try:
            original = load_and_resize(path)
        except Exception as e:
            print(f"    ERRO: {e}")
            continue

        # Save original
        orig_name = f"{cls}_00_ORIGINAL.jpg"
        orig_path = os.path.join(OUTPUT_DIR, orig_name)
        original.save(orig_path, quality=95)
        generated.append(orig_path)
        print(f"    {orig_name}")

        for prof_name, aug_fn in profiles:
            # Reset seed per profile for reproducibility but different results per profile
            random.seed(SEED + hash(prof_name) % 10000)

            if prof_name == "B3_cutmix":
                result = aug_b3_cutmix(original.copy(), cls)
            else:
                result = aug_fn(original.copy())

            fname = f"{cls}_{prof_name}.jpg"
            fpath = os.path.join(OUTPUT_DIR, fname)
            result.save(fpath, quality=95)
            generated.append(fpath)
            print(f"    {fname}")

    # Also generate a combined grid per class (original + 4 augments side by side)
    print(f"\n  Gerando grids combinados...")
    for cls, path in sorted(selected.items()):
        try:
            original = load_and_resize(path)
        except Exception:
            continue

        cols = 5  # original + 4 profiles
        pad = 4
        label_h = 20
        grid_w = cols * IMG_SIZE + (cols - 1) * pad
        grid_h = IMG_SIZE + label_h

        grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
        draw = ImageDraw.Draw(grid)

        images = [original]
        labels = ["ORIGINAL"]

        for prof_name, aug_fn in profiles:
            random.seed(SEED + hash(prof_name) % 10000)
            if prof_name == "B3_cutmix":
                images.append(aug_b3_cutmix(original.copy(), cls))
            else:
                images.append(aug_fn(original.copy()))
            labels.append(prof_name)

        for i, (img, label) in enumerate(zip(images, labels)):
            x = i * (IMG_SIZE + pad)
            grid.paste(img, (x, 0))
            draw.text((x + 4, IMG_SIZE + 2), label, fill=(200, 200, 200))

        grid_name = f"GRID_{cls}.jpg"
        grid_path = os.path.join(OUTPUT_DIR, grid_name)
        grid.save(grid_path, quality=95)
        generated.append(grid_path)
        print(f"    {grid_name}")

    print(f"\n{'=' * 60}")
    print(f"  {len(generated)} arquivos gerados em:")
    print(f"  {OUTPUT_DIR}/")
    print(f"{'=' * 60}")

    print(f"\nArquivos:")
    for p in sorted(generated):
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
