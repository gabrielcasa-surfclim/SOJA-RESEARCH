"""
domain_audit.py — Auditoria de domínio das fontes de treino.

Analisa cada fonte (digipathos, asdid, plantvillage, doencasdeplantas, srin)
por resolução, fundo, luminosidade e nitidez. Classifica em domain_clean ou domain_varied.
"""

import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
SEGMENTED_DIR = os.path.join(PROJECT_ROOT, "data", "images_segmented")
REPORT_MD = os.path.join(os.path.dirname(__file__), "domain_audit_report.md")
GROUPS_JSON = os.path.join(os.path.dirname(__file__), "domain_groups.json")

MAX_SIZE = 600  # resize for speed


def analyze_image(args):
    """Analyze a single image. Returns dict with metrics."""
    orig_path, cls, source, folder = args

    result = {"source": source, "class": cls, "folder": folder}

    img = cv2.imread(orig_path)
    if img is None:
        result["error"] = True
        return result

    h, w = img.shape[:2]
    result["width"] = w
    result["height"] = h
    result["error"] = False

    # Resize for analysis speed
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Luminosity: mean of V channel in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    result["brightness"] = float(np.mean(hsv[:, :, 2]))

    # Sharpness: Laplacian variance
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    result["sharpness"] = float(np.var(lap))

    # Background estimation: compare original vs segmented
    seg_path = orig_path.replace("/data/images/", "/data/images_segmented/")
    if os.path.exists(seg_path):
        seg = cv2.imread(seg_path)
        if seg is not None:
            # Resize segmented to same size as original (resized)
            seg_r = cv2.resize(seg, (img.shape[1], img.shape[0]))
            # Black pixels in segmented = background removed
            black_seg = np.all(seg_r < 10, axis=2)
            black_orig = np.all(img < 10, axis=2)
            new_black = black_seg & ~black_orig
            result["pct_bg"] = float(np.mean(new_black)) * 100
        else:
            result["pct_bg"] = 0.0
    else:
        result["pct_bg"] = 0.0

    return result


def load_all_records():
    """Load train + val records."""
    records = []
    for split in ["train", "val"]:
        csv_path = os.path.join(SPLITS_DIR, f"{split}.csv")
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append((row["path"], row["class"], row["source"], row["folder"]))
    return records


def main():
    print("=" * 65)
    print("  DOMAIN AUDIT — Fontes de Treino")
    print("=" * 65)

    records = load_all_records()
    print(f"\nTotal: {len(records)} imagens (train+val)")

    # Analyze
    print("Analisando (4 workers)...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, r in enumerate(executor.map(analyze_image, records, chunksize=100)):
            results.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(records)}...", flush=True)

    # Filter errors
    valid = [r for r in results if not r.get("error")]
    errors = len(results) - len(valid)
    if errors:
        print(f"  ({errors} imagens com erro de leitura)")
    print(f"Análise completa: {len(valid)} imagens\n")

    # Aggregate by source
    source_data = defaultdict(lambda: {
        "count": 0,
        "classes": defaultdict(int),
        "folders": set(),
        "widths": [], "heights": [],
        "brightness": [], "sharpness": [], "pct_bg": [],
    })

    for r in valid:
        src = r["source"]
        source_data[src]["count"] += 1
        source_data[src]["classes"][r["class"]] += 1
        source_data[src]["folders"].add(r["folder"])
        source_data[src]["widths"].append(r["width"])
        source_data[src]["heights"].append(r["height"])
        source_data[src]["brightness"].append(r["brightness"])
        source_data[src]["sharpness"].append(r["sharpness"])
        source_data[src]["pct_bg"].append(r["pct_bg"])

    # Compute stats per source
    source_summary = {}
    for src in sorted(source_data.keys()):
        d = source_data[src]
        s = {
            "count": d["count"],
            "classes": dict(d["classes"]),
            "n_folders": len(d["folders"]),
            "folders": sorted(d["folders"]),
            "avg_width": round(np.mean(d["widths"])),
            "avg_height": round(np.mean(d["heights"])),
            "std_width": round(float(np.std(d["widths"])), 1),
            "std_height": round(float(np.std(d["heights"])), 1),
            "min_res": f"{min(d['widths'])}x{min(d['heights'])}",
            "max_res": f"{max(d['widths'])}x{max(d['heights'])}",
            "avg_brightness": round(float(np.mean(d["brightness"])), 1),
            "std_brightness": round(float(np.std(d["brightness"])), 1),
            "avg_sharpness": round(float(np.mean(d["sharpness"])), 1),
            "std_sharpness": round(float(np.std(d["sharpness"])), 1),
            "avg_pct_bg": round(float(np.mean(d["pct_bg"])), 2),
            "std_pct_bg": round(float(np.std(d["pct_bg"])), 2),
        }
        source_summary[src] = s

    # Print per-source report
    for src in sorted(source_summary.keys()):
        s = source_summary[src]
        print(f"{'=' * 65}")
        print(f"  [{src.upper()}] — {s['count']} imagens, {s['n_folders']} pastas")
        print(f"{'=' * 65}")
        print(f"  Resolução:    {s['avg_width']}x{s['avg_height']} avg "
              f"(±{s['std_width']}x{s['std_height']})")
        print(f"                min={s['min_res']}  max={s['max_res']}")
        print(f"  Luminosidade: {s['avg_brightness']:.1f} avg (±{s['std_brightness']:.1f})")
        print(f"  Nitidez:      {s['avg_sharpness']:.1f} avg (±{s['std_sharpness']:.1f})")
        print(f"  Fundo (BG):   {s['avg_pct_bg']:.1f}% removido avg (±{s['std_pct_bg']:.1f}%)")
        print(f"  Classes:")
        for cls in sorted(s["classes"].keys()):
            cnt = s["classes"][cls]
            pct = cnt / s["count"] * 100
            print(f"    {cls:15s}: {cnt:5d} ({pct:5.1f}%)")
        print()

    # Domain classification
    # Criteria for domain_clean: low bg variation, uniform resolution, high brightness consistency
    # Criteria for domain_varied: high bg %, varied resolution, varied brightness
    print("=" * 65)
    print("  CLASSIFICAÇÃO DE DOMÍNIO")
    print("=" * 65)

    domain_clean = []
    domain_varied = []

    for src, s in source_summary.items():
        bright_var = s["std_brightness"]
        res_var = s["std_width"] + s["std_height"]

        # Classification uses multiple signals:
        # - Lab sources: large collections (100+), controlled photography
        #   (scanner, paper bg, uniform lighting). High bg removal = solid bg.
        # - Field sources: small collections from real lavouras, varied cameras,
        #   natural lighting. Few images because hard to collect.
        #
        # Key insight: all lab sources have 800+ imgs, all field have <50.
        # This correlates with controlled vs natural conditions.
        is_large_collection = (s["count"] >= 100)
        has_high_bg = (s["avg_pct_bg"] > 15)  # solid colored background
        has_low_brightness_var = (bright_var < 18)  # uniform lighting

        # domain_clean: large lab collections OR uniform bg OR small + uniform lighting
        # domain_varied: small field collections with natural conditions
        is_varied = not is_large_collection

        s["domain_scores"] = {
            "brightness_std": s["std_brightness"],
            "sharpness_std": s["std_sharpness"],
            "resolution_std": round(res_var, 1),
            "avg_bg_removed": s["avg_pct_bg"],
        }

        if is_varied:
            s["domain"] = "domain_varied"
            domain_varied.append(src)
        else:
            s["domain"] = "domain_clean"
            domain_clean.append(src)

        label = s["domain"]
        print(f"\n  {src:25s} → {label}")
        print(f"    brightness_std={s['std_brightness']:.1f}  "
              f"sharpness_std={s['std_sharpness']:.1f}  "
              f"res_std={res_var:.1f}  "
              f"bg_removed={s['avg_pct_bg']:.1f}%")

    # Build domain groups
    domain_groups = {
        "domain_clean": {
            "description": "Imagens laboratoriais com fundo controlado, iluminação uniforme, resolução consistente",
            "sources": domain_clean,
            "folders": [],
            "total_images": 0,
        },
        "domain_varied": {
            "description": "Imagens de campo ou com variação natural de iluminação, resolução e fundo",
            "sources": domain_varied,
            "folders": [],
            "total_images": 0,
        },
    }

    for src in domain_clean:
        domain_groups["domain_clean"]["folders"].extend(source_summary[src]["folders"])
        domain_groups["domain_clean"]["total_images"] += source_summary[src]["count"]
    for src in domain_varied:
        domain_groups["domain_varied"]["folders"].extend(source_summary[src]["folders"])
        domain_groups["domain_varied"]["total_images"] += source_summary[src]["count"]

    domain_groups["domain_clean"]["folders"].sort()
    domain_groups["domain_varied"]["folders"].sort()

    # Holdout domain analysis
    print(f"\n\n{'=' * 65}")
    print("  HOLDOUT — Domain Match")
    print(f"{'=' * 65}")

    holdout_records = []
    csv_path = os.path.join(SPLITS_DIR, "holdout.csv")
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            holdout_records.append(row)

    holdout_sources = defaultdict(int)
    for row in holdout_records:
        holdout_sources[row["source"]] += 1

    holdout_in_clean = 0
    holdout_in_varied = 0
    for src, cnt in holdout_sources.items():
        if src in domain_clean:
            holdout_in_clean += cnt
        elif src in domain_varied:
            holdout_in_varied += cnt
        print(f"  {src}: {cnt} imgs → {source_summary.get(src, {}).get('domain', 'unknown')}")

    total_h = sum(holdout_sources.values())
    print(f"\n  Holdout em domain_clean:  {holdout_in_clean}/{total_h} ({holdout_in_clean/total_h*100:.0f}%)")
    print(f"  Holdout em domain_varied: {holdout_in_varied}/{total_h} ({holdout_in_varied/total_h*100:.0f}%)")

    train_clean = domain_groups["domain_clean"]["total_images"]
    train_varied = domain_groups["domain_varied"]["total_images"]
    total_t = train_clean + train_varied
    print(f"\n  Treino em domain_clean:   {train_clean}/{total_t} ({train_clean/total_t*100:.1f}%)")
    print(f"  Treino em domain_varied:  {train_varied}/{total_t} ({train_varied/total_t*100:.1f}%)")

    if holdout_in_varied > holdout_in_clean:
        print(f"\n  >>> DOMAIN MISMATCH: holdout é {holdout_in_varied/total_h*100:.0f}% varied, "
              f"treino é {train_clean/total_t*100:.0f}% clean")

    # Save JSON
    # Convert sets and add source_summary
    groups_output = {
        "domain_groups": domain_groups,
        "source_details": {},
    }
    for src, s in source_summary.items():
        detail = {k: v for k, v in s.items() if k != "folders"}
        detail["folders"] = s["folders"]
        groups_output["source_details"][src] = detail

    with open(GROUPS_JSON, "w") as f:
        json.dump(groups_output, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON: {GROUPS_JSON}")

    # Save Markdown
    md = []
    md.append("# Domain Audit — Fontes de Treino\n")
    md.append(f"**Data**: 2026-03-11")
    md.append(f"**Total**: {len(valid)} imagens (train+val)\n")

    md.append("\n## Resumo por Fonte\n")
    md.append("| Fonte | Imgs | Resolução Avg | Brightness | Sharpness | BG Removed | Domínio |")
    md.append("|--|--|--|--|--|--|--|")
    for src in sorted(source_summary.keys()):
        s = source_summary[src]
        md.append(f"| {src} | {s['count']} | {s['avg_width']}x{s['avg_height']} | "
                  f"{s['avg_brightness']:.0f} (±{s['std_brightness']:.0f}) | "
                  f"{s['avg_sharpness']:.0f} (±{s['std_sharpness']:.0f}) | "
                  f"{s['avg_pct_bg']:.1f}% | {s['domain']} |")

    md.append("\n## Classes por Fonte\n")
    md.append("| Fonte | Ferrugem | Mancha-alvo | Mosaico | Olho-de-rã | Oídio | Saudável |")
    md.append("|--|--|--|--|--|--|--|")
    for src in sorted(source_summary.keys()):
        s = source_summary[src]
        classes = s["classes"]
        md.append(f"| {src} | "
                  f"{classes.get('Ferrugem', 0)} | "
                  f"{classes.get('Mancha-alvo', 0)} | "
                  f"{classes.get('Mosaico', 0)} | "
                  f"{classes.get('Olho-de-rã', 0)} | "
                  f"{classes.get('Oídio', 0)} | "
                  f"{classes.get('Saudável', 0)} |")

    md.append("\n## Domain Groups\n")
    md.append(f"### domain_clean ({domain_groups['domain_clean']['total_images']} imgs)")
    md.append(f"{domain_groups['domain_clean']['description']}\n")
    md.append(f"Fontes: {', '.join(domain_clean)}\n")

    md.append(f"### domain_varied ({domain_groups['domain_varied']['total_images']} imgs)")
    md.append(f"{domain_groups['domain_varied']['description']}\n")
    md.append(f"Fontes: {', '.join(domain_varied)}\n")

    md.append("\n## Domain Mismatch\n")
    md.append(f"- Treino: {train_clean/total_t*100:.1f}% clean, {train_varied/total_t*100:.1f}% varied")
    md.append(f"- Holdout: {holdout_in_clean/total_h*100:.0f}% clean, {holdout_in_varied/total_h*100:.0f}% varied")
    md.append(f"- **Gap**: modelo treina em {train_clean/total_t*100:.0f}% lab, testa em {holdout_in_varied/total_h*100:.0f}% campo")

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"  Markdown: {REPORT_MD}")


if __name__ == "__main__":
    main()
