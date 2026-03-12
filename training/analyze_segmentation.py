"""
analyze_segmentation.py — Análise quantitativa da segmentação v2.

Compara data/images/ vs data/images_segmented/ sem treinar modelo.
Gera segmentation_report.md e segmentation_report.json.
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images")
SEGMENTED_DIR = os.path.join(PROJECT_ROOT, "data", "images_segmented")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
REPORT_MD = os.path.join(os.path.dirname(__file__), "segmentation_report.md")
REPORT_JSON = os.path.join(os.path.dirname(__file__), "segmentation_report.json")


def load_split(split_name):
    """Load split CSV, return list of (path, class, source, folder)."""
    csv_path = os.path.join(SPLITS_DIR, f"{split_name}.csv")
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append((row["path"], row["class"], row["source"], row["folder"]))
    return records


def analyze_single_image(args):
    """Compare original vs segmented for one image. Returns metrics dict."""
    orig_path, seg_path, cls, source, folder = args

    result = {
        "class": cls,
        "source": source,
        "folder": folder,
        "orig_exists": os.path.exists(orig_path),
        "seg_exists": os.path.exists(seg_path),
    }

    if not result["orig_exists"] or not result["seg_exists"]:
        result["category"] = "missing"
        return result

    orig = cv2.imread(orig_path)
    seg = cv2.imread(seg_path)

    if orig is None or seg is None:
        result["category"] = "unreadable"
        return result

    # Tamanhos
    oh, ow = orig.shape[:2]
    sh, sw = seg.shape[:2]
    result["orig_size"] = (ow, oh)
    result["seg_size"] = (sw, sh)
    result["size_changed"] = (ow != sw) or (oh != sh)

    # Para comparar pixels, redimensionar ao mesmo tamanho
    if result["size_changed"]:
        seg_resized = cv2.resize(seg, (ow, oh))
    else:
        seg_resized = seg

    # Diferença absoluta média (0-255 por canal)
    diff = cv2.absdiff(orig, seg_resized).astype(np.float32)
    mean_diff = float(np.mean(diff))
    max_diff = float(np.max(diff))
    result["mean_pixel_diff"] = round(mean_diff, 2)
    result["max_pixel_diff"] = round(max_diff, 2)

    # Porcentagem de pixels alterados (threshold=5 para ignorar compressão JPEG)
    changed_mask = np.any(diff > 5, axis=2)
    pct_changed = float(np.mean(changed_mask)) * 100
    result["pct_pixels_changed"] = round(pct_changed, 2)

    # Detectar pixels pretos na segmentada (fundo removido)
    # Pixel é "preto" se todos os canais < 10
    black_mask_seg = np.all(seg_resized < 10, axis=2)
    black_mask_orig = np.all(orig < 10, axis=2)
    new_black = black_mask_seg & ~black_mask_orig
    pct_bg_removed = float(np.mean(new_black)) * 100
    result["pct_bg_removed"] = round(pct_bg_removed, 2)

    # Categorizar mudança
    if pct_changed < 1.0:
        result["category"] = "identical"       # < 1% mudou
    elif pct_bg_removed > 5.0:
        result["category"] = "bg_removed"      # fundo removido significativamente
    elif pct_changed < 10.0:
        result["category"] = "minor_change"    # pequenas mudanças (compressão, crop)
    else:
        result["category"] = "significant_change"

    # Verificar se houve crop (segmentada menor)
    if sh < oh * 0.9 or sw < ow * 0.9:
        result["was_cropped"] = True
    else:
        result["was_cropped"] = False

    return result


def get_segmented_path(orig_path):
    """Map original path to segmented path."""
    return orig_path.replace("/data/images/", "/data/images_segmented/")


def main():
    print("=" * 60)
    print("  ANÁLISE DE SEGMENTAÇÃO v2")
    print("  data/images/ vs data/images_segmented/")
    print("=" * 60)

    # Load all splits
    all_records = []
    for split in ["train", "val", "holdout"]:
        records = load_split(split)
        for path, cls, source, folder in records:
            all_records.append((path, cls, source, folder, split))

    print(f"\nTotal de imagens nos splits: {len(all_records)}")

    # Prepare comparison tasks
    tasks = []
    for orig_path, cls, source, folder, split in all_records:
        seg_path = get_segmented_path(orig_path)
        tasks.append((orig_path, seg_path, cls, source, folder))

    # Process in parallel
    print("Analisando imagens (4 workers)...")
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i, result in enumerate(executor.map(analyze_single_image, tasks, chunksize=50)):
            results.append(result)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)}...", flush=True)

    # Add split info
    for i, (_, _, _, _, split) in enumerate(all_records):
        results[i]["split"] = split

    print(f"Análise completa: {len(results)} imagens\n")

    # =========================================================================
    # ESTATÍSTICAS GERAIS
    # =========================================================================
    categories = defaultdict(int)
    for r in results:
        categories[r["category"]] += 1

    print("=" * 60)
    print("  1. DISTRIBUIÇÃO POR CATEGORIA")
    print("=" * 60)
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        print(f"  {cat:25s}: {count:6d} ({pct:5.1f}%)")

    # =========================================================================
    # POR FONTE (SOURCE)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  2. ANÁLISE POR FONTE")
    print("=" * 60)

    source_stats = defaultdict(lambda: {
        "total": 0, "categories": defaultdict(int),
        "mean_diff": [], "pct_bg": [], "pct_changed": [],
        "cropped": 0,
    })

    for r in results:
        src = r["source"]
        source_stats[src]["total"] += 1
        source_stats[src]["categories"][r["category"]] += 1
        if "mean_pixel_diff" in r:
            source_stats[src]["mean_diff"].append(r["mean_pixel_diff"])
            source_stats[src]["pct_bg"].append(r["pct_bg_removed"])
            source_stats[src]["pct_changed"].append(r["pct_pixels_changed"])
        if r.get("was_cropped"):
            source_stats[src]["cropped"] += 1

    source_summary = {}
    for src in sorted(source_stats.keys()):
        s = source_stats[src]
        summary = {
            "total": s["total"],
            "categories": dict(s["categories"]),
            "avg_pixel_diff": round(np.mean(s["mean_diff"]), 2) if s["mean_diff"] else 0,
            "avg_pct_bg_removed": round(np.mean(s["pct_bg"]), 2) if s["pct_bg"] else 0,
            "avg_pct_changed": round(np.mean(s["pct_changed"]), 2) if s["pct_changed"] else 0,
            "cropped_count": s["cropped"],
        }
        source_summary[src] = summary

        print(f"\n  [{src}] ({summary['total']} imgs)")
        print(f"    Avg pixel diff:      {summary['avg_pixel_diff']:.1f}")
        print(f"    Avg % BG removed:    {summary['avg_pct_bg_removed']:.1f}%")
        print(f"    Avg % pixels changed: {summary['avg_pct_changed']:.1f}%")
        print(f"    Cropped:             {summary['cropped_count']}")
        for cat, cnt in sorted(summary["categories"].items(), key=lambda x: -x[1]):
            pct = cnt / summary["total"] * 100
            print(f"    {cat:23s}: {cnt:5d} ({pct:5.1f}%)")

    # =========================================================================
    # POR CLASSE
    # =========================================================================
    print("\n" + "=" * 60)
    print("  3. ANÁLISE POR CLASSE")
    print("=" * 60)

    class_stats = defaultdict(lambda: {
        "total": 0, "categories": defaultdict(int),
        "mean_diff": [], "pct_bg": [],
    })

    for r in results:
        cls = r["class"]
        class_stats[cls]["total"] += 1
        class_stats[cls]["categories"][r["category"]] += 1
        if "mean_pixel_diff" in r:
            class_stats[cls]["mean_diff"].append(r["mean_pixel_diff"])
            class_stats[cls]["pct_bg"].append(r["pct_bg_removed"])

    class_summary = {}
    for cls in sorted(class_stats.keys()):
        s = class_stats[cls]
        summary = {
            "total": s["total"],
            "categories": dict(s["categories"]),
            "avg_pixel_diff": round(np.mean(s["mean_diff"]), 2) if s["mean_diff"] else 0,
            "avg_pct_bg_removed": round(np.mean(s["pct_bg"]), 2) if s["pct_bg"] else 0,
        }
        class_summary[cls] = summary

        bg_removed_pct = summary["categories"].get("bg_removed", 0) / summary["total"] * 100
        identical_pct = summary["categories"].get("identical", 0) / summary["total"] * 100
        print(f"  {cls:15s}: {summary['total']:5d} imgs | "
              f"BG removed: {bg_removed_pct:5.1f}% | "
              f"Identical: {identical_pct:5.1f}% | "
              f"Avg diff: {summary['avg_pixel_diff']:.1f}")

    # =========================================================================
    # HOLDOUT ESPECIAL
    # =========================================================================
    print("\n" + "=" * 60)
    print("  4. HOLDOUT (58 imagens de campo)")
    print("=" * 60)

    holdout_results = [r for r in results if r["split"] == "holdout"]
    for r in holdout_results:
        cat = r["category"]
        diff = r.get("mean_pixel_diff", 0)
        bg = r.get("pct_bg_removed", 0)
        print(f"  {r['source']:20s} | {r['class']:15s} | {cat:20s} | "
              f"diff={diff:5.1f} bg={bg:4.1f}%")

    holdout_cats = defaultdict(int)
    for r in holdout_results:
        holdout_cats[r["category"]] += 1
    print(f"\n  Holdout summary:")
    for cat, cnt in sorted(holdout_cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {cnt}")

    # =========================================================================
    # 3 CENÁRIOS DE DATASET
    # =========================================================================
    print("\n" + "=" * 60)
    print("  5. CENÁRIOS DE DATASET")
    print("=" * 60)

    # Cenário A: tudo original
    scenario_a = {"name": "A: Original (sem segmentação)", "total": len(results)}

    # Cenário B: tudo segmentado
    scenario_b = {"name": "B: Segmentado completo", "total": len(results)}

    # Cenário C: seletivo — segmentado onde houve remoção de fundo real,
    # original onde ficou igual
    scenario_c_seg = 0
    scenario_c_orig = 0
    for r in results:
        if r["category"] in ("bg_removed", "significant_change"):
            scenario_c_seg += 1
        else:
            scenario_c_orig += 1

    scenario_c = {
        "name": "C: Seletivo (segmentado só onde removeu fundo)",
        "segmented": scenario_c_seg,
        "original": scenario_c_orig,
        "total": len(results),
    }

    print(f"\n  Cenário A: {scenario_a['name']}")
    print(f"    Todas {scenario_a['total']} imagens originais")

    print(f"\n  Cenário B: {scenario_b['name']}")
    print(f"    Todas {scenario_b['total']} imagens segmentadas")

    print(f"\n  Cenário C: {scenario_c['name']}")
    print(f"    Segmentadas: {scenario_c_seg} ({scenario_c_seg/len(results)*100:.1f}%)")
    print(f"    Originais:   {scenario_c_orig} ({scenario_c_orig/len(results)*100:.1f}%)")

    # Quais fontes mais se beneficiam
    print("\n  Fontes que mais se beneficiam da segmentação:")
    for src in sorted(source_summary.keys(), key=lambda s: -source_summary[s]["avg_pct_bg_removed"]):
        s = source_summary[src]
        bg_count = s["categories"].get("bg_removed", 0)
        print(f"    {src:25s}: {s['avg_pct_bg_removed']:5.1f}% BG removido "
              f"({bg_count}/{s['total']} com remoção significativa)")

    # =========================================================================
    # RECOMENDAÇÃO
    # =========================================================================
    print("\n" + "=" * 60)
    print("  6. RECOMENDAÇÃO")
    print("=" * 60)

    total_bg_removed = categories.get("bg_removed", 0)
    total_identical = categories.get("identical", 0)
    total_minor = categories.get("minor_change", 0)

    pct_effective = total_bg_removed / len(results) * 100
    pct_unchanged = (total_identical + total_minor) / len(results) * 100

    if pct_effective > 30:
        recommendation = "B"
        reason = (f"Segmentação efetiva em {pct_effective:.1f}% das imagens. "
                  "Usar dataset segmentado completo (Cenário B).")
    elif pct_effective > 10:
        recommendation = "C"
        reason = (f"Segmentação efetiva em {pct_effective:.1f}% das imagens "
                  f"({pct_unchanged:.1f}% sem mudança). "
                  "Usar cenário seletivo (Cenário C) para maximizar benefício.")
    else:
        recommendation = "A"
        reason = (f"Segmentação efetiva em apenas {pct_effective:.1f}% das imagens. "
                  f"{pct_unchanged:.1f}% ficaram iguais. "
                  "Segmentação não agrega valor significativo — manter originais (Cenário A).")

    # Resultado empírico do autoresearch
    print(f"\n  Dados empíricos (autoresearch parcial):")
    print(f"    EfficientNet-B0 original:   67.2% holdout")
    print(f"    EfficientNet-B0 segmentado: 70.7% holdout (+3.5pp)")

    print(f"\n  Análise quantitativa:")
    print(f"    Imagens com BG removido:  {total_bg_removed} ({pct_effective:.1f}%)")
    print(f"    Imagens sem mudança:      {total_identical + total_minor} ({pct_unchanged:.1f}%)")

    print(f"\n  >>> RECOMENDAÇÃO: Cenário {recommendation}")
    print(f"  >>> {reason}")

    if recommendation != "A":
        print(f"\n  Nota: resultado empírico confirma — segmentação melhorou "
              f"holdout de 67.2% → 70.7% (+3.5pp)")

    # =========================================================================
    # SALVAR REPORT
    # =========================================================================
    report_data = {
        "total_images": len(results),
        "categories": dict(categories),
        "source_summary": source_summary,
        "class_summary": class_summary,
        "holdout_categories": dict(holdout_cats),
        "scenarios": {
            "A": scenario_a,
            "B": scenario_b,
            "C": scenario_c,
        },
        "recommendation": {
            "scenario": recommendation,
            "reason": reason,
            "empirical": {
                "original_holdout": 67.2,
                "segmented_holdout": 70.7,
                "improvement_pp": 3.5,
            }
        },
    }

    with open(REPORT_JSON, "w") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n  JSON salvo: {REPORT_JSON}")

    # Markdown report
    md = []
    md.append("# Relatório de Segmentação v2\n")
    md.append(f"**Total de imagens analisadas**: {len(results)}\n")

    md.append("\n## Distribuição por Categoria\n")
    md.append("| Categoria | Count | % |")
    md.append("|--|--|--|")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        md.append(f"| {cat} | {count} | {pct:.1f}% |")

    md.append("\n## Por Fonte\n")
    md.append("| Fonte | Total | Avg Diff | Avg BG Removed | BG Count |")
    md.append("|--|--|--|--|--|")
    for src in sorted(source_summary.keys()):
        s = source_summary[src]
        bg_count = s["categories"].get("bg_removed", 0)
        md.append(f"| {src} | {s['total']} | {s['avg_pixel_diff']:.1f} | "
                  f"{s['avg_pct_bg_removed']:.1f}% | {bg_count} |")

    md.append("\n## Por Classe\n")
    md.append("| Classe | Total | Avg Diff | Avg BG Removed |")
    md.append("|--|--|--|--|")
    for cls in sorted(class_summary.keys()):
        s = class_summary[cls]
        md.append(f"| {cls} | {s['total']} | {s['avg_pixel_diff']:.1f} | "
                  f"{s['avg_pct_bg_removed']:.1f}% |")

    md.append("\n## Cenários\n")
    md.append(f"- **A**: Original — todas {len(results)} imagens sem segmentação")
    md.append(f"- **B**: Segmentado completo — todas {len(results)} imagens segmentadas")
    md.append(f"- **C**: Seletivo — {scenario_c_seg} segmentadas + "
              f"{scenario_c_orig} originais")

    md.append(f"\n## Recomendação\n")
    md.append(f"**Cenário {recommendation}**: {reason}\n")
    md.append(f"Resultado empírico: original 67.2% → segmentado 70.7% (+3.5pp holdout)")

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"  Markdown salvo: {REPORT_MD}")


if __name__ == "__main__":
    main()
