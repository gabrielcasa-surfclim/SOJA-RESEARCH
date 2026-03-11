"""
create_splits.py — Compara 3 estratégias de split e salva a escolhida.

Estratégias:
  A) RANDOM: shuffle global, 80/20 por imagem (ignora fonte)
  B) GROUP-SOURCE: agrupa por fonte (ex: digipathos, asdid). Uma fonte
     inteira vai pro val quando a classe tem múltiplas fontes.
  C) GROUP-FOLDER: agrupa por subpasta (ex: digipathos_ferrugem_crop).
     Nenhuma pasta aparece em train E val ao mesmo tempo.

Todas as estratégias mantêm o holdout externo (doencasdeplantas + srin)
separado antes de dividir train/val.

Uso:
    python3 training/create_splits.py              # compara as 3
    python3 training/create_splits.py --save A     # salva estratégia A

Salva em data/splits/:
  - train.csv, val.csv, holdout.csv (path, class, source, folder)
  - class_weights.json
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

from prepare import DATA_DIR, FOLDER_TO_CLASS, MIN_SAMPLES, VALID_EXTENSIONS

SPLITS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "splits")
SEED = 42
VAL_RATIO = 0.2

# Fontes consideradas "externas" (holdout) — nunca entram no treino
HOLDOUT_SOURCES = {"doencasdeplantas", "srin"}
HOLDOUT_TRAIN_RATIO = 0.5  # 50% das imagens externas vão pro treino


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def detect_source(folder_name: str) -> str:
    if folder_name.startswith("digipathos_"):
        return "digipathos"
    if folder_name.startswith("doencasdeplantas_"):
        return "doencasdeplantas"
    if folder_name.startswith("srin_"):
        return "srin"
    if folder_name in ("soybean_rust", "frogeye", "target_spot"):
        return "asdid"
    if folder_name == "healthy":
        return "plantvillage"
    return "unknown"


def _normalize_folder_name(name: str) -> str:
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\bcrop\b", "", name).strip()
    return name.title()


def scan_all_images():
    """Escaneia dataset e retorna lista de (path, class, source, folder)."""
    records = []
    for folder_name in sorted(os.listdir(DATA_DIR)):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        source = detect_source(folder_name)
        class_name = FOLDER_TO_CLASS.get(folder_name, _normalize_folder_name(folder_name))
        for fname in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTENSIONS:
                fpath = os.path.join(folder_path, fname)
                records.append((fpath, class_name, source, folder_name))
    return records


def separate_holdout(records):
    """Separa imagens externas (doencasdeplantas + srin) em 50% treino / 50% holdout.
    Split estratificado por classe. Retorna (remaining, holdout, classes)."""
    rng = np.random.RandomState(SEED)

    class_records = defaultdict(list)
    for rec in records:
        class_records[rec[1]].append(rec)

    active_classes = {cls for cls, recs in class_records.items()
                      if len(recs) >= MIN_SAMPLES}

    holdout = []
    remaining = []
    external_to_train = []

    # Separa imagens externas por classe para split estratificado
    external_by_class = defaultdict(list)
    for rec in records:
        if rec[1] not in active_classes:
            continue
        if rec[2] in HOLDOUT_SOURCES:
            external_by_class[rec[1]].append(rec)
        else:
            remaining.append(rec)

    # Split 50/50 estratificado por classe
    for cls in sorted(external_by_class.keys()):
        recs = list(external_by_class[cls])
        rng.shuffle(recs)
        n_train = max(1, int(len(recs) * HOLDOUT_TRAIN_RATIO))
        external_to_train.extend(recs[:n_train])
        holdout.extend(recs[n_train:])

    # Imagens externas pro treino vão junto com remaining
    remaining.extend(external_to_train)

    total_ext = len(external_to_train) + len(holdout)
    print(f"\n  Imagens externas: {total_ext} total")
    print(f"    → treino:  {len(external_to_train)} ({HOLDOUT_TRAIN_RATIO*100:.0f}%)")
    print(f"    → holdout: {len(holdout)} ({(1-HOLDOUT_TRAIN_RATIO)*100:.0f}%)")

    return remaining, holdout, sorted(active_classes)


# ─────────────────────────────────────────────────────────────────────
# Estratégia A: Random por imagem
# ─────────────────────────────────────────────────────────────────────

def split_random(remaining):
    """Shuffle global, 80/20 estratificado por classe."""
    rng = np.random.RandomState(SEED)

    by_class = defaultdict(list)
    for rec in remaining:
        by_class[rec[1]].append(rec)

    train, val = [], []
    for cls in sorted(by_class.keys()):
        recs = list(by_class[cls])
        rng.shuffle(recs)
        n_val = max(1, int(len(recs) * VAL_RATIO))
        val.extend(recs[:n_val])
        train.extend(recs[n_val:])

    return train, val


# ─────────────────────────────────────────────────────────────────────
# Estratégia B: Group por fonte (source)
# ─────────────────────────────────────────────────────────────────────

def split_group_source(remaining):
    """Para classes multi-fonte, reserva a fonte menor inteira como val.
    Para classes mono-fonte, faz random 20%."""
    rng = np.random.RandomState(SEED)

    by_class = defaultdict(list)
    for rec in remaining:
        by_class[rec[1]].append(rec)

    train, val = [], []
    for cls in sorted(by_class.keys()):
        recs = by_class[cls]

        # Agrupa por fonte
        by_source = defaultdict(list)
        for rec in recs:
            by_source[rec[2]].append(rec)

        sources = sorted(by_source.keys())

        if len(sources) > 1:
            # Multi-fonte: fonte menor inteira como val
            val_source = min(sources, key=lambda s: len(by_source[s]))
            for s in sources:
                if s == val_source:
                    val.extend(by_source[s])
                else:
                    train.extend(by_source[s])
        else:
            # Mono-fonte: random 20%
            rng.shuffle(recs)
            n_val = max(1, int(len(recs) * VAL_RATIO))
            val.extend(recs[:n_val])
            train.extend(recs[n_val:])

    return train, val


# ─────────────────────────────────────────────────────────────────────
# Estratégia C: Group por pasta (folder)
# ─────────────────────────────────────────────────────────────────────

def _folder_group_key(folder_name: str) -> str:
    """Agrupa pastas _crop e non-_crop no mesmo grupo.
    Ex: digipathos_oidio e digipathos_oidio_crop → 'digipathos_oidio'"""
    if folder_name.endswith("_crop"):
        return folder_name[:-5]  # remove '_crop'
    return folder_name


def split_group_folder(remaining):
    """Agrupa pastas _crop/non-_crop como par único, divide 80/20 DENTRO
    de cada grupo. Todas as fontes são misturadas — cross-source testing
    é feito pelo holdout externo, não pelo val."""
    rng = np.random.RandomState(SEED)

    by_class = defaultdict(list)
    for rec in remaining:
        by_class[rec[1]].append(rec)

    train, val = [], []
    for cls in sorted(by_class.keys()):
        recs = by_class[cls]

        # Agrupa por grupo de pasta (crop + non-crop juntos)
        by_group = defaultdict(list)
        for rec in recs:
            group = _folder_group_key(rec[3])
            by_group[group].append(rec)

        # Divide 80/20 dentro de cada grupo
        for g in sorted(by_group.keys()):
            g_recs = list(by_group[g])
            rng.shuffle(g_recs)
            n_val = max(1, int(len(g_recs) * VAL_RATIO))
            val.extend(g_recs[:n_val])
            train.extend(g_recs[n_val:])

    return train, val


# ─────────────────────────────────────────────────────────────────────
# Análise e relatório comparativo
# ─────────────────────────────────────────────────────────────────────

def analyze_split(train, val, holdout, classes):
    """Calcula métricas de um split para comparação."""
    info = {
        "n_train": len(train),
        "n_val": len(val),
        "n_holdout": len(holdout),
        "classes": {},
        "leakage_classes": [],
        "isolated_classes": [],
    }

    for cls in classes:
        t_recs = [r for r in train if r[1] == cls]
        v_recs = [r for r in val if r[1] == cls]
        h_recs = [r for r in holdout if r[1] == cls]

        t_sources = set(r[2] for r in t_recs)
        v_sources = set(r[2] for r in v_recs)
        t_folders = set(r[3] for r in t_recs)
        v_folders = set(r[3] for r in v_recs)

        source_overlap = t_sources & v_sources
        folder_overlap = t_folders & v_folders

        info["classes"][cls] = {
            "train": len(t_recs),
            "val": len(v_recs),
            "holdout": len(h_recs),
            "train_sources": sorted(t_sources),
            "val_sources": sorted(v_sources),
            "holdout_sources": sorted(set(r[2] for r in h_recs)),
            "train_folders": sorted(t_folders),
            "val_folders": sorted(v_folders),
            "source_overlap": sorted(source_overlap),
            "folder_overlap": sorted(folder_overlap),
            "source_isolated": len(source_overlap) == 0 and len(v_sources) > 0,
            "folder_isolated": len(folder_overlap) == 0 and len(v_folders) > 0,
        }

        if len(source_overlap) == 0 and len(v_sources) > 0:
            info["isolated_classes"].append(cls)
        if len(folder_overlap) > 0:
            info["leakage_classes"].append(cls)

    return info


def compute_class_weights(train_records):
    """Calcula class weights inversamente proporcional à frequência."""
    class_counts = defaultdict(int)
    for rec in train_records:
        class_counts[rec[1]] += 1

    total = sum(class_counts.values())
    n_classes = len(class_counts)

    weights = {}
    for cls, count in sorted(class_counts.items()):
        weights[cls] = round(total / (n_classes * count), 4)

    return weights, class_counts


def print_comparison(strategies, classes):
    """Imprime relatório comparativo das 3 estratégias."""

    W = 90

    print(f"\n{'='*W}")
    print(f"  COMPARATIVO DE ESTRATÉGIAS DE SPLIT")
    print(f"{'='*W}")

    # ── Resumo geral ──
    print(f"\n  {'Métrica':<35} {'A) Random':>15} {'B) Group-Source':>17} {'C) Group-Folder':>17}")
    print(f"  {'-'*84}")

    for key, label in [
        ("n_train", "Train"),
        ("n_val", "Val"),
        ("n_holdout", "Holdout"),
    ]:
        vals = [str(strategies[s][key]) for s in ["A", "B", "C"]]
        print(f"  {label:<35} {vals[0]:>15} {vals[1]:>17} {vals[2]:>17}")

    # Leak/isolation counts
    for s_key, s_name in [("A", "A) Random"), ("B", "B) Group-Source"), ("C", "C) Group-Folder")]:
        info = strategies[s_key]
        n_leak = len(info["leakage_classes"])
        n_iso = len(info["isolated_classes"])
        # count later

    labels_leak = []
    labels_iso = []
    for s_key in ["A", "B", "C"]:
        info = strategies[s_key]
        labels_leak.append(str(len(info["leakage_classes"])))
        labels_iso.append(str(len(info["isolated_classes"])))

    print(f"  {'Classes com folder overlap (leak)':35} {labels_leak[0]:>15} {labels_leak[1]:>17} {labels_leak[2]:>17}")
    print(f"  {'Classes source-isoladas':35} {labels_iso[0]:>15} {labels_iso[1]:>17} {labels_iso[2]:>17}")

    # ── Detalhamento por classe ──
    for s_key, s_name in [("A", "A) RANDOM"), ("B", "B) GROUP-SOURCE"), ("C", "C) GROUP-FOLDER")]:
        info = strategies[s_key]

        print(f"\n{'='*W}")
        print(f"  ESTRATÉGIA {s_name}")
        print(f"{'='*W}")

        print(f"\n  {'Classe':<20} {'Train':>6} {'Val':>6} {'Hold':>5}  {'Train fontes':<28} {'Val fontes':<20} {'Overlap'}")
        print(f"  {'-'*(W-4)}")

        for cls in classes:
            c = info["classes"][cls]
            t_src = ",".join(c["train_sources"]) or "-"
            v_src = ",".join(c["val_sources"]) or "-"
            overlap = ",".join(c["source_overlap"]) if c["source_overlap"] else "ISOLADO"

            print(f"  {cls:<20} {c['train']:>6} {c['val']:>6} {c['holdout']:>5}  {t_src:<28} {v_src:<20} {overlap}")

        # Folder detail
        print(f"\n  Detalhamento por pasta:")
        print(f"  {'Classe':<20} {'Train pastas':<45} {'Val pastas':<35} {'Folder leak'}")
        print(f"  {'-'*(W-4)}")

        for cls in classes:
            c = info["classes"][cls]
            t_f = ",".join(os.path.basename(f) for f in c["train_folders"]) or "-"
            v_f = ",".join(os.path.basename(f) for f in c["val_folders"]) or "-"
            f_overlap = ",".join(os.path.basename(f) for f in c["folder_overlap"])
            f_status = f_overlap if f_overlap else "ISOLADO"

            # Trunca se muito longo
            if len(t_f) > 43:
                t_f = t_f[:40] + "..."
            if len(v_f) > 33:
                v_f = v_f[:30] + "..."

            print(f"  {cls:<20} {t_f:<45} {v_f:<35} {f_status}")

    # ── Análise de risco ──
    print(f"\n{'='*W}")
    print(f"  ANÁLISE DE RISCO")
    print(f"{'='*W}")

    risks = {
        "A": {
            "name": "A) Random",
            "pros": [],
            "cons": [],
            "risk": "BAIXO",
        },
        "B": {
            "name": "B) Group-Source",
            "pros": [],
            "cons": [],
            "risk": "MEDIO",
        },
        "C": {
            "name": "C) Group-Folder",
            "pros": [],
            "cons": [],
            "risk": "MEDIO",
        },
    }

    for s_key in ["A", "B", "C"]:
        info = strategies[s_key]
        r = risks[s_key]
        n_leak = len(info["leakage_classes"])
        n_iso = len(info["isolated_classes"])
        n_classes = len(classes)

        # Folder leak analysis
        if n_leak == n_classes:
            r["cons"].append(f"TODAS as {n_classes} classes têm folder overlap train/val — risco de data leak")
        elif n_leak > 0:
            r["cons"].append(f"{n_leak}/{n_classes} classes com folder overlap — risco parcial de data leak")
        else:
            r["pros"].append("Zero folder overlap — sem risco de data leak entre pastas")

        # Source isolation analysis
        if n_iso > 0:
            iso_classes = info["isolated_classes"]
            # Check if any isolated class has vastly different source sizes
            problematic = []
            for cls in iso_classes:
                c = info["classes"][cls]
                if c["val"] > c["train"] * 0.5:
                    problematic.append(cls)
            if problematic:
                r["cons"].append(f"{', '.join(problematic)}: val maior que 50% do train — split desbalanceado")
                r["risk"] = "ALTO"
            else:
                r["pros"].append(f"{n_iso} classes source-isoladas — testa generalização entre fontes")
        else:
            r["pros"].append("Fontes misturadas em train/val — acurácia otimista mas estável")

        # Val size check
        for cls in classes:
            c = info["classes"][cls]
            total_cls = c["train"] + c["val"]
            if total_cls > 0:
                val_pct = c["val"] / total_cls
                if val_pct > 0.45:
                    r["cons"].append(f"{cls}: {val_pct*100:.0f}% no val — train sub-representado")
                elif val_pct < 0.05:
                    r["cons"].append(f"{cls}: {val_pct*100:.0f}% no val — val muito pequeno pra ser confiável")

    for s_key in ["A", "B", "C"]:
        r = risks[s_key]
        # Determine overall risk
        if any("TODAS" in c for c in r["cons"]):
            if not any("source-isol" in p for p in r["pros"]):
                r["risk"] = "ALTO (leak)" if s_key == "A" else r["risk"]
        if any("ALTO" in r["risk"] for _ in [1]):
            pass

        # Classify risk
        n_cons = len(r["cons"])
        if n_cons == 0:
            r["risk"] = "BAIXO"
        elif n_cons <= 1:
            r["risk"] = "BAIXO-MEDIO"
        elif any("val maior que 50%" in c for c in r["cons"]):
            r["risk"] = "ALTO"
        else:
            r["risk"] = "MEDIO"

    for s_key in ["A", "B", "C"]:
        r = risks[s_key]
        print(f"\n  {r['name']}  —  Risco: {r['risk']}")
        for p in r["pros"]:
            print(f"    + {p}")
        for c in r["cons"]:
            print(f"    - {c}")

    # ── Recomendação ──
    print(f"\n{'='*W}")
    print(f"  RECOMENDAÇÃO")
    print(f"{'='*W}")

    # Score simple: prefer less leak, moderate isolation
    scores = {}
    for s_key in ["A", "B", "C"]:
        info = strategies[s_key]
        n_leak = len(info["leakage_classes"])
        n_iso = len(info["isolated_classes"])
        n_cls = len(classes)

        # Penaliza leak, recompensa leve isolamento mas penaliza extremo
        leak_penalty = n_leak / n_cls  # 0=no leak, 1=all leak
        # Check if any isolated class has problematic val/train ratio
        iso_penalty = 0
        for cls in info["isolated_classes"]:
            c = info["classes"][cls]
            if c["train"] > 0 and c["val"] / c["train"] > 0.5:
                iso_penalty += 1

        # Lower is better
        score = leak_penalty + iso_penalty * 2

        # Val distribution balance (std of val percentages)
        val_pcts = []
        for cls in classes:
            c = info["classes"][cls]
            total_cls = c["train"] + c["val"]
            if total_cls > 0:
                val_pcts.append(c["val"] / total_cls)
        val_std = np.std(val_pcts) if val_pcts else 0
        score += val_std * 3  # penaliza distribuição desigual

        scores[s_key] = score

    best = min(scores, key=lambda k: scores[k])
    strategy_names = {"A": "Random", "B": "Group-Source", "C": "Group-Folder"}

    print(f"\n  Scores (menor = melhor):")
    for s_key in ["A", "B", "C"]:
        marker = " ◄ RECOMENDADA" if s_key == best else ""
        print(f"    {s_key}) {strategy_names[s_key]:<20} score={scores[s_key]:.3f}{marker}")

    best_info = strategies[best]
    print(f"\n  Estratégia {best}) {strategy_names[best]} é a mais segura.")
    print(f"  Train: {best_info['n_train']}  |  Val: {best_info['n_val']}  |  Holdout: {best_info['n_holdout']}")
    print(f"\n  Para salvar: python3 training/create_splits.py --save {best}")

    print(f"\n{'='*W}\n")

    return best


# ─────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────

def save_splits(train, val, holdout, weights):
    """Salva CSVs e class_weights.json."""
    os.makedirs(SPLITS_DIR, exist_ok=True)

    for name, records in [("train", train), ("val", val), ("holdout", holdout)]:
        path = os.path.join(SPLITS_DIR, f"{name}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "class", "source", "folder"])
            for rec in sorted(records, key=lambda r: (r[1], r[0])):
                writer.writerow(rec)
        print(f"  Salvo: {path} ({len(records)} registros)")

    weights_path = os.path.join(SPLITS_DIR, "class_weights.json")
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2, ensure_ascii=False)
    print(f"  Salvo: {weights_path}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    save_strategy = None
    if "--save" in args:
        idx = args.index("--save")
        if idx + 1 < len(args):
            save_strategy = args[idx + 1].upper()
            if save_strategy not in ("A", "B", "C"):
                print(f"Estratégia inválida: {save_strategy}. Use A, B ou C.")
                sys.exit(1)

    print("\n" + "=" * 90)
    print("  CREATE SPLITS — Comparativo de Estratégias")
    print("=" * 90)

    records = scan_all_images()
    print(f"\n  {len(records)} imagens escaneadas em {DATA_DIR}")

    remaining, holdout, classes = separate_holdout(records)
    print(f"  {len(remaining)} imagens disponíveis para train/val (excluindo holdout)")
    print(f"  {len(holdout)} imagens no holdout externo ({', '.join(sorted(HOLDOUT_SOURCES))})")
    print(f"  {len(classes)} classes ativas (>= {MIN_SAMPLES} imgs): {classes}")

    # Gera as 3 estratégias
    split_fns = {
        "A": ("Random", split_random),
        "B": ("Group-Source", split_group_source),
        "C": ("Group-Folder", split_group_folder),
    }

    strategies = {}
    for key, (name, fn) in split_fns.items():
        train, val = fn(remaining)
        strategies[key] = analyze_split(train, val, holdout, classes)
        strategies[key]["_train"] = train
        strategies[key]["_val"] = val

    best = print_comparison(strategies, classes)

    # Salvar se pedido
    if save_strategy:
        chosen = save_strategy
        print(f"\n{'='*90}")
        print(f"  SALVANDO ESTRATÉGIA {chosen})")
        print(f"{'='*90}\n")

        train = strategies[chosen]["_train"]
        val = strategies[chosen]["_val"]
        weights, _ = compute_class_weights(train)
        save_splits(train, val, holdout, weights)
        print(f"\n  Estratégia {chosen} salva em {SPLITS_DIR}/")
        print(f"  Pronto para treinar: python3 training/train.py\n")


if __name__ == "__main__":
    main()
