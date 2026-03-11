"""
audit_dataset.py — Auditoria completa do dataset de imagens.

Relatório:
  1. Imagens por classe e por fonte
  2. Duplicatas (MD5) entre fontes
  3. Desbalanceamento entre classes
  4. Sugestão de holdout externo

Uso:
    python3 training/audit_dataset.py
"""

import hashlib
import os
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config — importa mapeamentos do prepare.py
# ---------------------------------------------------------------------------
from prepare import DATA_DIR, FOLDER_TO_CLASS, MIN_SAMPLES, VALID_EXTENSIONS


def _normalize_folder_name(name: str) -> str:
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\bcrop\b", "", name).strip()
    return name.title()


# ---------------------------------------------------------------------------
# Detecta fonte pelo prefixo da pasta
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Scan dataset
# ---------------------------------------------------------------------------
def scan_dataset():
    """Escaneia data/images/ e retorna estrutura completa."""
    # folder -> list of (filepath, filename)
    folders = {}
    # class -> source -> list of filepaths
    class_source_files = defaultdict(lambda: defaultdict(list))
    # all files for duplicate detection: md5 -> list of (filepath, folder, source)
    all_files = []

    if not os.path.isdir(DATA_DIR):
        print(f"Erro: {DATA_DIR} nao encontrado")
        return None

    for folder_name in sorted(os.listdir(DATA_DIR)):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        source = detect_source(folder_name)
        class_name = FOLDER_TO_CLASS.get(folder_name, _normalize_folder_name(folder_name))

        files = []
        for fname in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTENSIONS:
                fpath = os.path.join(folder_path, fname)
                files.append(fpath)
                all_files.append((fpath, folder_name, source, class_name))

        folders[folder_name] = {
            "source": source,
            "class": class_name,
            "count": len(files),
            "files": files,
        }
        class_source_files[class_name][source].extend(files)

    return folders, class_source_files, all_files


# ---------------------------------------------------------------------------
# 1. Relatório por pasta
# ---------------------------------------------------------------------------
def report_folders(folders):
    print("=" * 80)
    print("  1. IMAGENS POR PASTA")
    print("=" * 80)
    print(f"\n  {'Pasta':<50} {'Fonte':<18} {'Classe':<22} {'Imgs':>5}")
    print("  " + "-" * 97)

    total = 0
    for name, info in sorted(folders.items()):
        print(f"  {name:<50} {info['source']:<18} {info['class']:<22} {info['count']:>5}")
        total += info["count"]

    print("  " + "-" * 97)
    print(f"  {'TOTAL':<50} {'':<18} {'':<22} {total:>5}")
    print(f"\n  {len(folders)} pastas, {total} imagens\n")


# ---------------------------------------------------------------------------
# 2. Relatório por classe e fonte
# ---------------------------------------------------------------------------
def report_classes(class_source_files):
    print("=" * 80)
    print("  2. IMAGENS POR CLASSE E FONTE")
    print("=" * 80)

    sources_all = sorted({s for cls in class_source_files.values() for s in cls.keys()})

    # Header
    header = f"\n  {'Classe':<25}"
    for s in sources_all:
        header += f" {s:>15}"
    header += f" {'TOTAL':>8} {'Status':>10}"
    print(header)
    print("  " + "-" * (25 + 16 * len(sources_all) + 20))

    grand_total = 0
    class_totals = {}

    for class_name in sorted(class_source_files.keys()):
        sources = class_source_files[class_name]
        row = f"  {class_name:<25}"
        total = 0
        for s in sources_all:
            count = len(sources.get(s, []))
            row += f" {count:>15}" if count > 0 else f" {'-':>15}"
            total += count
        status = "ATIVA" if total >= MIN_SAMPLES else "ignorada"
        row += f" {total:>8} {status:>10}"
        print(row)
        grand_total += total
        class_totals[class_name] = total

    print("  " + "-" * (25 + 16 * len(sources_all) + 20))
    print(f"  {'TOTAL':<25}", end="")
    for s in sources_all:
        src_total = sum(len(class_source_files[c].get(s, [])) for c in class_source_files)
        print(f" {src_total:>15}", end="")
    print(f" {grand_total:>8}")

    # Classes ativas
    active = {c: t for c, t in class_totals.items() if t >= MIN_SAMPLES}
    ignored = {c: t for c, t in class_totals.items() if t < MIN_SAMPLES}
    print(f"\n  Classes ativas (>= {MIN_SAMPLES} imgs): {len(active)}")
    print(f"  Classes ignoradas (< {MIN_SAMPLES} imgs): {len(ignored)}")
    print()

    return class_totals, active


# ---------------------------------------------------------------------------
# 3. Detecção de duplicatas por MD5
# ---------------------------------------------------------------------------
def report_duplicates(all_files):
    print("=" * 80)
    print("  3. DUPLICATAS (MD5)")
    print("=" * 80)

    print("\n  Calculando hashes MD5...")

    md5_map = defaultdict(list)  # hash -> list of (path, folder, source, class)
    total = len(all_files)

    for i, (fpath, folder, source, class_name) in enumerate(all_files):
        if (i + 1) % 2000 == 0 or i + 1 == total:
            print(f"  Processadas: {i+1}/{total}", end="\r")
        try:
            h = hashlib.md5(open(fpath, "rb").read()).hexdigest()
            md5_map[h].append((fpath, folder, source, class_name))
        except Exception:
            pass

    print(f"  Processadas: {total}/{total}    ")

    # Find duplicates
    duplicates = {h: entries for h, entries in md5_map.items() if len(entries) > 1}

    # Cross-source duplicates (more interesting)
    cross_source = {}
    same_source = {}
    cross_class = {}

    for h, entries in duplicates.items():
        sources = set(e[2] for e in entries)
        classes = set(e[3] for e in entries)
        if len(classes) > 1:
            cross_class[h] = entries
        elif len(sources) > 1:
            cross_source[h] = entries
        else:
            same_source[h] = entries

    total_dup_files = sum(len(e) for e in duplicates.values())
    unique_dups = len(duplicates)

    print(f"\n  Imagens unicas (por hash): {len(md5_map)}")
    print(f"  Grupos duplicados:         {unique_dups}")
    print(f"  Arquivos duplicados:       {total_dup_files}")
    print(f"\n  Duplicatas entre fontes:   {len(cross_source)} grupos")
    print(f"  Duplicatas mesma fonte:    {len(same_source)} grupos")
    print(f"  Duplicatas entre CLASSES:  {len(cross_class)} grupos (PROBLEMA!)")

    if cross_class:
        print(f"\n  *** ATENCAO: {len(cross_class)} imagens aparecem em classes diferentes! ***")
        for h, entries in list(cross_class.items())[:5]:
            print(f"\n  Hash: {h}")
            for fpath, folder, source, class_name in entries:
                fname = os.path.basename(fpath)
                print(f"    [{source}] {class_name} <- {folder}/{fname}")

    if cross_source:
        print(f"\n  Duplicatas entre fontes (primeiros 10):")
        for h, entries in list(cross_source.items())[:10]:
            classes = entries[0][3]
            sources_str = ", ".join(sorted(set(e[2] for e in entries)))
            fname = os.path.basename(entries[0][0])
            print(f"    {fname[:40]:<42} classe={classes:<15} fontes={sources_str}")

    removable = sum(len(e) - 1 for e in duplicates.values())
    print(f"\n  Duplicatas removiveis: {removable} arquivos")
    print()

    return duplicates, md5_map


# ---------------------------------------------------------------------------
# 4. Desbalanceamento
# ---------------------------------------------------------------------------
def report_imbalance(active_classes):
    print("=" * 80)
    print("  4. DESBALANCEAMENTO (classes ativas)")
    print("=" * 80)

    if not active_classes:
        print("\n  Sem classes ativas.\n")
        return

    counts = sorted(active_classes.items(), key=lambda x: -x[1])
    max_count = counts[0][1]
    min_count = counts[-1][1]
    mean_count = sum(c for _, c in counts) / len(counts)
    ratio = max_count / max(min_count, 1)

    print(f"\n  {'Classe':<25} {'Imgs':>7} {'Ratio':>7} {'Barra'}")
    print("  " + "-" * 70)

    for class_name, count in counts:
        bar_len = int(40 * count / max_count)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        r = count / max(min_count, 1)
        print(f"  {class_name:<25} {count:>7} {r:>6.1f}x  {bar}")

    print(f"\n  Maior classe:   {counts[0][0]} ({max_count})")
    print(f"  Menor classe:   {counts[-1][0]} ({min_count})")
    print(f"  Media:          {mean_count:.0f}")
    print(f"  Ratio max/min:  {ratio:.1f}x")

    if ratio > 10:
        print(f"\n  ⚠  Desbalanceamento SEVERO ({ratio:.0f}x).")
        print(f"     Considere: oversampling das classes menores, class weights, ou coletar mais dados.")
    elif ratio > 5:
        print(f"\n  ⚠  Desbalanceamento MODERADO ({ratio:.1f}x).")
        print(f"     Data augmentation e class weights podem ajudar.")
    else:
        print(f"\n  ✓  Desbalanceamento ACEITAVEL ({ratio:.1f}x).")

    print()


# ---------------------------------------------------------------------------
# 5. Sugestão de holdout externo
# ---------------------------------------------------------------------------
def report_holdout(class_source_files, active_classes):
    print("=" * 80)
    print("  5. SUGESTAO DE HOLDOUT EXTERNO")
    print("=" * 80)

    print("""
  Estrategia: separar imagens de uma fonte diferente como holdout externo.
  Isso testa se o modelo generaliza para imagens "nunca vistas" de contexto diferente.
  Fontes menores (doencasdeplantas, srin) sao ideais — fotos de campo reais.
""")

    # For each active class, check which sources have data
    print(f"  {'Classe':<25} {'Fonte holdout':<20} {'Holdout':>8} {'Treino':>8} {'% hold':>7}")
    print("  " + "-" * 72)

    total_holdout = 0
    total_train = 0
    holdout_plan = {}

    for class_name in sorted(active_classes.keys()):
        sources = class_source_files[class_name]
        total = active_classes[class_name]

        # Prefer doencasdeplantas or srin as holdout (field photos, different distribution)
        holdout_source = None
        holdout_count = 0
        for preferred in ["doencasdeplantas", "srin"]:
            if preferred in sources and len(sources[preferred]) >= 3:
                holdout_source = preferred
                holdout_count = len(sources[preferred])
                break

        if holdout_source and holdout_count / total <= 0.2:
            train_count = total - holdout_count
            pct = holdout_count / total * 100
            print(f"  {class_name:<25} {holdout_source:<20} {holdout_count:>8} {train_count:>8} {pct:>6.1f}%")
            total_holdout += holdout_count
            total_train += train_count
            holdout_plan[class_name] = {
                "source": holdout_source,
                "count": holdout_count,
                "files": sources[holdout_source],
            }
        else:
            reason = "sem fonte externa" if not holdout_source else f"holdout > 20% ({holdout_count}/{total})"
            print(f"  {class_name:<25} {'—':<20} {'—':>8} {total:>8} {'—':>7}  ({reason})")
            total_train += total

    print("  " + "-" * 72)
    print(f"  {'TOTAL':<25} {'':<20} {total_holdout:>8} {total_train:>8}")

    if holdout_plan:
        print(f"\n  Holdout externo: {total_holdout} imagens de {len(holdout_plan)} classes")
        print(f"  Treino+val:      {total_train} imagens")
        print(f"\n  Para implementar, mova as pastas de holdout para data/holdout/ antes de treinar.")
    else:
        print(f"\n  Nenhum holdout sugerido — fontes externas insuficientes.")

    print()

    return holdout_plan


# ---------------------------------------------------------------------------
# 6. Resumo de fontes
# ---------------------------------------------------------------------------
def report_sources(all_files):
    print("=" * 80)
    print("  6. RESUMO POR FONTE")
    print("=" * 80)

    source_counts = defaultdict(int)
    source_classes = defaultdict(set)
    for fpath, folder, source, class_name in all_files:
        source_counts[source] += 1
        source_classes[source].add(class_name)

    print(f"\n  {'Fonte':<20} {'Imagens':>8} {'Classes':>8} {'Tipo'}")
    print("  " + "-" * 55)
    for source in sorted(source_counts.keys(), key=lambda s: -source_counts[s]):
        count = source_counts[source]
        n_classes = len(source_classes[source])
        tipo = {
            "asdid": "Academico (campo)",
            "digipathos": "Academico (EMBRAPA)",
            "plantvillage": "Academico (lab)",
            "doencasdeplantas": "Site BR (campo)",
            "srin": "Site US (campo)",
            "unknown": "Desconhecido",
        }.get(source, "—")
        print(f"  {source:<20} {count:>8} {n_classes:>8}  {tipo}")

    print(f"\n  Total: {sum(source_counts.values())} imagens de {len(source_counts)} fontes")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 80)
    print("  AUDIT DATASET — Soja Research")
    print(f"  Diretorio: {DATA_DIR}")
    print(f"  MIN_SAMPLES: {MIN_SAMPLES}")
    print("=" * 80 + "\n")

    result = scan_dataset()
    if result is None:
        return

    folders, class_source_files, all_files = result

    report_folders(folders)
    class_totals, active_classes = report_classes(class_source_files)
    report_sources(all_files)
    duplicates, md5_map = report_duplicates(all_files)
    report_imbalance(active_classes)
    report_holdout(class_source_files, active_classes)

    # Final summary
    print("=" * 80)
    print("  RESUMO FINAL")
    print("=" * 80)
    total_images = sum(f["count"] for f in folders.values())
    total_classes = len(class_totals)
    active_count = len(active_classes)
    total_active_imgs = sum(active_classes.values())
    dup_removable = sum(len(e) - 1 for e in duplicates.values())
    print(f"""
  Imagens totais:       {total_images}
  Imagens em classes ativas: {total_active_imgs}
  Classes totais:       {total_classes}
  Classes ativas:       {active_count} (>= {MIN_SAMPLES} imgs)
  Fontes:               {len(set(f['source'] for f in folders.values()))}
  Duplicatas removiveis: {dup_removable}
  Hashes unicos:        {len(md5_map)}
""")


if __name__ == "__main__":
    main()
