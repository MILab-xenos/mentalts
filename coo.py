import glob
import csv
import os
import re
import json
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

CACHE_FILE = "accuracy_cache.json"
RESULT_CSV = "classification_summary.csv"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)
else:
    cache = {}

TARGET_PREFIXES = {
    "dep_cls": "depression",
    "anx_cls": "anxiety",
    "sui_cls": "suiside",
    "ovr_cls": "overall",
}

MODEL_NAMES = [
    "Autoformer", "Crossformer", "DLinear", "FEDformer", "FiLM",
    "Informer", "iTransformer", "LightTS", "PatchTST", "Pyraformer",
    "Reformer", "TimesNet", "Transformer", "Linearformer", "Performer",
]

def parse_setting(setting):
    target = "unknown"
    for prefix, name in TARGET_PREFIXES.items():
        if prefix in setting:
            target = name
            break
    model = "unknown"
    for m in MODEL_NAMES:
        if re.search(rf'_{m}_', setting):
            model = m
            break
    return target, model

def parse_metrics(text):
    m = {}
    # Overall Accuracy
    r = re.search(r'Overall Accuracy:\s*([\d.]+)', text)
    m['accuracy'] = float(r.group(1)) if r else None

    # Macro P/R/F1
    r = re.search(r'Macro Precision:\s*([\d.]+),\s*Recall:\s*([\d.]+),\s*F1:\s*([\d.]+)', text)
    if r:
        m['macro_precision'] = float(r.group(1))
        m['macro_recall']    = float(r.group(2))
        m['macro_f1']        = float(r.group(3))
    else:
        m['macro_precision'] = m['macro_recall'] = m['macro_f1'] = None

    # Weighted P/R/F1
    r = re.search(r'Weighted Precision:\s*([\d.]+),\s*Recall:\s*([\d.]+),\s*F1:\s*([\d.]+)', text)
    if r:
        m['weighted_precision'] = float(r.group(1))
        m['weighted_recall']    = float(r.group(2))
        m['weighted_f1']        = float(r.group(3))
    else:
        m['weighted_precision'] = m['weighted_recall'] = m['weighted_f1'] = None

    # Macro ROC-AUC, PR-AUC
    r = re.search(r'Macro ROC-AUC:\s*([\d.]+),\s*PR-AUC:\s*([\d.]+)', text)
    if r:
        m['macro_roc_auc'] = float(r.group(1))
        m['macro_pr_auc']  = float(r.group(2))
    else:
        m['macro_roc_auc'] = m['macro_pr_auc'] = None

    # Weighted ROC-AUC, PR-AUC
    r = re.search(r'Weighted ROC-AUC:\s*([\d.]+),\s*PR-AUC:\s*([\d.]+)', text)
    if r:
        m['weighted_roc_auc'] = float(r.group(1))
        m['weighted_pr_auc']  = float(r.group(2))
    else:
        m['weighted_roc_auc'] = m['weighted_pr_auc'] = None

    return m

def extract_with_cache(path):
    try:
        setting = os.path.basename(os.path.dirname(path))
        mtime = os.path.getmtime(path)

        if path in cache and cache[path].get("mtime") == mtime and "macro_f1" in cache[path]:
            return setting, cache[path]

        with open(path, "rb") as f:
            try:
                f.seek(-4096, os.SEEK_END)
            except OSError:
                f.seek(0)
            text = f.read().decode(errors="ignore")

        metrics = parse_metrics(text)
        if metrics['accuracy'] is None:
            return None

        metrics['mtime'] = mtime
        cache[path] = metrics
        return setting, metrics

    except Exception as e:
        print(f"Warning: parse {path} failed: {e}")
        return None


paths = glob.glob("test_results/*/result_classification.txt")
if not paths:
    print("No result_classification.txt found in test_results/.")
    exit()

print(f"Scanning {len(paths)} result files...")

# {target: {setting: metrics}}  — keep all configs, not just best-per-model
by_target = defaultdict(dict)

with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
    for result in executor.map(extract_with_cache, paths):
        if result:
            setting, metrics = result
            target, model = parse_setting(setting)
            by_target[target][setting] = {**metrics, 'model': model, 'setting': setting}

if not by_target:
    print("No valid records found.")
    exit()

METRIC_KEYS = [
    'accuracy', 'macro_f1', 'macro_precision', 'macro_recall',
    'weighted_f1', 'weighted_precision', 'weighted_recall',
    'macro_roc_auc', 'macro_pr_auc', 'weighted_roc_auc', 'weighted_pr_auc',
]

all_rows = []

for target in sorted(by_target):
    entries = by_target[target]
    n = len(entries)
    print(f"\n{'='*70}")
    print(f"Target: {target}  ({n} configs)")
    print(f"{'='*70}")
    header = f"{'Model':<20} {'Acc':>6} {'MacF1':>7} {'MacP':>7} {'MacR':>7} {'WF1':>7} {'ROCAUC':>8} {'PRAUC':>7}"
    print(header)
    print('-' * 70)

    sorted_entries = sorted(entries.values(), key=lambda x: x['accuracy'] or 0, reverse=True)
    for e in sorted_entries:
        def fmt(v): return f"{v:.4f}" if v is not None else "  N/A "
        print(f"{e['model']:<20} {fmt(e['accuracy'])} {fmt(e['macro_f1'])} {fmt(e['macro_precision'])} {fmt(e['macro_recall'])} {fmt(e['weighted_f1'])} {fmt(e['macro_roc_auc'])} {fmt(e['macro_pr_auc'])}")
        all_rows.append([target, e['model']] + [e.get(k) for k in METRIC_KEYS] + [e['setting']])

    # averages over all configs for this target
    for key in METRIC_KEYS:
        vals = [e[key] for e in entries.values() if e.get(key) is not None]
        if vals:
            avg = sum(vals) / len(vals)
            best_e = max(entries.values(), key=lambda x: x.get(key) or 0)
            print(f"  avg {key}: {avg:.4f}  |  best: {best_e.get(key):.4f} ({best_e['model']})")

with open(RESULT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Target", "Model"] + METRIC_KEYS + ["Setting"])
    writer.writerows(all_rows)

with open(CACHE_FILE, "w") as f:
    json.dump(cache, f, indent=2)

print(f"\nSaved to {RESULT_CSV} ({len(all_rows)} rows)")
