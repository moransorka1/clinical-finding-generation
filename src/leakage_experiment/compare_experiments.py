#!/usr/bin/env python3
"""
Compare Baseline vs Leakage Experiment Results
================================================

Loads CSVs from both experiments and produces a side-by-side analysis.

Key question answered:
    If scores are similar  → model ignores context → DATA LEAKAGE
    If leakage >> baseline → model uses context    → NO LEAKAGE

Usage:
    # Auto-detect latest unified CSVs in output/ and output_leakage/
    python3 src/leakage_experiment/compare_experiments.py

    # Explicit paths
    python3 src/leakage_experiment/compare_experiments.py \\
        --baseline output/all_cases_unified_*.csv \\
        --leakage  output_leakage/all_leakage_unified_*.csv

    # Save report
    python3 src/leakage_experiment/compare_experiments.py \\
        --report output_leakage/comparison_report.md

    # Also generate charts
    python3 src/leakage_experiment/compare_experiments.py \\
        --report output_leakage/comparison_report.md --charts
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────
# Helpers (shared with analyze_results.py)
# ──────────────────────────────────────────────────────────

CATEGORY_LABELS = {
    "laboratory_tests": "Laboratory Tests",
    "imaging": "Imaging",
    "physical_examination_or_assessments": "Physical Exam / Assessments",
}


def _bool(val: str) -> bool | None:
    v = val.strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    return None


def _float(val: str) -> float | None:
    v = val.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _int(val: str) -> int | None:
    v = val.strip()
    if not v:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def safe_mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def safe_std(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def safe_median(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]


def pct(n: int, d: int) -> str:
    return f"{100*n/d:.1f}%" if d else "N/A"


def fmt(v: float | None, dp: int = 2) -> str:
    return f"{v:.{dp}f}" if v is not None else "N/A"


def delta_str(baseline: float | None, leakage: float | None) -> str:
    if baseline is None or leakage is None:
        return "N/A"
    d = leakage - baseline
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}"


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_rows(raw: list[dict]) -> list[dict]:
    parsed = []
    for r in raw:
        cid = _int(r.get("case_id", ""))
        if cid is None:
            continue
        parsed.append({
            "entry_id": r.get("entry_id", "").strip(),
            "case_id": cid,
            "clinical_data_id": r.get("clinical_data_id", "").strip(),
            "run_number": _int(r.get("run_number", "")),
            "model": r.get("model", "").strip() or "unknown",
            "category": r.get("category", "").strip(),
            "success": _bool(r.get("success", "")),
            "exact_match": _bool(r.get("exact_match", "")),
            "contains_ground_truth": _bool(r.get("contains_ground_truth", "")),
            "data_leakage_detected": _bool(r.get("data_leakage_detected", "")),
            "llm_gt_similarity": _float(r.get("llm_ground_truth_similarity", "")),
            "llm_clinical_plausibility": _float(r.get("llm_clinical_plausibility", "")),
            "numeric_matches_count": _int(r.get("numeric_matches_count", "")),
            "numeric_values_count_gt": _int(r.get("numeric_values_count_gt", "")),
            "mean_numeric_distance_pct": _float(r.get("mean_numeric_distance_percent", "")),
            "imaging_location_overlap": _bool(r.get("imaging_location_overlap", "")),
            "llm_red_flags": r.get("llm_red_flags", "").strip(),
        })
    return parsed


# ──────────────────────────────────────────────────────────
# Find CSVs
# ──────────────────────────────────────────────────────────

def find_latest(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(directory.glob(pattern))
    return candidates[-1] if candidates else None


# ──────────────────────────────────────────────────────────
# Metric extraction
# ──────────────────────────────────────────────────────────

def compute_metrics(rows: list[dict], label: str) -> dict[str, Any]:
    n = len(rows)
    succ = [r for r in rows if r["success"] is not None]
    n_succ = sum(1 for r in succ if r["success"])

    gt_sims = [r["llm_gt_similarity"] for r in rows if r["llm_gt_similarity"] is not None]
    cps = [r["llm_clinical_plausibility"] for r in rows if r["llm_clinical_plausibility"] is not None]

    exact = [r for r in rows if r["exact_match"] is not None]
    n_exact = sum(1 for r in exact if r["exact_match"])
    cont = [r for r in rows if r["contains_ground_truth"] is not None]
    n_cont = sum(1 for r in cont if r["contains_ground_truth"])
    leak = [r for r in rows if r["data_leakage_detected"] is not None]
    n_leak = sum(1 for r in leak if r["data_leakage_detected"])
    n_rf = sum(1 for r in rows if r["llm_red_flags"])

    num_rows = [r for r in rows if r["numeric_values_count_gt"] and r["numeric_values_count_gt"] > 0]
    total_gt_nums = sum(r["numeric_values_count_gt"] for r in num_rows if r["numeric_values_count_gt"])
    total_matches = sum(r["numeric_matches_count"] for r in num_rows if r["numeric_matches_count"] is not None)
    dists = [r["mean_numeric_distance_pct"] for r in num_rows if r["mean_numeric_distance_pct"] is not None]

    img_rows = [r for r in rows if r["category"] == "imaging"]
    loc = [r for r in img_rows if r["imaging_location_overlap"] is not None]
    n_loc = sum(1 for r in loc if r["imaging_location_overlap"])

    return {
        "label": label,
        "rows": n,
        "success_rate": n_succ / len(succ) * 100 if succ else None,
        "gt_sim_mean": safe_mean(gt_sims),
        "gt_sim_std": safe_std(gt_sims),
        "gt_sim_median": safe_median(gt_sims),
        "cp_mean": safe_mean(cps),
        "cp_std": safe_std(cps),
        "exact_match_pct": n_exact / len(exact) * 100 if exact else None,
        "contains_gt_pct": n_cont / len(cont) * 100 if cont else None,
        "leakage_pct": n_leak / len(leak) * 100 if leak else None,
        "red_flag_pct": n_rf / n * 100 if n else None,
        "numeric_match_pct": total_matches / total_gt_nums * 100 if total_gt_nums else None,
        "numeric_dist_mean": safe_mean(dists),
        "numeric_dist_median": safe_median(dists),
        "img_location_overlap_pct": n_loc / len(loc) * 100 if loc else None,
        # per-category
        "_gt_sims_by_cat": {
            cat: [r["llm_gt_similarity"] for r in rows if r["category"] == cat and r["llm_gt_similarity"] is not None]
            for cat in sorted({r["category"] for r in rows})
        },
        "_cp_by_cat": {
            cat: [r["llm_clinical_plausibility"] for r in rows if r["category"] == cat and r["llm_clinical_plausibility"] is not None]
            for cat in sorted({r["category"] for r in rows})
        },
        # per-case
        "_gt_sims_by_case": {
            cid: [r["llm_gt_similarity"] for r in rows if r["case_id"] == cid and r["llm_gt_similarity"] is not None]
            for cid in sorted({r["case_id"] for r in rows})
        },
        # score dist (1-5)
        "_gt_sim_dist": {
            s: sum(1 for v in gt_sims if round(v) == s)
            for s in range(1, 6)
        },
    }


# ──────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────

def build_report(b: dict, l: dict, md: bool = False) -> str:
    lines: list[str] = []

    def h1(t):
        lines.append(f"\n# {t}\n" if md else f"\n{'='*60}\n  {t}\n{'='*60}\n")

    def h2(t):
        lines.append(f"\n## {t}\n" if md else f"\n── {t} ──\n")

    def kv(k, v, indent=2):
        pad = " " * indent
        lines.append(f"- **{k}**: {v}" if md else f"{pad}{k:<50} {v}")

    def tbl(headers, data, widths=None):
        if not data:
            lines.append("  (no data)")
            return
        if widths is None:
            widths = [max(len(str(h)), max((len(str(r[i])) for r in data), default=0)) for i, h in enumerate(headers)]
        if md:
            lines.append("| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |")
            lines.append("| " + " | ".join("-" * w for w in widths) + " |")
            for row in data:
                lines.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |")
        else:
            lines.append("  " + "  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
            lines.append("  " + "  ".join("-" * w for w in widths))
            for row in data:
                lines.append("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, widths)))

    # ── Title ──
    h1("Baseline vs Leakage Experiment Comparison")

    h2("1. Dataset Size")
    kv("Baseline rows", b["rows"])
    kv("Leakage  rows", l["rows"])

    # ── Side-by-side table ──
    h2("2. Overall Metrics — Side by Side")

    metrics = [
        ("Success rate (%)", "success_rate"),
        ("GT Similarity (mean)", "gt_sim_mean"),
        ("GT Similarity (std)", "gt_sim_std"),
        ("GT Similarity (median)", "gt_sim_median"),
        ("Clinical Plausibility (mean)", "cp_mean"),
        ("Exact match (%)", "exact_match_pct"),
        ("Contains GT (%)", "contains_gt_pct"),
        ("Data leakage detected (%)", "leakage_pct"),
        ("Red flags (%)", "red_flag_pct"),
        ("Numeric match (%)", "numeric_match_pct"),
        ("Numeric distance mean (%)", "numeric_dist_mean"),
        ("Numeric distance median (%)", "numeric_dist_median"),
        ("Imaging location overlap (%)", "img_location_overlap_pct"),
    ]

    headers = ["Metric", "Baseline", "Leakage", "Δ (Leak-Base)", "Signal"]
    data = []
    for label, key in metrics:
        bv = b.get(key)
        lv = l.get(key)
        d = delta_str(bv, lv)

        # Interpret signal
        if bv is not None and lv is not None:
            diff = lv - bv
            if key == "gt_sim_mean":
                # The KEY metric — if Δ > 0.5 → good (model uses context)
                if diff > 0.5:
                    sig = "✅ model uses context"
                elif diff > 0.2:
                    sig = "⚠️  weak signal"
                else:
                    sig = "🚨 POSSIBLE LEAKAGE"
            elif key in ("leakage_pct", "red_flag_pct", "numeric_dist_mean"):
                sig = ""  # lower is better — interpretation varies
            else:
                sig = ""
        else:
            sig = ""

        data.append([label, fmt(bv), fmt(lv), d, sig])

    tbl(headers, data)

    # ── Score distributions ──
    h2("3. GT Similarity Score Distribution (1–5)")
    headers = ["Score", "Baseline", "Base %", "Leakage", "Leak %", "Δ%"]
    data = []
    for s in range(1, 6):
        bc = b["_gt_sim_dist"].get(s, 0)
        lc = l["_gt_sim_dist"].get(s, 0)
        bn = sum(b["_gt_sim_dist"].values()) or 1
        ln = sum(l["_gt_sim_dist"].values()) or 1
        bp = bc / bn * 100
        lp = lc / ln * 100
        data.append([s, bc, f"{bp:.1f}%", lc, f"{lp:.1f}%", f"{lp-bp:+.1f}%"])
    tbl(headers, data)

    # ── Per-category ──
    h2("4. Per-Category Comparison — GT Similarity Mean")
    all_cats = sorted(set(list(b["_gt_sims_by_cat"].keys()) + list(l["_gt_sims_by_cat"].keys())))
    headers = ["Category", "Base μ", "Leak μ", "Δ", "Signal"]
    data = []
    for cat in all_cats:
        bv = safe_mean(b["_gt_sims_by_cat"].get(cat, []))
        lv = safe_mean(l["_gt_sims_by_cat"].get(cat, []))
        d = delta_str(bv, lv)
        if bv is not None and lv is not None:
            diff = lv - bv
            sig = "✅ uses context" if diff > 0.5 else ("⚠️  weak" if diff > 0.2 else "🚨 LEAKAGE?")
        else:
            sig = ""
        data.append([CATEGORY_LABELS.get(cat, cat), fmt(bv), fmt(lv), d, sig])
    tbl(headers, data)

    # ── Per-case ──
    h2("5. Per-Case Comparison — GT Similarity Mean")
    all_cases = sorted(set(list(b["_gt_sims_by_case"].keys()) + list(l["_gt_sims_by_case"].keys())))
    headers = ["Case", "Base μ", "Leak μ", "Δ", "Signal"]
    data = []
    for cid in all_cases:
        bv = safe_mean(b["_gt_sims_by_case"].get(cid, []))
        lv = safe_mean(l["_gt_sims_by_case"].get(cid, []))
        d = delta_str(bv, lv)
        if bv is not None and lv is not None:
            diff = lv - bv
            sig = "✅" if diff > 0.5 else ("⚠️" if diff > 0.2 else "🚨")
        else:
            sig = ""
        data.append([f"case_{cid}", fmt(bv), fmt(lv), d, sig])
    tbl(headers, data)

    # ── Verdict ──
    h2("6. Verdict")
    bm = b["gt_sim_mean"]
    lm = l["gt_sim_mean"]
    if bm is not None and lm is not None:
        delta = lm - bm
        kv("Baseline GT Similarity mean", fmt(bm))
        kv("Leakage  GT Similarity mean", fmt(lm))
        kv("Delta (Leakage − Baseline)", f"{delta:+.3f}")
        lines.append("")
        if delta > 0.5:
            lines.append("  ✅ VERDICT: The synthesiser USES context — scores rise significantly")
            lines.append("     when the ground-truth test is present.  No evidence of data leakage.")
        elif delta > 0.2:
            lines.append("  ⚠️  VERDICT: Weak signal — scores rise slightly.  Context may be")
            lines.append("     partially used.  Some leakage is possible.  Investigate per-category.")
        else:
            lines.append("  🚨 VERDICT: Scores are essentially the same whether the data is in the")
            lines.append("     context or not.  This suggests the model may be generating from")
            lines.append("     memorised patterns rather than context — POSSIBLE DATA LEAKAGE.")
    else:
        lines.append("  Insufficient data to render a verdict.")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────────────────

def generate_comparison_charts(b_rows: list[dict], l_rows: list[dict], charts_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib/numpy not installed — skipping charts.", file=sys.stderr)
        return []

    charts_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    cats = sorted(set(r["category"] for r in b_rows + l_rows))
    cat_labels = [CATEGORY_LABELS.get(c, c) for c in cats]

    # ── Chart 1: GT Similarity — Baseline vs Leakage per category (grouped bar) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cats))
    w = 0.35
    b_means = [safe_mean([r["llm_gt_similarity"] for r in b_rows if r["category"] == c and r["llm_gt_similarity"] is not None]) or 0 for c in cats]
    l_means = [safe_mean([r["llm_gt_similarity"] for r in l_rows if r["category"] == c and r["llm_gt_similarity"] is not None]) or 0 for c in cats]
    bars1 = ax.bar(x - w / 2, b_means, w, label="Baseline (excluded)", color="#4C8BF5", alpha=0.8)
    bars2 = ax.bar(x + w / 2, l_means, w, label="Leakage (included)", color="#34A853", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=15, ha="right")
    ax.set_ylabel("Mean GT Similarity (1–5)")
    ax.set_ylim(0, 5.5)
    ax.set_title("GT Similarity — Baseline vs Leakage by Category", fontweight="bold")
    ax.legend()
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    p = charts_dir / "01_gt_similarity_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # ── Chart 2: Score distributions side by side (histogram) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    b_gts = [r["llm_gt_similarity"] for r in b_rows if r["llm_gt_similarity"] is not None]
    l_gts = [r["llm_gt_similarity"] for r in l_rows if r["llm_gt_similarity"] is not None]

    for ax, vals, title, color in [
        (axes[0], b_gts, "Baseline (test excluded)", "#4C8BF5"),
        (axes[1], l_gts, "Leakage (test included)", "#34A853"),
    ]:
        bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        counts, _, patches = ax.hist(vals, bins=bins, color=color, edgecolor="white", alpha=0.8, rwidth=0.85)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("GT Similarity Score")
        ax.set_ylabel("Count")
        ax.set_title(title, fontweight="bold")
        m = safe_mean(vals) if vals else 0
        ax.axvline(m, color="red", linestyle="--", label=f"Mean={m:.2f}")
        ax.legend()
    fig.suptitle("GT Similarity Score Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = charts_dir / "02_score_distributions.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    saved.append(p)

    # ── Chart 3: Per-case delta (leakage − baseline) ──
    b_by_case = defaultdict(list)
    l_by_case = defaultdict(list)
    for r in b_rows:
        if r["llm_gt_similarity"] is not None:
            b_by_case[r["case_id"]].append(r["llm_gt_similarity"])
    for r in l_rows:
        if r["llm_gt_similarity"] is not None:
            l_by_case[r["case_id"]].append(r["llm_gt_similarity"])

    common = sorted(set(b_by_case.keys()) & set(l_by_case.keys()))
    if common:
        deltas = [(safe_mean(l_by_case[c]) or 0) - (safe_mean(b_by_case[c]) or 0) for c in common]
        colors = ["#34A853" if d > 0.5 else "#FBB907" if d > 0.2 else "#EA4335" for d in deltas]

        fig, ax = plt.subplots(figsize=(max(10, len(common) * 0.5), 5))
        ax.bar(range(len(common)), deltas, color=colors, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(0.5, color="green", linestyle="--", alpha=0.5, label="Threshold (0.5)")
        ax.set_xticks(range(len(common)))
        ax.set_xticklabels([f"c{c}" for c in common], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Δ GT Similarity (Leakage − Baseline)")
        ax.set_title("Per-Case Delta — Does Adding Data Improve Scores?", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        p = charts_dir / "03_per_case_delta.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    return saved


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs leakage experiment results.")
    parser.add_argument("--baseline", default=None, help="Baseline unified CSV path")
    parser.add_argument("--leakage", default=None, help="Leakage unified CSV path")
    parser.add_argument("--report", default=None, help="Save report to file (.md for markdown)")
    parser.add_argument("--charts", action="store_true", help="Generate comparison charts")
    parser.add_argument("--charts-dir", default="output_leakage/charts", help="Chart output directory")
    args = parser.parse_args()

    # ── Find baseline CSV ──
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.is_absolute():
            baseline_path = PROJECT_ROOT / baseline_path
    else:
        baseline_path = find_latest(PROJECT_ROOT / "output", "all_cases_unified_*.csv")
        if not baseline_path:
            print("ERROR: No baseline unified CSV found. Run merge_case_csvs.py first.", file=sys.stderr)
            sys.exit(1)

    # ── Find leakage CSV ──
    if args.leakage:
        leakage_path = Path(args.leakage)
        if not leakage_path.is_absolute():
            leakage_path = PROJECT_ROOT / leakage_path
    else:
        leakage_path = find_latest(PROJECT_ROOT / "output_leakage", "all_leakage_unified_*.csv")
        if not leakage_path:
            print("ERROR: No leakage unified CSV found. Run merge_leakage_csvs.py first.", file=sys.stderr)
            sys.exit(1)

    print(f"Baseline : {baseline_path.relative_to(PROJECT_ROOT)}")
    print(f"Leakage  : {leakage_path.relative_to(PROJECT_ROOT)}")

    b_raw = load_csv(baseline_path)
    l_raw = load_csv(leakage_path)
    print(f"  Baseline rows: {len(b_raw):,}")
    print(f"  Leakage  rows: {len(l_raw):,}")

    b_rows = parse_rows(b_raw)
    l_rows = parse_rows(l_raw)

    b_metrics = compute_metrics(b_rows, "Baseline")
    l_metrics = compute_metrics(l_rows, "Leakage")

    is_md = bool(args.report and args.report.endswith(".md"))
    report = build_report(b_metrics, l_metrics, md=is_md)

    print(report)

    if args.report:
        rp = Path(args.report)
        if not rp.is_absolute():
            rp = PROJECT_ROOT / rp
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(report, encoding="utf-8")
        print(f"\n✓ Report saved → {rp.relative_to(PROJECT_ROOT)}")

    if args.charts:
        cd = Path(args.charts_dir)
        if not cd.is_absolute():
            cd = PROJECT_ROOT / cd
        print(f"\nGenerating charts → {cd.relative_to(PROJECT_ROOT)} …")
        saved = generate_comparison_charts(b_rows, l_rows, cd)
        for p in saved:
            print(f"  ✓ {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
