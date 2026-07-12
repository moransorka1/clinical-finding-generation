#!/usr/bin/env python3
"""
Merge leakage experiment per-case CSVs into a single unified CSV.

Identical to src/scripts/merge_case_csvs.py but defaults to
output_leakage/ as the source and writes all_leakage_unified_<ts>.csv.

Usage:
    python3 src/leakage_experiment/merge_leakage_csvs.py
    python3 src/leakage_experiment/merge_leakage_csvs.py --case-ids 1 9 13
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def find_case_csvs(cases_dir: Path, case_ids: list[int] | None, latest_only: bool) -> dict[int, list[Path]]:
    pattern = re.compile(r"^case_(\d+)$")
    result: dict[int, list[Path]] = {}
    for subdir in sorted(cases_dir.iterdir()):
        m = pattern.match(subdir.name)
        if not m:
            continue
        cid = int(m.group(1))
        if case_ids and cid not in case_ids:
            continue
        csvs = sorted(subdir.glob("*_leakage_summary_*.csv"))
        if not csvs:
            # Fall back to any summary CSV
            csvs = sorted(subdir.glob("*_summary_*.csv"))
        if not csvs:
            print(f"  [WARN] No CSV in {subdir}", file=sys.stderr)
            continue
        result[cid] = [csvs[-1]] if latest_only else csvs
    return result


def load_and_tag(csv_path: Path, case_id: int) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row["case_id"] = str(case_id)
            rows.append(row)
    return rows


def deduplicate(rows: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for row in rows:
        key = (row.get("entry_id", ""), row.get("case_id", ""), row.get("run_number", ""), row.get("model", ""))
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


COLUMNS = [
    "entry_id", "case_id", "clinical_data_id", "run_number", "model",
    "category", "diagnosis", "timestamp", "success", "user_request",
    "extracted_test_name", "ground_truth", "generated_response",
    "exact_match", "contains_ground_truth", "data_leakage_detected",
    "numeric_values_count_gt", "numeric_values_count_generated",
    "numeric_matches_count", "mean_numeric_distance_percent",
    "numeric_comparison_detailed",
    "imaging_location_overlap", "imaging_measurement_exact_match",
    "llm_ground_truth_similarity", "llm_ground_truth_similarity_reasoning",
    "llm_clinical_plausibility", "llm_clinical_plausibility_reasoning",
    "llm_summary", "llm_red_flags",
]


def main():
    parser = argparse.ArgumentParser(description="Merge leakage experiment CSVs.")
    parser.add_argument("--cases-dir", default="output_leakage", help="Root dir with case_N/ subdirs")
    parser.add_argument("--output-dir", default="output_leakage", help="Where to write unified CSV")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--case-ids", type=int, nargs="+", default=None)
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    args = parser.parse_args()

    cases_dir = PROJECT_ROOT / args.cases_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {cases_dir} …")
    case_map = find_case_csvs(cases_dir, args.case_ids, not args.all_runs)
    if not case_map:
        print("ERROR: No CSVs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(case_map)} case(s): {sorted(case_map.keys())}\n")

    all_rows: list[dict] = []
    for cid in sorted(case_map):
        for p in case_map[cid]:
            rows = load_and_tag(p, cid)
            all_rows.extend(rows)
            print(f"  case {cid:>4}  {len(rows):>5} rows  ← {p.name}")

    print(f"\nTotal rows before dedup: {len(all_rows):,}")
    if not args.no_dedup:
        all_rows = deduplicate(all_rows)
        print(f"Total rows after  dedup: {len(all_rows):,}")

    out_name = args.output_file or f"all_leakage_unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if not out_name.endswith(".csv"):
        out_name += ".csv"
    out_path = output_dir / out_name

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✓ Unified leakage CSV → {out_path.relative_to(PROJECT_ROOT)}")
    print(f"  Cases: {len(case_map)}  |  Rows: {len(all_rows):,}")


if __name__ == "__main__":
    main()
