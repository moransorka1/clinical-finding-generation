# Leakage Experiment — Positive Control

Detects whether the synthetic response generator is leaking memorised data
or genuinely using the clinical context it is given.

## Concept

| Experiment | Context contains target test? | Purpose |
|------------|------------------------------|---------|
| **Baseline** (`src/scripts/run_full_experiment.py`) | ❌ Excluded | Measures inference quality when the answer is NOT in context |
| **Leakage** (`src/leakage_experiment/run_leakage_experiment.py`) | ✅ Included | Positive control — the answer IS in context |

**Interpretation of the delta (Leakage − Baseline):**

| Δ GT Similarity | Meaning |
|-----------------|---------|
| > 0.5 | ✅ Model uses context — no evidence of leakage |
| 0.2 – 0.5 | ⚠️ Weak signal — investigate per-category |
| < 0.2 | 🚨 Scores unchanged — possible data leakage |

---

## Scripts

### `run_leakage_experiment.py`

Runs the experiment with the target test result **included** in context.

```bash
# Run on 10 cases with pro model and 5 repetitions (recommended)
python3 src/leakage_experiment/run_leakage_experiment.py \
    --ground-truth-file data/ground_truth_neurology.json \
    --case-ids 1 2 3 4 5 6 7 8 9 10 \
    --repetitions 5 \
    --model gemini-2.5-pro

# Run all neurology cases
python3 src/leakage_experiment/run_leakage_experiment.py \
    --ground-truth-file data/ground_truth_neurology.json \
    --repetitions 5 \
    --model gemini-2.5-pro

# Single case quick test
python3 src/leakage_experiment/run_leakage_experiment.py \
    --ground-truth-file data/ground_truth_neurology.json \
    --case-ids 6 --repetitions 1 --model gemini-2.5-pro
```

#### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--case-ids` | all | Case ID(s) to process |
| `--repetitions` | 1 | Runs per test (use 5 for consistency) |
| `--model` | `gemini-2.5-flash` | **Use `gemini-2.5-pro` for final runs** |
| `--ground-truth-file` | `data/ground_truth_neurology.json` | Ground truth JSON |
| `--limit` | None | Max entries (when `--case-ids` not used) |
| `--start-from` | 0 | Index offset for resuming |

#### Output

```
output_leakage/
├── leakage_experiment_<ts>.json        # Full results with prompts
├── leakage_summary_<ts>.json           # Summary without prompts
└── case_<id>/
    ├── case_<id>_leakage_summary_<ts>.json
    ├── case_<id>_leakage_summary_<ts>.csv
    └── <entry>_run<N>_prompts_<ts>.json
```

---

### `merge_leakage_csvs.py`

Merges per-case leakage CSVs into a single unified CSV.

```bash
python3 src/leakage_experiment/merge_leakage_csvs.py
python3 src/leakage_experiment/merge_leakage_csvs.py --case-ids 1 2 3
```

---

### `compare_experiments.py`

Side-by-side comparison of baseline vs leakage results.

```bash
# Auto-detect latest unified CSVs
python3 src/leakage_experiment/compare_experiments.py

# Save markdown report + charts
python3 src/leakage_experiment/compare_experiments.py \
    --report output_leakage/comparison_report.md --charts

# Explicit CSV paths
python3 src/leakage_experiment/compare_experiments.py \
    --baseline output/all_cases_unified_20260312_074738.csv \
    --leakage  output_leakage/all_leakage_unified_*.csv
```

#### Report sections

| # | Section |
|---|---------|
| 1 | Dataset Size |
| 2 | Overall Metrics — Side by Side (with Δ and signal) |
| 3 | GT Similarity Score Distribution (1–5) |
| 4 | Per-Category Comparison |
| 5 | Per-Case Comparison |
| 6 | Verdict (✅ / ⚠️ / 🚨) |

#### Charts (`--charts`)

| File | Description |
|------|-------------|
| `01_gt_similarity_comparison.png` | Grouped bar — baseline vs leakage by category |
| `02_score_distributions.png` | Side-by-side histograms of GT similarity |
| `03_per_case_delta.png` | Per-case Δ bar chart (green/yellow/red) |

---

## Full Workflow

```bash
# 1. Run leakage experiment (use pro model!)
python3 src/leakage_experiment/run_leakage_experiment.py \
    --ground-truth-file data/ground_truth_neurology.json \
    --case-ids 1 2 3 4 5 6 7 8 9 10 \
    --repetitions 5 --model gemini-2.5-pro

# 2. Merge CSVs
python3 src/leakage_experiment/merge_leakage_csvs.py

# 3. Compare with baseline
python3 src/leakage_experiment/compare_experiments.py \
    --report output_leakage/comparison_report.md --charts
```
