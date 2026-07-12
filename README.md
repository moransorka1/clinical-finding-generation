# Clinical Finding Generation — Reproduction Code

> Code for reproducing the experiments in:
>
> **"Clinical text from large language models describes diseases, not patients"**
> Sorka M, Shalat A, Abu Husei R, Goldstein A, Aran D, Shelly S.

---

## Overview

This repository implements the full experimental pipeline described in the paper. Four LLMs were given the complete clinical context of 127 published cases across six specialties and asked to generate one withheld clinical finding per entry (1,471 findings × 5 runs = 7,355 outputs per model, 29,388 total). Every output was then evaluated on two independent scales — ground-truth similarity and clinical plausibility — by all four models acting as judges (117,251 evaluations).

The repository reproduces three complementary results:

| Component | Script | Purpose |
|-----------|--------|---------|
| Baseline generation | `src/scripts/run_full_experiment.py` | Generate withheld findings; context **excludes** the target test |
| Leakage control | `src/leakage_experiment/run_leakage_experiment.py` | Positive control; context **includes** the target test (validates that the GT-similarity metric detects context use) |
| Multi-judge evaluation | `src/scripts/run_judge_evaluations.py` | Re-evaluate every output with all 4 judge models using separate similarity and plausibility prompts |

---

## Repository Structure

```
clinical-finding-generation/
├── src/
│   ├── app_code/
│   │   └── synthetic_response_generator.py   # Stage 1 (generate) + Stage 2 (sanitize)
│   ├── evaluation/
│   │   └── metrics.py                        # GT similarity, plausibility, numeric comparison
│   ├── extraction/
│   │   ├── case_reconstructor.py             # Builds ablated case context (excludes target test)
│   │   ├── test_extractor.py                 # Extracts test name from ground-truth entry
│   │   └── context_ablator.py                # Legacy text-ablation fallback
│   ├── leakage_experiment/
│   │   ├── run_leakage_experiment.py         # Positive-control pipeline
│   │   ├── merge_leakage_csvs.py             # Merge per-case CSVs
│   │   └── compare_experiments.py            # Side-by-side baseline vs leakage comparison
│   ├── llm/
│   │   ├── client.py                         # Abstract base + model router
│   │   ├── config.py                         # Model names and defaults
│   │   ├── vertex_client.py                  # Google Vertex AI (Gemini, Gemma)
│   │   ├── bedrock_client.py                 # AWS Bedrock (Claude Sonnet 4.6, Qwen3)
│   │   └── openai_client.py / azure_client.py
│   ├── scripts/
│   │   ├── run_full_experiment.py            # Main baseline experiment runner
│   │   ├── run_judge_evaluations.py          # Multi-model judge pipeline
│   │   └── summary_to_csv.py                 # Convert JSON results to CSV
│   └── utils/
│       └── db_utils.py                       # SQLAlchemy session + case helpers
└── README.md
```

---

## Pipeline Architecture

The pipeline mirrors the Methods section of the paper exactly.

### Stage 1 — Generate raw findings (`_step1_generate_realistic_results`)

The generator receives:
- Confirmed diagnosis
- Full case text (with the target test result removed)
- The test/imaging/examination to perform

Output: structured JSON with raw findings (values, units, reference ranges).

### Stage 2 — Format and sanitize (`_step2_format_and_sanitize`)

An independent LLM call takes the raw findings and:
- Formats results as table (labs), list, or narrative (imaging/exam)
- Strips diagnostic conclusions ("consistent with…", "indicates…") while preserving objective findings

### Evaluation (calls 3–5 per entry)

| Call | Method | Input | Scores |
|------|--------|-------|--------|
| call_3 | Numeric extraction (LLM) | GT text + generated text | Paired numeric values for deterministic comparison |
| call_4 | `_evaluate_similarity_with_llm` | GT + generated | GT similarity (1–5) + data leakage (bool) |
| call_5 | `_evaluate_plausibility_with_llm` | Generated only (no GT) | Clinical plausibility (1–5) |

Separating calls 4 and 5 ensures the plausibility judge never sees the ground-truth finding, which is the design described in the paper.

### Leakage positive control

`run_leakage_experiment.py` uses the same pipeline but passes a full context that **includes** the target test result. The expected result is a ΔGT ≈ +1.94 over baseline (paper Extended Data Fig. 6b), confirming the metric detects context use and ruling out training-data memorization as an explanation for baseline scores.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/moransorka1/clinical-finding-generation.git
cd clinical-finding-generation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure credentials

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```dotenv
# Google Vertex AI (Gemini 2.5 Pro, Gemma3 12B)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# AWS Bedrock (Claude Sonnet 4.6, Qwen3 32B)
AWS_REGION=eu-west-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Optional model overrides
VERTEX_FLASH_MODEL=gemini-2.5-pro
```

### 3. Database

The pipeline reads case text and clinical data from a SQLite database.
Place `cliniclue.db` in `db/` (not included in this repository — see Data Availability).

---

## Running the Experiments

### Baseline generation (reproduces paper Table 1 / Figs 2–5)

```bash
# Single case, 5 repetitions, Gemini 2.5 Pro
python src/scripts/run_full_experiment.py \
    --ground-truth-file data/ground_truth_neurology.json \
    --case-ids 1 --repetitions 5 --model gemini-2.5-pro

# All neurology cases, all models
for MODEL in gemini-2.5-pro eu.anthropic.claude-sonnet-4-6 google.gemma-3-12b-it qwen.qwen3-32b-v1:0; do
  python src/scripts/run_full_experiment.py \
      --ground-truth-file data/ground_truth_neurology.json \
      --repetitions 5 --model $MODEL
done
```

### Multi-judge evaluation (reproduces Fig. 4 / Extended Data Fig. 5)

```bash
# Re-evaluate all generator outputs with all 4 judge models
python src/scripts/run_judge_evaluations.py \
    --generators gemini-2.5-pro sonnet-4.6 gemma-3-12b qwen3-32b \
    --specialties neurology_neurosurgery cardiology \
    --judges gemini-2.5-pro sonnet-4.6 gemma-3-12b qwen3-32b
```

### Leakage positive control (reproduces Extended Data Fig. 6)

```bash
python src/leakage_experiment/run_leakage_experiment.py \
    --ground-truth-file data/ground_truth_neurology.json \
    --case-ids 1 2 3 4 5 --repetitions 5 --model gemini-2.5-pro

# Merge and compare against baseline
python src/leakage_experiment/merge_leakage_csvs.py
python src/leakage_experiment/compare_experiments.py \
    --report output_leakage/comparison_report.md --charts
```

---

## Prompt Templates

The exact prompts used in the paper are documented in `docs/prompts.txt` (LaTeX source for Supplementary Note 3). In code they map to:

| Prompt | Code location | Paper reference |
|--------|---------------|-----------------|
| Stage 1 generation | `synthetic_response_generator._step1_generate_realistic_results` | Supplementary Note 3, Stage 1 |
| Stage 2 sanitization | `synthetic_response_generator._step2_format_and_sanitize` | Supplementary Note 3, Stage 2 |
| GT similarity + leakage | `metrics._evaluate_similarity_with_llm` | Supplementary Note 3, Ground-truth similarity prompt |
| Clinical plausibility | `metrics._evaluate_plausibility_with_llm` | Supplementary Note 3, Clinical plausibility prompt |
| Numeric extraction | `metrics._extract_numeric_from_both_with_llm` | Supplementary Note 3 / Supplementary Note 2 |

---

## Models Used

| Model | Provider | API |
|-------|----------|-----|
| Gemini 2.5 Pro (`gemini-2.5-pro`) | Google DeepMind | Vertex AI |
| Claude Sonnet 4.6 (`eu.anthropic.claude-sonnet-4-6`) | Anthropic | AWS Bedrock |
| Gemma3 12B (`google.gemma-3-12b-it`) | Google | AWS Bedrock |
| Qwen3 32B (`qwen.qwen3-32b-v1:0`) | Alibaba | AWS Bedrock |

---

## Data Availability

The 127-case corpus, ground-truth JSON files, and the SQLite database are not included in this repository because they derive from copyrighted published case reports. The ground-truth JSON files and the database schema will be made available upon reasonable request to the corresponding author (s_shelly@rambam.health.gov.il), subject to any applicable data-sharing agreements.

---

## Citation

If you use this code, please cite:

```bibtex
@unpublished{sorka2025clinical,
  title  = {Clinical text from large language models describes diseases, not patients},
  author = {Sorka, Moran and Shalat, Adham and Abu Husei, Ram and
            Goldstein, Ariel and Aran, Dvir and Shelly, Shahar},
}
```

---

## License

MIT
