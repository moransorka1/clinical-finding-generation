# Clinical Finding Generation: Reproduction Code

> Code for the experiments in:
>
> **"Clinical plausibility does not establish patient fidelity in generated clinical findings"**
> Sorka M, Shalata A, Husein RA, Goldstein A, Aran D, Shelly S.

---

## Overview

This repository implements the generation and evaluation pipeline described in the paper.
Four LLMs were given the clinical context of 127 published cases across six specialties and
asked to generate one withheld clinical finding per entry. With 1,471 entries, five runs per
entry and four generators this gives 29,420 generation attempts, of which 29,388 produced a
valid output. Every output was then scored on two separate scales, ground-truth similarity
and clinical plausibility, by all four models acting as judges, yielding 117,251 valid
evaluations.

| Component | Script |
|-----------|--------|
| Baseline generation; context **excludes** the target finding | `src/scripts/run_full_experiment.py` |
| Answer-present control; context **includes** the target finding | `src/leakage_experiment/run_leakage_experiment.py` |
| Multi-judge evaluation with all four judges | `src/scripts/run_judge_evaluations.py` |

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
│   │   ├── case_reconstructor.py             # Builds ablated case context (excludes target finding)
│   │   ├── test_extractor.py                 # Extracts test name from ground-truth entry
│   │   └── context_ablator.py                # Legacy text-ablation fallback
│   ├── leakage_experiment/
│   │   ├── run_leakage_experiment.py         # Answer-present control pipeline
│   │   ├── merge_leakage_csvs.py             # Merge per-case CSVs
│   │   └── compare_experiments.py            # Side-by-side baseline vs answer-present comparison
│   ├── llm/
│   │   ├── client.py                         # Abstract base + model router
│   │   ├── config.py                         # Model names and defaults
│   │   ├── vertex_client.py                  # Google Vertex AI (Gemini 2.5 Pro)
│   │   ├── bedrock_client.py                 # AWS Bedrock (Claude Sonnet 4.6, Gemma3 12B, Qwen3 32B)
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

## Pipeline

### Stage 1, generate raw findings (`_step1_generate_realistic_results`)

The generator receives the confirmed diagnosis, the case text with the target finding
removed, and the test, imaging study or examination to perform. Output is structured JSON
with raw findings (values, units, reference ranges).

### Stage 2, format and sanitize (`_step2_format_and_sanitize`)

An independent LLM call formats the raw findings as a table (laboratory), a list, or
narrative text (imaging, examination), and strips diagnostic conclusions
("consistent with...", "indicates...") while preserving the objective findings.

### Evaluation (calls 3 to 5 per entry)

| Call | Method | Input | Scores |
|------|--------|-------|--------|
| call_3 | `metrics._extract_numeric_from_both_with_llm` | documented + generated | Paired numeric values for deterministic comparison |
| call_4 | `metrics._evaluate_similarity_with_llm` | documented + generated | GT similarity (1 to 5) + leakage flag |
| call_5 | `metrics._evaluate_plausibility_with_llm` | generated only | Clinical plausibility (1 to 5) |

Separating calls 4 and 5 keeps the plausibility judge from seeing the documented finding,
which is the design described in the paper.

### Answer-present control

`run_leakage_experiment.py` uses the same pipeline but supplies a context that **includes**
the target finding. Across 41 entries this raised GT similarity by +1.94 points
(95% CI +1.50 to +2.38), showing that the pipeline responds to patient information supplied
in context. It is not a test of training-data memorization, and it does not exclude
memorization of published cases during pretraining.

---

## Setup

```bash
git clone https://github.com/moransorka1/clinical-finding-generation.git
cd clinical-finding-generation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in your credentials
```

```dotenv
# Google Vertex AI (Gemini 2.5 Pro)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# AWS Bedrock (Claude Sonnet 4.6, Gemma3 12B, Qwen3 32B)
AWS_REGION=eu-west-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

The pipeline reads case text and clinical data from a SQLite database and from
`data/ground_truth_<specialty>.json` files. Neither is included here, because both contain
text from copyrighted published case reports. See Data Availability. The commands below
therefore document the pipeline rather than run out of the box.

---

## Running

```bash
# Baseline generation: single case, 5 repetitions, Gemini 2.5 Pro
python src/scripts/run_full_experiment.py \
    --ground-truth-file data/ground_truth_neurology.json \
    --case-ids 1 --repetitions 5 --model gemini-2.5-pro

# Baseline generation: all cases in a specialty, all four models
for MODEL in gemini-2.5-pro eu.anthropic.claude-sonnet-4-6 google.gemma-3-12b-it qwen.qwen3-32b-v1:0; do
  python src/scripts/run_full_experiment.py \
      --ground-truth-file data/ground_truth_neurology.json \
      --repetitions 5 --model $MODEL
done

# Multi-judge evaluation
python src/scripts/run_judge_evaluations.py \
    --generators gemini-2.5-pro sonnet-4.6 gemma-3-12b qwen3-32b \
    --specialties neurology_neurosurgery cardiology \
    --judges gemini-2.5-pro sonnet-4.6 gemma-3-12b qwen3-32b

# Answer-present control
python src/leakage_experiment/run_leakage_experiment.py \
    --ground-truth-file data/ground_truth_neurology.json \
    --case-ids 1 2 3 4 5 --repetitions 5 --model gemini-2.5-pro
python src/leakage_experiment/merge_leakage_csvs.py
python src/leakage_experiment/compare_experiments.py \
    --report output_leakage/comparison_report.md --charts
```

---

## Prompts

All five prompts are reproduced verbatim in Supplementary Note 3 of the paper. In code they
map to:

| Prompt | Code location |
|--------|---------------|
| Stage 1 generation | `synthetic_response_generator._step1_generate_realistic_results` |
| Stage 2 sanitization | `synthetic_response_generator._step2_format_and_sanitize` |
| Ground-truth similarity | `metrics._evaluate_similarity_with_llm` |
| Clinical plausibility | `metrics._evaluate_plausibility_with_llm` |
| Numeric extraction | `metrics._extract_numeric_from_both_with_llm` |

Decoding parameters and per-stage call counts are in Supplementary Note 2.

---

## Models

| Model | Provider | Serving platform |
|-------|----------|------------------|
| Gemini 2.5 Pro (`gemini-2.5-pro`) | Google DeepMind | Google Vertex AI |
| Claude Sonnet 4.6 (`eu.anthropic.claude-sonnet-4-6`) | Anthropic | AWS Bedrock |
| Gemma3 12B (`google.gemma-3-12b-it`) | Google | AWS Bedrock |
| Qwen3 32B (`qwen.qwen3-32b-v1:0`) | Alibaba | AWS Bedrock |

All four served as both generator and judge. Generation used temperature 0.3.

---

## Data Availability

The score-level data underlying the reported analyses are deposited at Zenodo:
<https://doi.org/10.5281/zenodo.21165186>. This includes the 117,251 individual judge
evaluations, the per-output scores, entry-level metadata, the numeric-extraction
comparisons, the physician plausibility ratings and the control data.

The 127 source case reports are copyrighted publications and cannot be redistributed, so
the deposit contains no case-report text and neither does this repository. Full citations
for all 127 cases are given in Supplementary Data 1 and in the deposit, so the originals
can be obtained from the publishers.

---

## Citation

```bibtex
@unpublished{sorka2026clinical,
  title  = {Clinical plausibility does not establish patient fidelity in
            generated clinical findings},
  author = {Sorka, Moran and Shalata, Adham and Husein, Ram Abu and
            Goldstein, Ariel and Aran, Dvir and Shelly, Shahar},
  year   = {2026},
  note   = {Manuscript under review},
}
```

Data deposit: Sorka M, Shalata A, Husein RA, Goldstein A, Aran D, Shelly S.
Zenodo. <https://doi.org/10.5281/zenodo.21165186>

---

## License

MIT. See `LICENSE`.
