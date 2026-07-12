#!/usr/bin/env python3
"""
Run LLM-as-judge evaluations across all generator models using multiple judge models.

For each prompt log file (generator output), this script:
  - For gemini-2.5-pro judge: extracts existing call_4 results (no API calls)
  - For new judge models (sonnet-4.6, qwen3-32b, gemma-3-12b): calls _evaluate_with_llm

Output structure:
    output/judge_evaluations/<judge_model>/<generator_model>/<specialty>/case_<N>/<entry_id>_run<N>_judge.json

Each judge output file contains:
    {
        "entry_id": ...,
        "case_id": ...,
        "run_number": ...,
        "generator_model": ...,
        "judge_model": ...,
        "specialty": ...,
        "timestamp": ...,
        "evaluation": { ground_truth_similarity, clinical_plausibility, data_leakage, summary, red_flags }
    }

Usage:
    # All generators, all judges (skip gemini = no API calls for it)
    python src/scripts/run_judge_evaluations.py

    # Specific judge model only
    python src/scripts/run_judge_evaluations.py --judges eu.anthropic.claude-sonnet-4-6

    # Specific generator model and specialty
    python src/scripts/run_judge_evaluations.py --generators gemini-2.5-pro --specialties cardiology

    # Dry run (show what would be done)
    python src/scripts/run_judge_evaluations.py --dry-run
"""

import sys
import json
import re
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / ".env")

import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_BASE = project_root / "output"

GENERATOR_MODELS = [
    "gemini-2.5-pro",
    "sonnet-4.6",
    "gemma-3-12b",
    "qwen3-32b",
]

SPECIALTIES = [
    "cardiology",
    "hematology_oncology",
    "infectious_disease",
    "neurology_neurosurgery",
    "obstetrics_gynecology",
    "rheumatology",
]

GROUND_TRUTH_FILES = {
    "cardiology":             "data/ground_truth_cardiology.json",
    "hematology_oncology":    "data/ground_truth_hematology_oncology.json",
    "infectious_disease":     "data/ground_truth_infectious_disease.json",
    "neurology_neurosurgery": "data/ground_truth_neurology.json",
    "obstetrics_gynecology":  "data/ground_truth_obstetrics_gynecology.json",
    "rheumatology":           "data/ground_truth_rheumatology.json",
}

# Judge model slug → actual model ID (for API calls)
JUDGE_MODELS = {
    "gemini-2.5-pro":          None,                          # copy from existing call_4, no API
    "sonnet-4.6":              "eu.anthropic.claude-sonnet-4-6",
    "gemma-3-12b":             "google.gemma-3-12b-it",
    "qwen3-32b":               "qwen.qwen3-32b-v1:0",
}

JUDGE_OUTPUT_DIR = OUTPUT_BASE / "judge_evaluations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_text(resp) -> str:
    """Extract text from a response that may be a dict or string."""
    if isinstance(resp, dict):
        return resp.get("text", "")
    if isinstance(resp, str):
        if resp.startswith("{") and "'text':" in resp:
            m = re.search(r"'text':\s*'(.*?)(?:',\s*'usage'|'\s*})", resp, re.DOTALL)
            if m:
                text = m.group(1)
                try:
                    return text.encode().decode("unicode_escape")
                except Exception:
                    return text.replace("\\'", "'")
        return resp
    return ""


def parse_json_from_text(text: str):
    """Parse JSON from response text, stripping markdown code fences if present."""
    if "```json" in text:
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
    elif "```" in text:
        m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        return None


def load_ground_truth(specialty: str) -> dict:
    """Load GT entries keyed by entry_id."""
    gt_path = project_root / GROUND_TRUTH_FILES[specialty]
    with open(gt_path) as f:
        gt_data = json.load(f)
    if "entries" in gt_data:
        return {e["id"]: e for e in gt_data["entries"]}
    if isinstance(gt_data, list):
        return {
            f"case_{e['case_id']}_{e['category']}_{e['clinical_data_id']}": e
            for e in gt_data
        }
    return {}


def extract_generated_response(prompt_log: dict) -> str:
    """Extract the generated clinical response from call_2."""
    call_2 = prompt_log.get("prompts", {}).get("call_2", {})
    resp_text = extract_text(call_2.get("response", ""))
    parsed = parse_json_from_text(resp_text)
    if parsed:
        return parsed.get("assistant_response", "")
    return resp_text


def extract_existing_call4(prompt_log: dict) -> dict | None:
    """Extract the existing gemini judge result from call_4."""
    call_4 = prompt_log.get("prompts", {}).get("call_4", {})
    if not call_4:
        return None
    resp_text = extract_text(call_4.get("response", ""))
    if not resp_text:
        return None
    parsed = parse_json_from_text(resp_text)
    if parsed and "ground_truth_similarity" in parsed:
        return parsed
    return None


def judge_output_path(judge_slug: str, generator_slug: str, specialty: str,
                      case_id, entry_id: str, run_number: int) -> Path:
    return (
        JUDGE_OUTPUT_DIR
        / judge_slug
        / generator_slug
        / specialty
        / f"case_{case_id}"
        / f"{entry_id}_run{run_number}_judge.json"
    )


def save_judge_result(path: Path, entry_id: str, case_id, run_number: int,
                      generator_model: str, judge_slug: str, specialty: str,
                      evaluation: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "entry_id": entry_id,
        "case_id": case_id,
        "run_number": run_number,
        "generator_model": generator_model,
        "judge_model": judge_slug,
        "specialty": specialty,
        "timestamp": datetime.now().isoformat(),
        "evaluation": evaluation,
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# LLM client cache (one client per judge model)
# ---------------------------------------------------------------------------

_client_cache: dict = {}

def get_judge_client(judge_model_id: str):
    if judge_model_id in _client_cache:
        return _client_cache[judge_model_id]
    from src.llm.client import get_client_for_model
    client = get_client_for_model(judge_model_id)
    _client_cache[judge_model_id] = client
    return client


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_llm_judge(judge_model_id: str, generated: str, ground_truth: str,
                  category: str, diagnosis: str, case_text: str = None) -> dict:
    """
    Call the two separate evaluation prompts matching prompts.txt.
    Returns merged dict with ground_truth_similarity, clinical_plausibility,
    data_leakage, summary, red_flags.
    """
    from src.evaluation.metrics import ResponseEvaluator
    client = get_judge_client(judge_model_id)
    evaluator = ResponseEvaluator(llm_client=client)
    return evaluator._evaluate_with_llm(
        generated=generated,
        ground_truth=ground_truth,
        category=category,
        diagnosis=diagnosis,
        case_text=case_text,
    )


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_prompt_log(prompt_log_path: Path, generator_slug: str, specialty: str,
                       gt_entries: dict, judge_slugs: list[str], dry_run: bool) -> dict:
    """Process one prompt log file for all requested judges. Returns counts dict."""
    counts = {j: {"done": 0, "skipped": 0, "error": 0} for j in judge_slugs}

    with open(prompt_log_path) as f:
        prompt_log = json.load(f)

    entry_id    = prompt_log.get("entry_id", "")
    case_id     = prompt_log.get("case_id")
    run_number  = prompt_log.get("run_number", 1)
    gen_model   = prompt_log.get("model", generator_slug)

    # Ground truth info
    gt_entry    = gt_entries.get(entry_id, {})
    ground_truth = gt_entry.get("raw_content", gt_entry.get("ground_truth", ""))
    category    = gt_entry.get("category", "")
    diagnosis   = gt_entry.get("primary_diagnosis", gt_entry.get("diagnosis", ""))

    generated   = extract_generated_response(prompt_log)

    if not ground_truth or not generated:
        for j in judge_slugs:
            counts[j]["skipped"] += 1
        return counts

    for judge_slug in judge_slugs:
        out_path = judge_output_path(judge_slug, generator_slug, specialty,
                                     case_id, entry_id, run_number)

        # Skip if already done
        if out_path.exists():
            counts[judge_slug]["skipped"] += 1
            continue

        if dry_run:
            counts[judge_slug]["done"] += 1
            continue

        try:
            if judge_slug == "gemini-2.5-pro":
                # Copy from existing call_4
                evaluation = extract_existing_call4(prompt_log)
                if evaluation is None:
                    counts[judge_slug]["skipped"] += 1
                    continue
            else:
                judge_model_id = JUDGE_MODELS[judge_slug]
                evaluation = run_llm_judge(
                    judge_model_id, generated, ground_truth, category, diagnosis
                )

            save_judge_result(out_path, entry_id, case_id, run_number,
                              gen_model, judge_slug, specialty, evaluation)
            counts[judge_slug]["done"] += 1

        except Exception as e:
            counts[judge_slug]["error"] += 1
            print(f"\n  ⚠️  Error [{judge_slug}] {entry_id} run{run_number}: {e}")
            if os.getenv("DEBUG"):
                traceback.print_exc()

    return counts


def run_all(generator_slugs: list[str], specialties: list[str],
            judge_slugs: list[str], dry_run: bool):

    label = "[DRY RUN] " if dry_run else ""
    print(f"\n{label}Judge evaluations")
    print(f"  Generators : {generator_slugs}")
    print(f"  Specialties: {specialties}")
    print(f"  Judges     : {judge_slugs}")
    print(f"  Output     : {JUDGE_OUTPUT_DIR}\n")

    total_counts = {j: {"done": 0, "skipped": 0, "error": 0} for j in judge_slugs}

    for generator_slug in generator_slugs:
        for specialty in specialties:
            gen_spec_dir = OUTPUT_BASE / generator_slug / specialty
            if not gen_spec_dir.is_dir():
                print(f"  ⚠️  Missing: {gen_spec_dir}, skipping")
                continue

            prompt_logs = sorted(gen_spec_dir.rglob("*_prompts_*.json"))
            if not prompt_logs:
                print(f"  ⚠️  No prompt logs in {gen_spec_dir}")
                continue

            gt_entries = load_ground_truth(specialty)

            desc = f"{generator_slug}/{specialty}"
            for pf in tqdm(prompt_logs, desc=desc, ncols=100):
                counts = process_prompt_log(pf, generator_slug, specialty,
                                            gt_entries, judge_slugs, dry_run)
                for j in judge_slugs:
                    for k in ("done", "skipped", "error"):
                        total_counts[j][k] += counts[j][k]

    print(f"\n{'='*60}")
    print(f"{label}Summary:")
    for j in judge_slugs:
        c = total_counts[j]
        print(f"  {j:30s}  done={c['done']}  skipped={c['skipped']}  errors={c['error']}")
    print("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge evaluations across all generator/judge combinations"
    )
    parser.add_argument(
        "--generators", nargs="+", default=GENERATOR_MODELS,
        choices=GENERATOR_MODELS,
        help="Generator model slugs to process (default: all)"
    )
    parser.add_argument(
        "--specialties", nargs="+", default=SPECIALTIES,
        choices=SPECIALTIES,
        help="Specialties to process (default: all)"
    )
    parser.add_argument(
        "--judges", nargs="+", default=list(JUDGE_MODELS.keys()),
        choices=list(JUDGE_MODELS.keys()),
        help="Judge model slugs to use (default: all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without making API calls or writing files"
    )
    args = parser.parse_args()

    run_all(args.generators, args.specialties, args.judges, args.dry_run)


if __name__ == "__main__":
    main()
