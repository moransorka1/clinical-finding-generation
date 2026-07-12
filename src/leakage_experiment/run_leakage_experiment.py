#!/usr/bin/env python3
"""
Leakage Experiment — Positive Control
======================================

This experiment is the **positive control** counterpart to the baseline
experiment (run_full_experiment.py).

Baseline:  context EXCLUDES the target test → generate → evaluate
Leakage:   context INCLUDES the target test → generate → evaluate

If the LLM is not leaking memorised data the leakage experiment scores
should be **significantly higher** than baseline because the answer is
literally in the context.  If both experiments score about the same the
model is ignoring context and generating from memory → **data leakage**.

Usage:
    python3 src/leakage_experiment/run_leakage_experiment.py \\
        --ground-truth-file data/ground_truth_neurology.json \\
        --case-ids 1 9 --repetitions 5

    python3 src/leakage_experiment/run_leakage_experiment.py \\
        --ground-truth-file data/ground_truth_neurology.json \\
        --case-ids 1 2 3 --model gemini-2.5-pro
"""

import sys
import json
import os
import asyncio
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# ── paths ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from src.utils.db_utils import get_sqlalchemy_session
from src.app_code.synthetic_response_generator import SyntheticResponseGenerator
from src.llm.vertex_client import VertexAIClient
from src.extraction.case_reconstructor import CaseReconstructor
from src.extraction.test_extractor import TestExtractor
from src.evaluation.metrics import ResponseEvaluator
from sqlalchemy import text as sql_text

# ── Output dirs ──
OUTPUT_DIR = PROJECT_ROOT / "output_leakage"


# ──────────────────────────────────────────────────────────────────────────────
# Prompt logging (identical to baseline)
# ──────────────────────────────────────────────────────────────────────────────

class PromptLoggingWrapper:
    """Wrapper that logs all prompts and responses sent to / from the LLM."""

    def __init__(self, client, log_dict):
        self.client = client
        self.log = log_dict
        self.call_count = 0

    def _serialize_response(self, response):
        if isinstance(response, str):
            return response
        if hasattr(response, "text"):
            return response.text
        return str(response)

    def text_completion(self, prompt, **kwargs):
        self.call_count += 1
        cid = f"call_{self.call_count}"
        self.log[cid] = {"prompt": prompt, "kwargs": kwargs}
        response = self.client.text_completion(prompt, **kwargs)
        self.log[cid]["response"] = self._serialize_response(response)
        return response

    def chat_completion(self, messages, **kwargs):
        self.call_count += 1
        cid = f"call_{self.call_count}"
        self.log[cid] = {"messages": messages, "kwargs": kwargs}
        response = self.client.chat_completion(messages, **kwargs)
        self.log[cid]["response"] = self._serialize_response(response)
        return response

    def __getattr__(self, name):
        return getattr(self.client, name)


# ──────────────────────────────────────────────────────────────────────────────
# Core: build full context WITH the target test included
# ──────────────────────────────────────────────────────────────────────────────

def build_full_context_with_test(reconstructor: CaseReconstructor, case_id: int) -> dict:
    """
    Build the complete case context **including every clinical data item**.

    This is the opposite of the baseline which calls
    ``reconstruct_case_without_test(…)`` to exclude the target.

    Returns the same shape dict that ``reconstruct_case_without_test`` returns
    so down-stream code can handle either transparently.
    """
    db = reconstructor.db_session

    hpi_result = db.execute(
        sql_text("SELECT hpi_raw, description, title FROM cases WHERE id = :case_id"),
        {"case_id": case_id},
    ).fetchone()

    if not hpi_result:
        raise ValueError(f"Case {case_id} not found")

    hpi = hpi_result.hpi_raw or ""
    case_description = hpi_result.description or ""
    case_title = hpi_result.title or ""

    clinical_data_results = db.execute(
        sql_text("""
            SELECT id, category, content, sequence_order
            FROM case_clinical_data
            WHERE case_id = :case_id
            ORDER BY sequence_order
        """),
        {"case_id": case_id},
    ).fetchall()

    clinical_data_by_category: dict[str, list[dict]] = {}
    total_items = 0

    for row in clinical_data_results:
        cat = row.category
        if cat not in clinical_data_by_category:
            clinical_data_by_category[cat] = []
        clinical_data_by_category[cat].append({
            "content": row.content,
            "sequence_order": row.sequence_order,
        })
        total_items += 1

    # Re-use the same narrative builder from CaseReconstructor
    full_context = reconstructor._build_narrative(
        hpi=hpi,
        case_description=case_description,
        case_title=case_title,
        clinical_data_by_category=clinical_data_by_category,
    )

    return {
        "hpi": hpi,
        "case_description": case_description,
        "case_title": case_title,
        "clinical_data": clinical_data_by_category,
        "full_context": full_context,
        "excluded_item": None,  # nothing excluded
        "stats": {
            "total_items": total_items,
            "excluded_count": 0,
            "included_count": total_items,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Process a single sample (with test IN context)
# ──────────────────────────────────────────────────────────────────────────────

def process_single_sample(
    entry: dict,
    generator: SyntheticResponseGenerator,
    evaluator: ResponseEvaluator,
    db_session,
    prompts_log: dict,
    diagnosis: str | None,
    case_context: str,
    test_name: str,
    context_meta: dict,
):
    """
    Generate a synthetic response for *entry* using a context that
    **includes** the ground-truth test result, then evaluate it.
    """
    result = {
        "entry_id": entry["id"],
        "case_id": entry["case_id"],
        "clinical_data_id": None,
        "category": entry["category"],
        "test_name": entry.get("test_name"),
        "ground_truth": entry["raw_content"],
        "diagnosis": diagnosis,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error": None,
        "experiment_type": "leakage_positive_control",
    }

    result["extracted_test_name"] = test_name
    result["context_preparation"] = context_meta

    try:
        # Temporarily replace case text in database
        case_query = db_session.execute(
            sql_text("SELECT id, text FROM cases WHERE id = :case_id"),
            {"case_id": entry["case_id"]},
        ).first()

        if not case_query:
            result["error"] = f"Case {entry['case_id']} not found"
            return result

        original_text = case_query.text

        db_session.execute(
            sql_text("UPDATE cases SET text = :context WHERE id = :case_id"),
            {"context": case_context, "case_id": entry["case_id"]},
        )
        db_session.commit()

        try:
            user_request = f"Perform {test_name}"
            result["user_request"] = user_request

            generated_response = asyncio.run(
                generator.generate_synthetic_response(
                    user_query=user_request,
                    case_id=entry["case_id"],
                    chat_history=[],
                    partial_response=None,
                    cpt_codes=None,
                )
            )

            result["generated_response"] = generated_response.get("assistant_response", "")
            result["formatting_type"] = generated_response.get("formatting_type", "")

            evaluation = evaluator.evaluate(
                generated=result["generated_response"],
                ground_truth=entry["raw_content"],
                category=entry["category"],
                diagnosis=diagnosis,
                case_text=entry["case_text"],
            )

            result["evaluation"] = evaluation
            result["success"] = True

        finally:
            db_session.execute(
                sql_text("UPDATE cases SET text = :original_text WHERE id = :case_id"),
                {"original_text": original_text, "case_id": entry["case_id"]},
            )
            db_session.commit()

    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

def run_leakage_experiment(
    case_ids: list[int] | None = None,
    limit: int | None = None,
    start_from: int = 0,
    repetitions: int = 1,
    model_name: str | None = None,
    ground_truth_file: str | None = None,
):
    print("=" * 80)
    print("LEAKAGE EXPERIMENT — Positive Control (context INCLUDES target test)")
    print("=" * 80)

    # ── Load ground truth ──
    if ground_truth_file:
        gt_path = Path(ground_truth_file)
    else:
        gt_path = PROJECT_ROOT / "data" / "ground_truth_neurology.json"

    with open(gt_path) as f:
        gt_data = json.load(f)

    if case_ids:
        entries = [e for e in gt_data["entries"] if e["case_id"] in case_ids]
    else:
        entries = gt_data["entries"][start_from: start_from + limit if limit else None]

    print(f"\n📊 Experiment Parameters:")
    print(f"  Entries         : {len(entries)}")
    print(f"  Repetitions     : {repetitions}")
    effective_model = model_name or os.getenv("VERTEX_FLASH_MODEL", "gemini-2.5-flash")
    print(f"  Model           : {effective_model}")
    print(f"  Context         : FULL (test result INCLUDED)")
    print(f"  Output dir      : {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")

    # ── Initialise components ──
    if model_name:
        os.environ["VERTEX_FLASH_MODEL"] = model_name
        os.environ["VERTEX_TEXT_MODEL"] = model_name
        os.environ["VERTEX_VISION_MODEL"] = model_name
        import src.llm.config as llm_config
        llm_config._config_cache = None

    base_llm = VertexAIClient(
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    )

    prompts_log: dict = {}
    llm_client = PromptLoggingWrapper(base_llm, prompts_log)

    db_session = get_sqlalchemy_session()
    generator = SyntheticResponseGenerator(db_session, llm_client)
    reconstructor = CaseReconstructor(db_session)
    test_extractor = TestExtractor(llm_client=base_llm)
    evaluator = ResponseEvaluator(llm_client=llm_client)

    print(f"\n  ✅ LLM           : {os.getenv('VERTEX_FLASH_MODEL')}")
    print(f"  ✅ Database      : {db_session.bind.url}")

    # ── Diagnoses ──
    all_case_ids = list({e["case_id"] for e in entries})
    diagnosis_map: dict[int, str] = {}
    for cid in all_case_ids:
        rows = db_session.execute(
            sql_text("""
                SELECT condition_text FROM case_diagnosis_conditions
                WHERE case_id = :case_id ORDER BY condition_order
            """),
            {"case_id": cid},
        ).fetchall()
        if rows:
            diagnosis_map[cid] = "; ".join(r.condition_text for r in rows)

    # ── Pre-build full contexts (once per case — all tests included) ──
    context_cache: dict[int, dict] = {}
    for cid in all_case_ids:
        ctx = build_full_context_with_test(reconstructor, cid)
        context_cache[cid] = ctx

    total_runs = len(entries) * repetitions
    print(f"\n🚀 Processing {len(entries)} entries × {repetitions} reps = {total_runs} runs …\n")

    results: list[dict] = []

    for idx, entry in enumerate(tqdm(entries, desc="Processing samples")):
        cid = entry["case_id"]
        diagnosis = diagnosis_map.get(cid)

        # Extract test name once per entry
        test_name = test_extractor.extract_test_name(
            content=entry["raw_content"],
            category=entry["category"],
        )

        ctx_data = context_cache[cid]
        case_context = ctx_data["full_context"]
        context_meta = {
            "method": "full_context_with_test",
            "success": True,
            "context_length": len(case_context),
            "items_included": ctx_data["stats"]["included_count"],
            "items_excluded": 0,
            "excluded_item": None,
        }

        for run_num in range(1, repetitions + 1):
            prompts_log.clear()
            llm_client.call_count = 0

            result = process_single_sample(
                entry=entry,
                generator=generator,
                evaluator=evaluator,
                db_session=db_session,
                prompts_log=prompts_log,
                diagnosis=diagnosis,
                case_context=case_context,
                test_name=test_name,
                context_meta=context_meta,
            )

            result["run_number"] = run_num
            result["model"] = os.getenv("VERTEX_FLASH_MODEL", "")

            # Save prompt logs
            case_dir = OUTPUT_DIR / f"case_{cid}"
            case_dir.mkdir(parents=True, exist_ok=True)

            if prompts_log:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                pf = case_dir / f"{entry['id']}_run{run_num}_prompts_{ts}.json"
                with open(pf, "w") as f:
                    json.dump({
                        "entry_id": entry["id"],
                        "case_id": cid,
                        "timestamp": result.get("timestamp"),
                        "run_number": run_num,
                        "model": os.getenv("VERTEX_FLASH_MODEL"),
                        "experiment_type": "leakage_positive_control",
                        "prompts": prompts_log,
                    }, f, indent=2)
                result["prompt_log_file"] = str(pf.relative_to(PROJECT_ROOT))

            results.append(result)

        if (idx + 1) % 10 == 0:
            ok = sum(1 for r in results if r.get("success"))
            print(f"\n  Progress: {idx+1}/{len(entries)} | Runs: {len(results)}/{total_runs} | Success: {ok}/{len(results)}")

    # ── Summary stats ──
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"\n✅ Processing complete!  Successful: {len(successful)}/{len(results)}  Failed: {len(failed)}")

    if failed:
        print(f"\n❌ Failed samples (first 5):")
        for r in failed[:5]:
            print(f"  - {r['entry_id']}: {r.get('error', '?')}")

    if successful:
        evaluations = [r["evaluation"] for r in successful]
        summary = evaluator.summarize_results(evaluations)
    else:
        summary = {}

    # ── Save results ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full results
    experiment_out = {
        "timestamp": datetime.now().isoformat(),
        "experiment_type": "leakage_positive_control",
        "parameters": {
            "total_samples": len(entries),
            "repetitions": repetitions,
            "model": os.getenv("VERTEX_FLASH_MODEL"),
            "context_method": "full_context_with_test",
        },
        "summary": summary,
        "results": results,
    }
    full_file = OUTPUT_DIR / f"leakage_experiment_{ts}.json"
    with open(full_file, "w") as f:
        json.dump(experiment_out, f, indent=2)
    print(f"\n💾 Full results   → {full_file.relative_to(PROJECT_ROOT)}")

    # Summary JSON (no prompt payloads)
    summary_file = OUTPUT_DIR / f"leakage_summary_{ts}.json"
    summary_out = {
        "timestamp": experiment_out["timestamp"],
        "experiment_type": "leakage_positive_control",
        "parameters": experiment_out["parameters"],
        "summary": summary,
        "results": [
            {
                "entry_id": r["entry_id"],
                "case_id": r["case_id"],
                "clinical_data_id": r.get("clinical_data_id"),
                "timestamp": r.get("timestamp"),
                "run_number": r.get("run_number"),
                "model": r.get("model", os.getenv("VERTEX_FLASH_MODEL")),
                "category": r["category"],
                "diagnosis": r.get("diagnosis"),
                "success": r["success"],
                "user_request": r.get("user_request"),
                "extracted_test_name": r.get("extracted_test_name"),
                "ground_truth": r.get("ground_truth"),
                "generated_response": r.get("generated_response"),
                "evaluation": r.get("evaluation", {}),
                "prompt_log_file": r.get("prompt_log_file"),
            }
            for r in results
        ],
    }
    with open(summary_file, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"📊 Summary JSON   → {summary_file.relative_to(PROJECT_ROOT)}")

    # Per-case CSVs
    print("\n📊 Creating per-case CSV files …")
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "scripts"))
    from summary_to_csv import json_to_csv

    results_by_case: dict[int, list] = defaultdict(list)
    for r in results:
        results_by_case[r["case_id"]].append(r)

    for cid, case_results in sorted(results_by_case.items()):
        case_dir = OUTPUT_DIR / f"case_{cid}"
        case_dir.mkdir(parents=True, exist_ok=True)

        case_json = {
            "timestamp": experiment_out["timestamp"],
            "parameters": experiment_out["parameters"],
            "results": case_results,
        }
        case_json_path = case_dir / f"case_{cid}_leakage_summary_{ts}.json"
        with open(case_json_path, "w") as f:
            json.dump(case_json, f, indent=2)

        csv_path = json_to_csv(str(case_json_path))
        print(f"  ✅ Case {cid} CSV: {csv_path}  ({len(case_results)} rows)")

    print("\n" + "=" * 80)
    return experiment_out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run leakage experiment (positive control — context INCLUDES target test).",
    )
    parser.add_argument("--case-ids", type=int, nargs="+", help="Case ID(s)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--ground-truth-file", type=str, default=None)
    args = parser.parse_args()

    run_leakage_experiment(
        case_ids=args.case_ids,
        limit=args.limit,
        start_from=args.start_from,
        repetitions=args.repetitions,
        model_name=args.model,
        ground_truth_file=args.ground_truth_file,
    )
