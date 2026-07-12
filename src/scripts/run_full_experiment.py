#!/usr/bin/env python3
"""
Run full experiment on all 100 ground truth samples.
Processes each sample, performs ablation and generation, evaluates results.
"""

import sys
import json
import warnings
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm
import asyncio

# Suppress Vertex AI SDK deprecation warning
warnings.filterwarnings('ignore', message='.*This feature is deprecated.*genai-vertexai-sdk.*')

# Load environment variables
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.db_utils import get_sqlalchemy_session
from src.app_code.synthetic_response_generator import SyntheticResponseGenerator
from src.llm.vertex_client import VertexAIClient
from src.llm.client import get_client_for_model
from src.extraction.context_ablator import ContextAblator
from src.extraction.case_reconstructor import CaseReconstructor
from src.extraction.test_extractor import TestExtractor
from src.evaluation.metrics import ResponseEvaluator
from sqlalchemy import text as sql_text
import os


class PromptLoggingWrapper:
    """Wrapper that logs all prompts and responses sent to/from the LLM."""
    def __init__(self, client, log_dict):
        self.client = client
        self.log = log_dict
        self.call_count = 0
    
    def _serialize_response(self, response):
        """Convert response to JSON-serializable format."""
        if isinstance(response, str):
            return response
        elif hasattr(response, 'text'):
            return response.text
        elif hasattr(response, '__dict__'):
            return str(response)
        else:
            return str(response)
    
    def text_completion(self, prompt, **kwargs):
        self.call_count += 1
        call_id = f"call_{self.call_count}"
        self.log[call_id] = {
            "prompt": prompt,
            "kwargs": kwargs
        }
        response = self.client.text_completion(prompt, **kwargs)
        self.log[call_id]["response"] = self._serialize_response(response)
        return response
    
    def chat_completion(self, messages, **kwargs):
        self.call_count += 1
        call_id = f"call_{self.call_count}"
        self.log[call_id] = {
            "messages": messages,
            "kwargs": kwargs
        }
        response = self.client.chat_completion(messages, **kwargs)
        self.log[call_id]["response"] = self._serialize_response(response)
        return response
    
    def __getattr__(self, name):
        # Delegate all other attributes/methods to the wrapped client
        return getattr(self.client, name)


def process_single_sample(entry, generator, reconstructor, evaluator, test_extractor, db_session, prompts_log, diagnosis, use_reconstruction=True,
                          precomputed_test_name=None, precomputed_units=None, precomputed_context=None, precomputed_context_meta=None, output_dir=None):
    """
    Process a single ground truth sample.
    
    Args:
        entry: Ground truth entry with case info and test content
        generator: SyntheticResponseGenerator instance
        reconstructor: CaseReconstructor instance (or ContextAblator for legacy mode)
        evaluator: ResponseEvaluator instance
        test_extractor: TestExtractor instance to extract specific test names
        db_session: Database session
        prompts_log: Dict to log all LLM prompts/responses
        diagnosis: Patient diagnosis string
        use_reconstruction: If True, use structured reconstruction; if False, use ablation (legacy)
        precomputed_test_name: Pre-extracted test name (avoids redundant LLM call across repetitions)
        precomputed_context: Pre-built case context string (avoids redundant reconstruction)
        precomputed_context_meta: Pre-built context_preparation metadata dict
    
    Returns:
        Dict with processing results and evaluation metrics
    """
    result = {
        "entry_id": entry['id'],
        "case_id": entry['case_id'],
        "clinical_data_id": None,  # Will be populated if using reconstruction
        "category": entry['category'],
        "test_name": entry.get('test_name'),
        "ground_truth": entry['raw_content'],
        "diagnosis": diagnosis,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error": None
    }
    
    try:
        # Step 0: Extract specific test name from clinical data content
        # Use pre-computed value if provided (avoids redundant LLM call across repetitions)
        if precomputed_test_name is not None:
            specific_test_name = precomputed_test_name
        else:
            print(f"🔍 Extracting specific test name from content...")
            specific_test_name = test_extractor.extract_test_name(
                content=entry['raw_content'],
                category=entry['category']
            )
            print(f"   Extracted test: '{specific_test_name}'")
        result["extracted_test_name"] = specific_test_name
        
        # Extract units from GT content (for unit-aware generation)
        if precomputed_units is not None:
            gt_units = precomputed_units
        else:
            gt_units = test_extractor.extract_units(
                content=entry['raw_content'],
                category=entry['category']
            )
        if gt_units:
            result["gt_units"] = gt_units
        
        # Step 1: Prepare case context (reconstruction vs ablation)
        # Use pre-computed context if provided (avoids redundant reconstruction across repetitions)
        if precomputed_context is not None:
            case_context = precomputed_context
            result["context_preparation"] = precomputed_context_meta
            if precomputed_context_meta and precomputed_context_meta.get('excluded_item'):
                result["clinical_data_id"] = precomputed_context_meta['excluded_item'].get('id')
        elif use_reconstruction:
            # NEW APPROACH: Reconstruct from structured data
            print(f"🔧 Reconstructing case context (structured data approach)...")
            reconstruction_result = reconstructor.reconstruct_case_without_test(
                case_id=entry['case_id'],
                exclude_test_content=entry['raw_content'],
                exclude_category=entry['category']
            )
            
            case_context = reconstruction_result['full_context']
            
            # Capture clinical_data_id if excluded item exists
            if reconstruction_result['excluded_item']:
                result["clinical_data_id"] = reconstruction_result['excluded_item'].get('id')
            
            result["context_preparation"] = {
                "method": "structured_reconstruction",
                "success": True,
                "context_length": len(case_context),
                "items_included": reconstruction_result['stats']['included_count'],
                "items_excluded": reconstruction_result['stats']['excluded_count'],
                "excluded_item": reconstruction_result['excluded_item']
            }
        else:
            # LEGACY APPROACH: Text ablation
            print(f"⚠️  Using legacy ablation approach...")
            ablation_result = reconstructor.ablate_with_llm(  # reconstructor is actually ablator in legacy mode
                case_text=entry['case_text'],
                raw_content=entry['raw_content'],
                category=entry['category'],
                test_name=entry.get('test_name', 'Unknown')
            )
            
            case_context = ablation_result.ablated_text
            
            result["context_preparation"] = {
                "method": "text_ablation",
                "success": ablation_result.success,
                "context_length": len(ablation_result.ablated_text),
                "ablation_method": ablation_result.ablation_method,
                "removed_length": len(ablation_result.removed_content)
            }
            
            if not ablation_result.success:
                result["error"] = "Ablation failed"
                return result
        
        # Step 2: Generate response with prepared context
        # Temporarily replace case text in database with prepared context
        case_query = db_session.execute(
            sql_text("SELECT id, text FROM cases WHERE id = :case_id"),
            {"case_id": entry['case_id']}
        ).first()
        
        if not case_query:
            result["error"] = f"Case {entry['case_id']} not found"
            return result
        
        original_text = case_query.text
        
        # Update with prepared context
        db_session.execute(
            sql_text("UPDATE cases SET text = :context WHERE id = :case_id"),
            {"context": case_context, "case_id": entry['case_id']}
        )
        db_session.commit()
        
        try:
            # Generate synthetic response using extracted test name
            user_request = f"Perform {specific_test_name}"
            if gt_units:
                units_str = ', '.join(gt_units)
                user_request += f" (report numeric values in: {units_str})"
            print(f"📝 User request: '{user_request}'")
            result["user_request"] = user_request
            
            generated_response = asyncio.run(
                generator.generate_synthetic_response(
                    user_query=user_request,
                    case_id=entry['case_id'],
                    chat_history=[],
                    partial_response=None,
                    cpt_codes=None
                )
            )
            
            result["generated_response"] = generated_response.get('assistant_response', '')
            result["formatting_type"] = generated_response.get('formatting_type', '')
            
            # Step 3: Evaluate (pass diagnosis and case text for LLM evaluation)
            evaluation = evaluator.evaluate(
                generated=result["generated_response"],
                ground_truth=entry['raw_content'],
                category=entry['category'],
                diagnosis=diagnosis,
                case_text=entry['case_text']
            )
            
            result["evaluation"] = evaluation
            result["success"] = True
            
        finally:
            # Restore original text
            db_session.execute(
                sql_text("UPDATE cases SET text = :original_text WHERE id = :case_id"),
                {"original_text": original_text, "case_id": entry['case_id']}
            )
            db_session.commit()
    
    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    
    return result


def run_full_experiment(case_ids=None, limit=None, start_from=0, use_reconstruction=True, repetitions=1, model_name=None, ground_truth_file=None, same_category_strategy='mask', output_dir=None):
    """
    Run experiment on ground truth samples.
    
    Args:
        case_ids: List of case IDs to process (None = all cases)
        limit: Maximum number of samples to process (None = all, only used if case_ids is None)
        start_from: Index to start from (for resuming, only used if case_ids is None)
        use_reconstruction: Use structured reconstruction (True) or legacy ablation (False)
        repetitions: Number of times to run each test (for consistency checking)
        model_name: Override the Vertex AI model (e.g. 'gemini-2.5-pro'). Defaults to VERTEX_FLASH_MODEL env var.
        ground_truth_file: Path to a ground truth JSON file. Defaults to data/ground_truth_sample.json.
    """
    
    print("=" * 80)
    print("FULL EXPERIMENT: Synthetic Response Generator Evaluation")
    print("=" * 80)
    
    # Load ground truth data
    if ground_truth_file:
        ground_truth_path = Path(ground_truth_file)
    else:
        ground_truth_path = project_root / "data" / "ground_truth_sample.json"

    with open(ground_truth_path, 'r') as f:
        ground_truth_data = json.load(f)
    
    # Filter by case_ids if provided, otherwise use start_from/limit
    if case_ids:
        entries = [e for e in ground_truth_data['entries'] if e['case_id'] in case_ids]
        print(f"\n📊 Experiment Parameters:")
        print(f"  Total samples in dataset: {len(ground_truth_data['entries'])}")
        print(f"  Processing: {len(entries)} samples from case(s): {', '.join(map(str, case_ids))}")
    else:
        entries = ground_truth_data['entries'][start_from:start_from + limit if limit else None]
        print(f"\n📊 Experiment Parameters:")
        print(f"  Total samples in dataset: {len(ground_truth_data['entries'])}")
        print(f"  Processing: {len(entries)} samples")
        print(f"  Starting from index: {start_from}")
    print(f"  Context method: {'Structured Reconstruction' if use_reconstruction else 'Text Ablation (Legacy)'}")
    print(f"  Repetitions per test: {repetitions}")
    print(f"  Model: {model_name or os.getenv('VERTEX_FLASH_MODEL', 'gemini-2.5-flash')}")
    
    # Initialize components
    print("\n🔧 Initializing components...")

    # Determine the effective model name
    effective_model = model_name or os.getenv("VERTEX_FLASH_MODEL", "gemini-2.5-flash")

    # Auto-route: Bedrock for provider-prefixed models, Vertex AI for Gemini, etc.
    _BEDROCK_PREFIXES = (
        "anthropic.", "amazon.", "meta.", "mistral.", "cohere.",
        "ai21.", "stability.", "writer.", "nvidia.", "qwen.",
        "minimax.", "twelvelabs.", "openai.gpt-oss", "zai.",
        "eu.anthropic.", "us.anthropic.", "ap.anthropic.", "global.anthropic.",
        "eu.amazon.", "us.amazon.", "ap.amazon.", "global.amazon.",
        "eu.meta.", "us.meta.", "ap.meta.", "global.meta.",
        "google.",
    )
    is_bedrock = any(effective_model.startswith(p) for p in _BEDROCK_PREFIXES)

    if is_bedrock:
        from src.llm.bedrock_client import BedrockClient
        base_llm_client = BedrockClient(
            region=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-west-1")),
            model_id=effective_model,
        )
        print(f"  🟠 Backend: AWS Bedrock | model={effective_model} | region={os.getenv('AWS_REGION', 'eu-west-1')}")
    else:
        # Vertex AI (Gemini) path — keep existing behaviour
        if model_name:
            os.environ["VERTEX_FLASH_MODEL"] = model_name
            os.environ["VERTEX_TEXT_MODEL"] = model_name
            os.environ["VERTEX_VISION_MODEL"] = model_name
            import src.llm.config as llm_config
            llm_config._config_cache = None
        base_llm_client = VertexAIClient(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION"),
            credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
        print(f"  🔵 Backend: Google Vertex AI | model={effective_model}")
    
    # Create wrapper for prompt logging
    prompts_log = {}
    llm_client = PromptLoggingWrapper(base_llm_client, prompts_log)
    
    db_session = get_sqlalchemy_session()
    generator = SyntheticResponseGenerator(db_session, llm_client)
    
    # Initialize context preparation strategy
    if use_reconstruction:
        context_preparer = CaseReconstructor(db_session, llm_client=base_llm_client, same_category_strategy=same_category_strategy)
        print(f"  ✅ Context strategy: Structured Reconstruction (database-level filtering + HPI ablation)")
        print(f"  ✅ Same-category strategy: {same_category_strategy}")
    else:
        context_preparer = ContextAblator(llm_client=llm_client)
        print(f"  ⚠️  Context strategy: Text Ablation (legacy LLM-based)")
    
    test_extractor = TestExtractor(llm_client=base_llm_client)
    print(f"  ✅ Test Extractor: Initialized")
    
    evaluator = ResponseEvaluator(llm_client=llm_client)
    
    print(f"  ✅ Evaluator LLM: {os.getenv('VERTEX_FLASH_MODEL', 'gemini-2.5-flash')} (scoring/judging only)")
    print(f"  ✅ Database: {db_session.bind.url}")
    print(f"  ✅ LLM-based evaluation: ENABLED")
    
    # Fetch diagnoses for all cases
    print("\n📋 Fetching case diagnoses...")
    case_ids = list(set(entry['case_id'] for entry in entries))
    diagnosis_map = {}
    
    for case_id in case_ids:
        diagnosis_results = db_session.execute(
            sql_text("""
                SELECT condition_text 
                FROM case_diagnosis_conditions 
                WHERE case_id = :case_id
                ORDER BY condition_order
            """),
            {"case_id": case_id}
        ).fetchall()
        
        if diagnosis_results:
            # Combine multiple diagnoses with semicolon
            diagnosis_map[case_id] = "; ".join([row.condition_text for row in diagnosis_results])
    
    print(f"  ✅ Loaded diagnoses for {len(diagnosis_map)} cases")
    
    # Process samples
    total_runs = len(entries) * repetitions
    print(f"\n🚀 Processing {len(entries)} samples × {repetitions} repetitions = {total_runs} total runs...")
    results = []
    
    for idx, entry in enumerate(tqdm(entries, desc="Processing samples")):
        # Pre-compute once per entry (shared across all repetitions)
        diagnosis = diagnosis_map.get(entry['case_id'])

        print(f"🔍 Extracting specific test name and units from content...")
        precomputed_test_name, precomputed_units = test_extractor.extract_test_name_and_units(
            content=entry['raw_content'],
            category=entry['category']
        )
        print(f"   Extracted test: '{precomputed_test_name}'")
        if precomputed_units:
            print(f"   Extracted units: {precomputed_units}")

        precomputed_context = None
        precomputed_context_meta = None
        if use_reconstruction:
            print(f"🔧 Reconstructing case context (structured data approach)...")
            reconstruction_result = context_preparer.reconstruct_case_without_test(
                case_id=entry['case_id'],
                exclude_test_content=entry['raw_content'],
                exclude_category=entry['category']
            )
            precomputed_context = reconstruction_result['full_context']
            precomputed_context_meta = {
                "method": "structured_reconstruction",
                "success": True,
                "context_length": len(precomputed_context),
                "items_included": reconstruction_result['stats']['included_count'],
                "items_excluded": reconstruction_result['stats']['excluded_count'],
                "excluded_item": reconstruction_result['excluded_item'],
                "hpi_sentences_removed": reconstruction_result.get('hpi_sentences_removed', []),
                "context_items_removed": [
                    {"category": it.get("category", "unknown"), "content": it.get("content", "")}
                    for it in reconstruction_result.get('context_items_removed', [])
                ],
                "same_category_strategy": reconstruction_result.get('same_category_strategy', 'none'),
                "same_category_masked_count": reconstruction_result.get('same_category_masked_count', 0),
                "same_category_removed_count": reconstruction_result.get('same_category_removed_count', 0),
                "vital_signs_masked_count": reconstruction_result.get('vital_signs_masked_count', 0),
            }

        # Run each test multiple times for consistency checking
        for run_num in range(1, repetitions + 1):
            # Reset prompts log for each run
            prompts_log.clear()
            llm_client.call_count = 0  # Reset the call counter for each run

            result = process_single_sample(
                entry,
                generator,
                context_preparer,  # Can be CaseReconstructor or ContextAblator
                evaluator,
                test_extractor,
                db_session,
                prompts_log,
                diagnosis,
                use_reconstruction=use_reconstruction,
                precomputed_test_name=precomputed_test_name,
                precomputed_units=precomputed_units,
                precomputed_context=precomputed_context,
                precomputed_context_meta=precomputed_context_meta,
                output_dir=output_dir
            )
            
            # Add run number to the result
            result['run_number'] = run_num
            
            # Create case-specific directory for logs
            _out = Path(output_dir) if output_dir else project_root / "output"
            case_dir = _out / f"case_{entry['case_id']}"
            case_dir.mkdir(parents=True, exist_ok=True)
            
            # Save prompts to case-specific directory with run number
            if prompts_log:
                prompt_file = case_dir / f"{entry['id']}_run{run_num}_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(prompt_file, 'w') as f:
                    json.dump({
                        "entry_id": entry['id'],
                        "case_id": entry['case_id'],
                        "clinical_data_id": result.get('clinical_data_id'),
                        "timestamp": result.get('timestamp'),
                        "run_number": run_num,
                        "model": os.getenv("VERTEX_FLASH_MODEL"),
                        "prompts": prompts_log
                    }, f, indent=2)
                result["prompt_log_file"] = str(prompt_file.resolve().relative_to(project_root.resolve()))
            
            # Add prompts to result (for small experiments)
            if len(entries) * repetitions <= 5:
                result["prompts"] = dict(prompts_log)
            
            results.append(result)
        
        # Progress update every 10 samples (accounting for repetitions)
        if (idx + 1) % 10 == 0:
            successful = sum(1 for r in results if r.get('success'))
            total_processed = (idx + 1) * repetitions
            print(f"\n  Progress: {idx + 1}/{len(entries)} samples | Total runs: {total_processed}/{total_runs} | Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
    # Calculate summary statistics
    print("\n📈 Calculating summary statistics...")
    
    successful_results = [r for r in results if r.get('success')]
    failed_results = [r for r in results if not r.get('success')]
    
    print(f"\n✅ Processing complete!")
    print(f"  Successful: {len(successful_results)}/{len(results)}")
    print(f"  Failed: {len(failed_results)}/{len(results)}")
    
    if failed_results:
        print(f"\n❌ Failed samples:")
        for r in failed_results[:5]:  # Show first 5 failures
            print(f"  - {r['entry_id']}: {r.get('error', 'Unknown error')}")
    
    # Evaluate results
    if successful_results:
        evaluations = [r['evaluation'] for r in successful_results]
        summary = evaluator.summarize_results(evaluations)
        
        print(f"\n📊 Evaluation Summary:")
        print(f"  Data Leakage Rate (heuristic): {summary['data_leakage_rate']:.1f}%")
        llm_eval_summary = summary.get('llm_evaluation', {})
        if 'llm_data_leakage_rate' in llm_eval_summary:
            print(f"  Data Leakage Rate (LLM judge): {llm_eval_summary['llm_data_leakage_rate']:.1f}%")
        print(f"  Exact Match Rate: {summary['exact_match_rate']:.1f}%")
        print(f"  Ground Truth Containment: {summary['ground_truth_containment_rate']:.1f}%")
        print(f"  Avg Length Ratio: {summary['avg_length_ratio']:.2f}")

        # Show GT vs Generated for leaked entries (prefer LLM judge, fall back to heuristic)
        leaked = [r for r in successful_results 
                  if r.get('evaluation', {}).get('llm_evaluation', {}).get('data_leakage', {}).get('leaked', False)]
        leak_source = "LLM judge"
        if not leaked:
            # Fall back to heuristic if LLM judge found nothing (or wasn't available)
            leaked = [r for r in successful_results if r.get('evaluation', {}).get('data_leakage_detected')]
            leak_source = "heuristic"
        if leaked:
            print(f"\n  🔴 Leaked entries — {leak_source} ({len(leaked)}):")
            for r in leaked:
                gt_short = r.get('ground_truth', '')[:120]
                gen_short = r.get('generated_response', '')[:120]
                llm_reason = r.get('evaluation', {}).get('llm_evaluation', {}).get('data_leakage', {}).get('reasoning', '')
                print(f"    ─ {r['entry_id']} (run {r.get('run_number', '?')})")
                print(f"      GT:  {gt_short}{'...' if len(r.get('ground_truth', '')) > 120 else ''}")
                print(f"      GEN: {gen_short}{'...' if len(r.get('generated_response', '')) > 120 else ''}")
                if llm_reason:
                    print(f"      LLM: {llm_reason[:150]}")
                ablated = r.get('context_preparation', {}).get('context_items_removed', [])
                if ablated:
                    print(f"      ABLATED: {[it['content'][:80] for it in ablated]}")
        else:
            print(f"\n  ✅ No data leakage detected in any entry.")
        
        if 'imaging' in summary:
            print(f"\n  Imaging Specific:")
            print(f"    Location Overlap: {summary['imaging']['location_overlap']}/{summary['imaging']['samples']}")
            print(f"    Measurement Match: {summary['imaging']['measurement_exact_match']}/{summary['imaging']['samples']}")
        
        if 'lab' in summary:
            print(f"\n  Lab Specific:")
            print(f"    Value Exact Match: {summary['lab']['value_exact_match']}/{summary['lab']['samples']}")
    else:
        summary = {}
    
    # Save results
    output_dir = Path(output_dir) if output_dir else project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"full_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    experiment_results = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "total_samples": len(entries),
            "start_from": start_from,
            "model": os.getenv('VERTEX_FLASH_MODEL'),
            "context_method": "structured_reconstruction" if use_reconstruction else "text_ablation"
        },
        "summary": summary,
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Log case-specific prompt files
    case_ids = list(set(r['case_id'] for r in results))
    if case_ids:
        print(f"\n📁 Case-specific prompt logs saved to:")
        for case_id in sorted(case_ids):
            case_dir = output_dir / f"case_{case_id}"
            if case_dir.exists():
                num_files = len(list(case_dir.glob('*_prompts_*.json')))
                print(f"  - output/case_{case_id}/ ({num_files} prompt log files)")
    
    # Create a summary file with metadata only (no prompts)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = output_dir / f"experiment_summary_{timestamp}.json"
    summary_results = {
        "timestamp": experiment_results['timestamp'],
        "parameters": experiment_results['parameters'],
        "summary": experiment_results['summary'],
        "results": [
            {
                "entry_id": r['entry_id'],
                "case_id": r['case_id'],
                "clinical_data_id": r.get('clinical_data_id'),
                "timestamp": r.get('timestamp'),
                "run_number": r.get('run_number'),
                "model": os.getenv("VERTEX_FLASH_MODEL"),
                "category": r['category'],
                "diagnosis": r.get('diagnosis'),
                "success": r['success'],
                "user_request": r.get('user_request'),
                "extracted_test_name": r.get('extracted_test_name'),
                "ground_truth": r.get('ground_truth'),
                "generated_response": r.get('generated_response'),
                "evaluation": r.get('evaluation', {}),
                "prompt_log_file": r.get('prompt_log_file')
            }
            for r in results
        ]
    }
    with open(summary_file, 'w') as f:
        json.dump(summary_results, f, indent=2)
    print(f"📊 Summary (without prompts) saved to: {summary_file}")
    
    # Group results by case_id and create separate CSV files
    print("\n📊 Creating case-specific CSV files...")
    from collections import defaultdict
    sys.path.insert(0, str(Path(__file__).parent))
    from summary_to_csv import json_to_csv
    
    # Group results by case_id
    results_by_case = defaultdict(list)
    for result in results:
        case_id = result.get('case_id')
        if case_id:
            results_by_case[case_id].append(result)
    
    # Create CSV for each case
    csv_files = []
    for case_id, case_results in results_by_case.items():
        # Create case-specific directory
        case_dir = output_dir / f"case_{case_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary JSON for this case
        case_summary = {
            "timestamp": experiment_results['timestamp'],
            "parameters": experiment_results['parameters'],
            "results": case_results
        }
        
        case_json_file = case_dir / f"case_{case_id}_summary_{timestamp}.json"
        with open(case_json_file, 'w') as f:
            json.dump(case_summary, f, indent=2)
        
        # Convert to CSV
        case_csv_file = json_to_csv(str(case_json_file))
        csv_files.append(case_csv_file)
        print(f"  ✅ Case {case_id} CSV: {case_csv_file} ({len(case_results)} tests)")
    
    print("\n" + "=" * 80)
    
    return experiment_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run synthetic response generator experiment")
    parser.add_argument("--case-ids", type=int, nargs='+', help="Case ID(s) to process (e.g., --case-ids 1 2 3)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process (used if --case-ids not provided)")
    parser.add_argument("--start-from", type=int, default=0, help="Start from this index (used if --case-ids not provided)")
    parser.add_argument('--use-ablation', action='store_true', help='Use legacy text ablation instead of structured reconstruction')
    parser.add_argument('--repetitions', type=int, default=1, help='Number of times to run each test for consistency checking (default: 1)')
    parser.add_argument('--model', type=str, default=None, help='Vertex AI model to use (e.g. gemini-2.5-flash, gemini-2.5-pro). Defaults to VERTEX_FLASH_MODEL env var.')
    parser.add_argument('--ground-truth-file', type=str, default=None, help='Path to a ground truth JSON file (default: data/ground_truth_sample.json)')
    parser.add_argument('--same-category-strategy', type=str, default='hybrid', choices=['hybrid', 'mask', 'remove', 'none'],
                        help='How to handle same-category items: hybrid=mask labs+remove imaging (default), mask=replace numerics with [VALUE], remove=remove all, none=legacy')
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory (default: output/)')

    args = parser.parse_args()

    # Default to reconstruction (new approach), unless --use-ablation flag is set
    use_reconstruction = not args.use_ablation

    run_full_experiment(
        case_ids=args.case_ids,
        limit=args.limit,
        start_from=args.start_from,
        use_reconstruction=use_reconstruction,
        repetitions=args.repetitions,
        model_name=args.model,
        ground_truth_file=args.ground_truth_file,
        same_category_strategy=args.same_category_strategy,
        output_dir=args.output_dir
    )
