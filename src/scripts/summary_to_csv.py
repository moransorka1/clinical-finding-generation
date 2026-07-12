#!/usr/bin/env python3
"""
Convert experiment summary JSON to CSV format.
"""
import json
import csv
import sys
from pathlib import Path

def json_to_csv(json_path: str, csv_path: str = None):
    """Convert experiment summary JSON to CSV."""
    
    # Handle Path objects
    from pathlib import Path
    if isinstance(json_path, Path):
        json_path = str(json_path)
    if csv_path and isinstance(csv_path, Path):
        csv_path = str(csv_path)
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Generate CSV path if not provided
    if csv_path is None:
        csv_path = json_path.replace('.json', '.csv')
    
    # Define CSV columns
    columns = [
        'entry_id',
        'case_id',
        'clinical_data_id',
        'run_number',
        'model',
        'category',
        'diagnosis',
        'timestamp',
        'success',
        'user_request',
        'extracted_test_name',
        'ground_truth',
        'generated_response',
        # Evaluation metrics
        'exact_match',
        'contains_ground_truth',
        'data_leakage_detected',
        # Numeric analysis
        'numeric_values_count_gt',
        'numeric_values_count_generated',
        'numeric_matches_count',
        'mean_numeric_distance_percent',
        'numeric_comparison_detailed',
        # Imaging specific
        'imaging_location_overlap',
        'imaging_measurement_exact_match',
        # LLM evaluation (2 metrics)
        'llm_ground_truth_similarity',
        'llm_ground_truth_similarity_reasoning',
        'llm_clinical_plausibility',
        'llm_clinical_plausibility_reasoning',
        'llm_data_leakage',
        'llm_data_leakage_reasoning',
        'llm_summary',
        'llm_red_flags'
    ]
    
    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for result in data['results']:
            row = {
                'entry_id': result.get('entry_id'),
                'case_id': result.get('case_id'),
                'clinical_data_id': result.get('clinical_data_id'),
                'run_number': result.get('run_number'),
                'model': result.get('model'),
                'category': result.get('category'),
                'diagnosis': result.get('diagnosis'),
                'timestamp': result.get('timestamp'),
                'success': result.get('success'),
                'user_request': result.get('user_request'),
                'extracted_test_name': result.get('extracted_test_name'),
                'ground_truth': result.get('ground_truth'),
                'generated_response': result.get('generated_response'),
            }
            
            # Add evaluation metrics
            eval_data = result.get('evaluation', {})
            row['exact_match'] = eval_data.get('exact_match')
            row['contains_ground_truth'] = eval_data.get('contains_ground_truth')
            row['data_leakage_detected'] = eval_data.get('data_leakage_detected')
            
            # Numeric analysis
            row['numeric_values_count_gt'] = len(eval_data.get('numeric_values_gt', []))
            row['numeric_values_count_generated'] = len(eval_data.get('numeric_values_generated', []))
            row['numeric_matches_count'] = len(eval_data.get('numeric_matches', []))
            row['mean_numeric_distance_percent'] = eval_data.get('mean_numeric_distance_percent')
            row['numeric_comparison_detailed'] = eval_data.get('numeric_comparison_detailed', '')
            
            # Imaging specific
            row['imaging_location_overlap'] = eval_data.get('imaging_location_overlap')
            row['imaging_measurement_exact_match'] = eval_data.get('imaging_measurement_exact_match')
            
            # LLM evaluation (2 metrics)
            llm_eval = eval_data.get('llm_evaluation', {})
            if llm_eval and not llm_eval.get('error'):
                gt_sim = llm_eval.get('ground_truth_similarity', {})
                row['llm_ground_truth_similarity'] = gt_sim.get('score')
                row['llm_ground_truth_similarity_reasoning'] = gt_sim.get('reasoning')
                
                clin_plaus = llm_eval.get('clinical_plausibility', {})
                row['llm_clinical_plausibility'] = clin_plaus.get('score')
                row['llm_clinical_plausibility_reasoning'] = clin_plaus.get('reasoning')
                
                data_leak = llm_eval.get('data_leakage', {})
                row['llm_data_leakage'] = data_leak.get('leaked')
                row['llm_data_leakage_reasoning'] = data_leak.get('reasoning')
                
                row['llm_summary'] = llm_eval.get('summary')
                
                # Red flags as comma-separated list
                red_flags = llm_eval.get('red_flags', [])
                row['llm_red_flags'] = ', '.join(red_flags) if red_flags else ''
            
            writer.writerow(row)
    
    print(f"✅ CSV created: {csv_path}")
    print(f"   Rows: {len(data['results'])}")
    return csv_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summary_to_csv.py <json_file> [csv_file]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    json_to_csv(json_path, csv_path)
