"""
Evaluation metrics for synthetic response quality assessment
"""

from typing import Dict, Any, Optional, List, Tuple
import re
import json
import hashlib
import os
from difflib import SequenceMatcher


class ResponseEvaluator:
    """Evaluates synthetic responses against ground truth."""
    
    def __init__(self, llm_client=None):
        """
        Initialize the evaluator.
        
        Args:
            llm_client: Optional LLM client for advanced evaluation
        """
        self.llm_client = llm_client
        # Cache ground truth numeric extractions (keyed by text hash)
        # Ground truth is identical across all repetitions of the same entry
        self._gt_numeric_cache: Dict[str, List[Dict[str, Any]]] = {}
        # Load clinical reference ranges for normalized distance
        self._ref_ranges = self._load_reference_ranges()
    
    def _extract_numeric_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """
        Use LLM to extract numeric values from text.
        
        Returns:
            List of dicts: [{"test_name": str, "value": float, "unit": str}, ...]
        """
        prompt = f"""Extract all medical test results with numeric values from the following text.
For each test, identify:
1. The test name (normalized, without qualifiers like "elevated" or "decreased")
2. The numeric value (as a number, not a string)
3. The unit of measurement

Return ONLY a JSON array with this exact structure:
[
  {{"test_name": "test name here", "value": 123.45, "unit": "unit here"}},
  ...
]

Important:
- Normalize test names (e.g., "ALT" → "alanine aminotransferase", "CK" → "creatine kinase")
- Remove qualifiers from test names (e.g., "elevated serum cortisol" → "cortisol")
- Extract the actual numeric value, handling commas (e.g., "24,000" → 24000)
- If a test appears in a table, extract all rows
- Return an empty array [] if no numeric values found

Text to analyze:
{text}

JSON array:"""

        result = self.llm_client.text_completion(prompt, temperature=0.0, max_tokens=9000)
        response = result["text"]
        
        # Parse JSON response
        try:
            # Extract JSON array from response (handle potential markdown formatting)
            response_text = response.strip()
            if response_text.startswith('```'):
                # Remove markdown code blocks
                response_text = re.sub(r'^```[a-z]*\n', '', response_text)
                response_text = re.sub(r'\n```$', '', response_text)
                response_text = response_text.strip()
            
            numeric_values = json.loads(response_text)
            
            # Validate structure
            if not isinstance(numeric_values, list):
                raise ValueError("Response is not a list")
            
            for item in numeric_values:
                if not all(k in item for k in ['test_name', 'value', 'unit']):
                    raise ValueError(f"Invalid item structure: {item}")
                # Ensure value is numeric
                item['value'] = float(item['value'])
            
            return numeric_values
        except (json.JSONDecodeError, ValueError) as e:
            raise Exception(f"Failed to parse LLM response: {e}. Response: {response[:200]}")
    
    def _extract_numeric_from_both_with_llm(self, ground_truth: str, generated: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Use LLM to extract numeric values from both ground truth and generated text,
        ensuring consistent test name normalization across both.
        
        Returns:
            Tuple of (gt_values, gen_values)
        """
        prompt = f"""Extract all medical test results with numeric values from BOTH texts below.
For each test, identify:
1. The test name (normalized, without qualifiers like "elevated" or "decreased")
2. The numeric value (as a number, not a string)
3. The unit of measurement

CRITICAL: Use the EXACT SAME normalized test name when the same test appears in both texts.
For example:
- "alanine transaminase", "ALT", "Alanine Aminotransferase" should all be normalized to "alanine transaminase"
- "creatine kinase", "CK", "Serum CK" should all be normalized to "creatine kinase"

Return ONLY a JSON object with this exact structure:
{{
  "ground_truth": [
    {{"test_name": "normalized test name", "value": 123.45, "unit": "unit here"}},
    ...
  ],
  "generated": [
    {{"test_name": "normalized test name", "value": 123.45, "unit": "unit here"}},
    ...
  ]
}}

Important:
- Normalize test names consistently (e.g., "ALT" → "alanine transaminase")
- Remove qualifiers from test names (e.g., "elevated serum cortisol" → "cortisol")
- Extract actual numeric values, handling commas (e.g., "24,000" → 24000)
- If a test appears in a table, extract all rows
- Use lowercase for test names
- Return empty arrays [] if no numeric values found

=== GROUND TRUTH TEXT ===
{ground_truth}

=== GENERATED TEXT ===
{generated}

JSON object:"""

        result = self.llm_client.text_completion(prompt, temperature=0.0, max_tokens=8000)
        response = result["text"]
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle potential markdown formatting)
            response_text = response.strip()
            if response_text.startswith('```'):
                # Remove markdown code blocks
                response_text = re.sub(r'^```[a-z]*\n', '', response_text)
                response_text = re.sub(r'\n```$', '', response_text)
                response_text = response_text.strip()
            
            result_obj = json.loads(response_text)
            
            # Validate structure
            if not isinstance(result_obj, dict) or 'ground_truth' not in result_obj or 'generated' not in result_obj:
                raise ValueError("Response must have 'ground_truth' and 'generated' keys")
            
            gt_values = result_obj['ground_truth']
            gen_values = result_obj['generated']
            
            # Validate each list
            for values_list in [gt_values, gen_values]:
                if not isinstance(values_list, list):
                    raise ValueError("Values must be lists")
                for item in values_list:
                    if not all(k in item for k in ['test_name', 'value', 'unit']):
                        raise ValueError(f"Invalid item structure: {item}")
                    # Ensure value is numeric
                    item['value'] = float(item['value'])
            
            return gt_values, gen_values
        except (json.JSONDecodeError, ValueError) as e:
            raise Exception(f"Failed to parse LLM response: {e}. Response: {response[:200]}")
    
    def evaluate(self, generated: str, ground_truth: str, category: str, 
                 diagnosis: Optional[str] = None, case_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a generated response against ground truth.
        
        Args:
            generated: The generated response text
            ground_truth: The ground truth response
            category: The test category (lab, imaging, physical_exam)
            diagnosis: Optional diagnosis for context
            case_text: Optional case text for context
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            "exact_match": self._exact_match(generated, ground_truth),
            "length_ratio": self._length_ratio(generated, ground_truth),
            "contains_ground_truth": self._contains_ground_truth(generated, ground_truth),
            "data_leakage_detected": self._check_data_leakage(generated, ground_truth)
        }
        
        # Extract and compare numeric values
        numeric_comparison = self._compare_numeric_values(generated, ground_truth)
        metrics.update(numeric_comparison)
        
        # Category-specific metrics
        if category == "imaging":
            metrics.update(self._evaluate_imaging(generated, ground_truth))
        elif category == "laboratory_tests":
            metrics.update(self._evaluate_lab(generated, ground_truth))
        
        # LLM-based evaluation if client is available
        if self.llm_client and diagnosis:
            llm_metrics = self._evaluate_with_llm(
                generated, ground_truth, category, diagnosis, case_text
            )
            metrics["llm_evaluation"] = llm_metrics
        
        return metrics
    
    def extract_numeric_values(self, text: str, use_llm: bool = True) -> List[Dict[str, Any]]:
        """
        Extract numeric values with their test names and units from text.
        
        Args:
            text: Text to extract numeric values from
            use_llm: If True and llm_client available, use LLM extraction. Otherwise use regex.
        
        Returns:
            List of dicts: [{"test_name": str, "value": float, "unit": str}, ...]
        """
        # Try LLM-based extraction first if available
        if use_llm and self.llm_client:
            try:
                return self._extract_numeric_with_llm(text)
            except Exception as e:
                # Fallback to regex if LLM fails
                print(f"⚠️  LLM extraction failed, falling back to regex: {e}")
        
        # Regex-based extraction (fallback or default)
        numeric_values = []
        
        # Pattern 1: "Test Name | Value Unit | Reference" (table format)
        # Look for lines with pipes and numbers
        lines = text.split('\n')
        for line in lines:
            if '|' in line and re.search(r'\d', line):
                # Skip header lines
                if 'test name' in line.lower() or 'result' in line.lower():
                    continue
                    
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    # parts[0] = test name, parts[1] = value + unit
                    test_name = parts[0]
                    value_unit = parts[1]
                    
                    # Extract value and unit from the second column (handle commas in numbers)
                    match = re.match(r'([\d,]+\.?\d*)\s*([A-Za-z/%µ]+(?:/[A-Za-z]+)?)', value_unit)
                    if match and test_name:
                        value_clean = match.group(1).replace(',', '')  # Remove commas
                        numeric_values.append({
                            "test_name": test_name,
                            "value": float(value_clean),
                            "unit": match.group(2)
                        })
        
        # Pattern 2: "Test: Value Unit" (colon format)
        colon_pattern = r'([A-Za-z][A-Za-z\s\(\)]{2,40}?):\s*([\d,]+\.?\d*)\s*([A-Za-z/%µ]+(?:/[A-Za-z]+)?)'
        matches = re.findall(colon_pattern, text)
        for test_name, value, unit in matches:
            test_clean = test_name.strip()
            value_clean = value.replace(',', '')  # Remove commas
            # Avoid duplicates
            if not any(nv['test_name'] == test_clean and nv['value'] == float(value_clean) 
                      for nv in numeric_values):
                numeric_values.append({
                    "test_name": test_clean,
                    "value": float(value_clean),
                    "unit": unit.strip()
                })
        
        # Pattern 3: "Test Name of/at Value Unit" (narrative with prepositions)
        # More specific pattern to avoid grabbing extra words
        narrative_pattern = r'\b([A-Za-z][A-Za-z\s\(\)]{2,40}?)\s+(?:of|at|was|level)\s+([\d,]+\.?\d*)\s*([A-Za-z/%µ]+(?:/[A-Za-z]+)?)'
        matches = re.findall(narrative_pattern, text, re.IGNORECASE)
        for test_name, value, unit in matches:
            test_clean = test_name.strip()
            value_clean = value.replace(',', '')  # Remove commas
            # Filter out non-test names and common words
            words_to_exclude = ['level', 'value', 'result', 'was', 'noted', 'rise', 'increase', 'decrease', 'elevation', 'corresponding', 'significant', 'in', 'on', 'to', 'from', 'with', 'and', 'or', 'the', 'a', 'an']
            # Clean up test name by removing excluded words from beginning and end
            test_words = test_clean.split()
            # Remove excluded words from the beginning
            while test_words and test_words[0].lower() in words_to_exclude:
                test_words = test_words[1:]
            # Remove excluded words from the end
            while test_words and test_words[-1].lower() in words_to_exclude:
                test_words = test_words[:-1]
            test_clean = ' '.join(test_words)
            
            if len(test_clean) > 3 and test_clean.lower() not in words_to_exclude:
                if not any(nv['test_name'] == test_clean and nv['value'] == float(value_clean) 
                          for nv in numeric_values):
                    numeric_values.append({
                        "test_name": test_clean,
                        "value": float(value_clean),
                        "unit": unit.strip()
                    })
        
        return numeric_values
    
    def _compare_numeric_values(self, generated: str, ground_truth: str) -> Dict[str, Any]:
        """
        Extract and compare numeric values between generated and ground truth.
        
        Returns:
            Dict with:
            - numeric_values_gt: List of ground truth values
            - numeric_values_generated: List of generated values
            - numeric_matches: List of matched values with distances
            - mean_numeric_distance_percent: Average percentage distance
        """
        # Cache ground truth extraction — it's identical for every repetition of the same entry
        gt_cache_key = hashlib.md5(ground_truth.encode()).hexdigest()
        if gt_cache_key in self._gt_numeric_cache:
            gt_values_cached = self._gt_numeric_cache[gt_cache_key]
        else:
            gt_values_cached = None  # Will be populated below

        # Try LLM-based extraction (which normalizes consistently across both texts)
        if self.llm_client:
            try:
                if gt_values_cached is not None:
                    # Only extract generated — reuse cached GT values
                    gen_values = self._extract_numeric_with_llm(generated)
                    gt_values = gt_values_cached
                else:
                    gt_values, gen_values = self._extract_numeric_from_both_with_llm(ground_truth, generated)
                    self._gt_numeric_cache[gt_cache_key] = gt_values
            except Exception as e:
                print(f"⚠️  LLM dual extraction failed, falling back to individual extraction: {e}")
                gt_values = gt_values_cached or self.extract_numeric_values(ground_truth)
                if gt_values_cached is None:
                    self._gt_numeric_cache[gt_cache_key] = gt_values
                gen_values = self.extract_numeric_values(generated)
        else:
            # Fallback to regex-based extraction
            if gt_values_cached is not None:
                gt_values = gt_values_cached
            else:
                gt_values = self.extract_numeric_values(ground_truth, use_llm=False)
                self._gt_numeric_cache[gt_cache_key] = gt_values
            gen_values = self.extract_numeric_values(generated, use_llm=False)
        
        matches = []
        
        # Match by test name (case-insensitive, normalized)
        for gt in gt_values:
            gt_name_norm = self._normalize_test_name(gt['test_name'])
            
            # Find matching test in generated
            for gen in gen_values:
                gen_name_norm = self._normalize_test_name(gen['test_name'])
                
                # Check if test names match (allow partial match)
                if self._test_names_match(gt_name_norm, gen_name_norm):
                    # Determine the generated value to compare (possibly unit-converted)
                    gen_compare_value = gen['value']
                    unit_converted = False
                    
                    if not self._units_compatible(gt['unit'], gen['unit']):
                        # Units differ — try to convert before giving up
                        factor = self._get_unit_conversion_factor(gen['unit'], gt['unit'])
                        if factor is not None:
                            gen_compare_value = gen['value'] * factor
                            unit_converted = True
                            print(f"🔄  Unit conversion for '{gt['test_name']}': {gen['value']} {gen['unit']} → {gen_compare_value} {gt['unit']} (×{factor})")
                        else:
                            # Truly incompatible — skip this match
                            print(f"⚠️  Test '{gt['test_name']}' has incompatible units: {gt['unit']} vs {gen['unit']} (no conversion available)")
                            continue
                    
                    # Calculate absolute distance
                    distance_absolute = abs(gt['value'] - gen_compare_value)
                    
                    # Calculate percentage distance
                    if gt['value'] != 0:
                        distance_percent = distance_absolute / gt['value'] * 100
                    else:
                        distance_percent = 0 if gen_compare_value == 0 else 100
                    
                    matches.append({
                        "test": gt['test_name'],
                        "gt_value": gt['value'],
                        "gt_unit": gt['unit'],
                        "generated_value": gen['value'],
                        "generated_unit": gen['unit'],
                        "generated_value_converted": round(gen_compare_value, 6) if unit_converted else None,
                        "unit_converted": unit_converted,
                        "distance_absolute": round(distance_absolute, 4),
                        "distance_percent": round(distance_percent, 2),
                        "distance_normalized": self._compute_normalized_distance(
                            gt['test_name'], gt['value'], gen_compare_value
                        )
                    })
                    break
        
        # Calculate mean distance
        mean_distance = (
            sum(m['distance_percent'] for m in matches) / len(matches) 
            if matches else None
        )
        
        # Calculate mean normalized distance (only for matches with ref ranges)
        norm_values = [m['distance_normalized'] for m in matches 
                       if m['distance_normalized'] is not None]
        mean_normalized = (
            sum(norm_values) / len(norm_values)
            if norm_values else None
        )
        
        # Format detailed comparison text
        comparison_text = self._format_numeric_comparison(matches)
        
        return {
            "numeric_values_gt": gt_values,
            "numeric_values_generated": gen_values,
            "numeric_matches": matches,
            "mean_numeric_distance_percent": round(mean_distance, 2) if mean_distance is not None else None,
            "mean_numeric_distance_normalized": round(mean_normalized, 2) if mean_normalized is not None else None,
            "numeric_comparison_detailed": comparison_text
        }
    
    def _load_reference_ranges(self) -> Dict[str, Dict]:
        """Load clinical reference ranges from data/reference_ranges.json."""
        ref_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'reference_ranges.json'
        )
        ref_path = os.path.abspath(ref_path)
        if os.path.exists(ref_path):
            try:
                with open(ref_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"\u26a0\ufe0f  Could not load reference ranges: {e}")
        return {}
    
    def _find_reference_range(self, test_name: str) -> Optional[Tuple[float, float]]:
        """Find reference range for a test name, with fuzzy matching."""
        if not self._ref_ranges:
            return None
        test_lower = test_name.lower().strip()
        # Exact match
        if test_lower in self._ref_ranges:
            r = self._ref_ranges[test_lower]
            return (r['low'], r['high'])
        # Substring match
        for key, r in self._ref_ranges.items():
            if test_lower in key or key in test_lower:
                return (r['low'], r['high'])
        return None
    
    def _compute_normalized_distance(
        self, test_name: str, gt_value: float, gen_value: float
    ) -> Optional[float]:
        """
        Compute reference-range-normalized distance.
        
        Formula: |generated - GT| / (ref_high - ref_low) × 100
        
        This normalizes by the clinical reference range span, making distances
        comparable across tests with different scales (e.g., potassium 3.5-5.0
        vs platelet count 150K-450K).
        
        Returns None if no reference range is available for this test.
        """
        ref = self._find_reference_range(test_name)
        if ref is None:
            return None
        lo, hi = ref
        span = hi - lo
        if span <= 0:
            return None
        return round(abs(gen_value - gt_value) / span * 100, 2)
    
    def _format_numeric_comparison(self, matches: List[Dict]) -> str:
        """Format numeric matches as readable text."""
        if not matches:
            return "No numeric matches found"
        
        lines = []
        lines.append("--- NUMERIC COMPARISON ---")
        lines.append(f"Matches found: {len(matches)}")
        lines.append("")
        lines.append("Detailed Matches:")
        
        for match in matches:
            lines.append("")
            lines.append(f"  Test: {match['test']}")
            lines.append(f"    Ground Truth: {match['gt_value']} {match['gt_unit']}")
            lines.append(f"    Generated:    {match['generated_value']} {match['generated_unit']}")
            lines.append(f"    Distance:     {match['distance_absolute']}")
            lines.append(f"    Distance prc: {match['distance_percent']}%")
            if match.get('distance_normalized') is not None:
                lines.append(f"    Distance norm (ref-range): {match['distance_normalized']}%")
        
        return "\n".join(lines)
    
    def _normalize_unit(self, unit) -> str:
        """Normalize measurement units for comparison."""
        if unit is None:
            return ""
        unit_lower = str(unit).lower().strip()
        # Collapse whitespace (e.g. "mm hg" → "mmhg", "cm h2o" → "cmh2o")
        unit_lower = re.sub(r'\s+', '', unit_lower)
        # Strip unicode variants (µ → u, ² → 2, etc.)
        unit_lower = unit_lower.replace('µ', 'u').replace('μ', 'u')

        # Common equivalent units
        equivalents = {
            # Enzyme units
            'iu/l': 'u/l',
            'u/l': 'u/l',
            'iu/ml': 'u/ml',
            'u/ml': 'u/ml',
            'miu/l': 'mu/l',
            'mu/l': 'mu/l',

            # Pressure (blood gas, blood pressure)
            'mmhg': 'mmhg',
            'mm hg': 'mmhg',    # handled above by whitespace collapse, kept for clarity
            'torr': 'mmhg',
            # CSF pressure
            'cmh2o': 'cmh2o',
            'cm h2o': 'cmh2o',

            # Microgram variations
            'ug/dl': 'mcg/dl',
            'mcg/dl': 'mcg/dl',
            'ug/l': 'mcg/l',
            'mcg/l': 'mcg/l',
            'ng/ml': 'ng/ml',
            'ng/dl': 'ng/dl',

            # Milligram variations
            'mg/dl': 'mg/dl',
            'mg/l': 'mg/l',

            # Molar variations
            'mmol/l': 'mmol/l',
            'meq/l': 'mmol/l',   # equivalent for monovalent electrolytes (Na, K, Cl, HCO3)
            'umol/l': 'umol/l',
            'nmol/l': 'nmol/l',

            # Cell counts (collapse common written variants)
            '/ul': '/ul',
            '/microliter': '/ul',
            'permicroliter': '/ul',
            'per microliter': '/ul',
            'cells/ul': '/ul',
            'cells/mm3': '/ul',
            'x10^3/ul': 'x10^3/ul',
            'x10^3/ul': 'x10^3/ul',
            '×10^3/ul': 'x10^3/ul',
            '10^3/ul': 'x10^3/ul',
            'k/ul': 'x10^3/ul',
            'thou/ul': 'x10^3/ul',

            # Percentage
            '%': '%',
            'percent': '%',

            # Dimensionless / ratio
            '': '',
            'ratio': '',
            'index': '',
        }

        return equivalents.get(unit_lower, unit_lower)

    def _get_unit_conversion_factor(self, from_unit: str, to_unit: str) -> Optional[float]:
        """Return multiplier to convert a value from `from_unit` to `to_unit`.
        
        Returns None if the units are not convertible.
        Returns 1.0 if units are already equivalent.
        
        Example: _get_unit_conversion_factor('x10^3/ul', '/ul') → 1000.0
                 (multiply the generated value by 1000 to match GT scale)
        """
        n_from = self._normalize_unit(from_unit)
        n_to = self._normalize_unit(to_unit)
        
        # Already equivalent
        if n_from == n_to:
            return 1.0
        # Either side missing — treat as 1:1
        if n_from == "" or n_to == "":
            return 1.0
        
        # Build a directed conversion table:  (from_norm, to_norm) → factor
        # factor means: value_in_to_unit = value_in_from_unit × factor
        conversions = {
            # Cell counts:  ×10³/µL  ↔  /µL
            ('x10^3/ul', '/ul'):    1000.0,
            ('/ul', 'x10^3/ul'):    0.001,
            # Cell counts:  ×10⁶/µL  ↔  /µL  (RBC)
            ('x10^6/ul', '/ul'):    1_000_000.0,
            ('/ul', 'x10^6/ul'):    1e-6,
            # Cell counts:  ×10⁹/L  ↔  /µL  (SI WBC)
            ('x10^9/l', '/ul'):     1000.0,
            ('/ul', 'x10^9/l'):     0.001,
            # Cell counts:  ×10¹²/L  ↔  /µL  (SI RBC)
            ('x10^12/l', '/ul'):    1_000_000.0,
            ('/ul', 'x10^12/l'):    1e-6,
            # Mass concentration:  g/dL  ↔  mg/dL
            ('g/dl', 'mg/dl'):      1000.0,
            ('mg/dl', 'g/dl'):      0.001,
            # Mass concentration:  mg/L  ↔  mg/dL
            ('mg/l', 'mg/dl'):      0.1,
            ('mg/dl', 'mg/l'):      10.0,
            # Molar:  µmol/L  ↔  mmol/L
            ('umol/l', 'mmol/l'):   0.001,
            ('mmol/l', 'umol/l'):   1000.0,
            # Molar:  nmol/L  ↔  µmol/L
            ('nmol/l', 'umol/l'):   0.001,
            ('umol/l', 'nmol/l'):   1000.0,
            # Molar:  nmol/L  ↔  mmol/L
            ('nmol/l', 'mmol/l'):   1e-6,
            ('mmol/l', 'nmol/l'):   1e6,
            # Microgram:  µg/dL  ↔  ng/mL  (1 µg/dL = 10 ng/mL)
            ('mcg/dl', 'ng/ml'):    10.0,
            ('ng/ml', 'mcg/dl'):    0.1,
            # Microgram:  µg/L  ↔  ng/mL  (same thing)
            ('mcg/l', 'ng/ml'):     1.0,
            ('ng/ml', 'mcg/l'):     1.0,
        }
        
        return conversions.get((n_from, n_to))

    def _units_compatible(self, unit1, unit2: str) -> bool:
        """Check if two units are compatible (equivalent).
        
        If either unit is None or empty, we allow the match — the LLM sometimes
        omits units for dimensionless values (pH, ratios) and we should not
        discard valid numeric comparisons because of a missing unit.
        """
        n1 = self._normalize_unit(unit1)
        n2 = self._normalize_unit(unit2)
        # If either side has no unit, treat as compatible
        if n1 == "" or n2 == "":
            return True
        return n1 == n2
    
    def _normalize_test_name(self, name: str) -> str:
        """Normalize test name for matching."""
        # Remove special characters, lowercase, remove extra spaces
        normalized = re.sub(r'[^\w\s]', '', name.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _test_names_match(self, name1: str, name2: str) -> bool:
        """Check if two test names match (allowing partial matches)."""
        # Exact match
        if name1 == name2:
            return True
        
        # Check if one is substring of other (for abbreviated names)
        if name1 in name2 or name2 in name1:
            return True
        
        # Check word overlap (at least 50% of words match)
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        min_words = min(len(words1), len(words2))
        
        return overlap / min_words >= 0.5
    
    def _exact_match(self, generated: str, ground_truth: str) -> bool:
        """Check if generated matches ground truth exactly."""
        gen_clean = self._normalize_text(generated)
        gt_clean = self._normalize_text(ground_truth)
        return gen_clean == gt_clean
    
    def _text_similarity(self, generated: str, ground_truth: str) -> float:
        """Calculate text similarity using SequenceMatcher."""
        gen_clean = self._normalize_text(generated)
        gt_clean = self._normalize_text(ground_truth)
        return SequenceMatcher(None, gen_clean, gt_clean).ratio()
    
    def _length_ratio(self, generated: str, ground_truth: str) -> float:
        """Calculate ratio of generated length to ground truth length."""
        gen_len = len(generated.strip())
        gt_len = len(ground_truth.strip())
        if gt_len == 0:
            return 0.0
        return gen_len / gt_len
    
    def _contains_ground_truth(self, generated: str, ground_truth: str) -> bool:
        """Check if generated contains substantial portion of ground truth."""
        gen_clean = self._normalize_text(generated)
        gt_clean = self._normalize_text(ground_truth)
        
        # Check if ground truth (or most of it) appears in generated
        if len(gt_clean) < 20:
            return gt_clean in gen_clean
        
        # For longer ground truth, check if significant portion matches
        gt_words = gt_clean.split()
        gen_words_set = set(gen_clean.split())
        
        matching_words = sum(1 for word in gt_words if word in gen_words_set)
        match_ratio = matching_words / len(gt_words) if gt_words else 0
        
        return match_ratio > 0.7  # 70% of ground truth words present
    
    def _check_data_leakage(self, generated: str, ground_truth: str) -> bool:
        """
        Check if generated response contains exact match of ground truth.
        This indicates data leakage - the model saw and copied the ground truth.
        Uses multiple detection strategies:
        1. Exact normalized substring match
        2. High n-gram overlap (detects paraphrased copies) - using 3-grams and lower threshold
        3. High word overlap (detects semantic copying)
        4. High overall similarity (>0.9)
        """
        gen_clean = self._normalize_text(generated)
        gt_clean = self._normalize_text(ground_truth)
        
        # Strategy 1: Check for exact substring match
        if gt_clean in gen_clean:
            return True
        
        # Strategy 2: Check for high n-gram overlap using 3-grams (more flexible)
        gt_words = gt_clean.split()
        if len(gt_words) >= 3:
            # Generate 3-grams from ground truth
            trigrams = [' '.join(gt_words[i:i+3]) for i in range(len(gt_words) - 2)]
            
            # Check how many appear in generated
            matches = sum(1 for tg in trigrams if tg in gen_clean)
            overlap_ratio = matches / len(trigrams) if trigrams else 0
            
            # If >50% of 3-grams match, likely copying
            if overlap_ratio > 0.5:
                return True
        
        # Strategy 3: Check for high word-level overlap (detects semantic copying)
        # If most key words appear in same context, likely copied even if reworded
        if len(gt_words) >= 5:
            # Count unique content words (not stopwords)
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'was', 'is', 'are'}
            content_words = [w for w in gt_words if w not in stopwords and len(w) > 2]
            
            if content_words:
                # Check how many content words appear in generated
                matching = sum(1 for w in content_words if w in gen_clean)
                content_overlap = matching / len(content_words)
                
                # If >80% of content words present, likely semantic copy
                if content_overlap > 0.8:
                    return True
        
        # Strategy 4: Check for very high overall similarity
        return self._text_similarity(generated, ground_truth) > 0.9
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def _evaluate_imaging(self, generated: str, ground_truth: str) -> Dict[str, Any]:
        """Evaluate imaging-specific aspects."""
        gen_lower = generated.lower()
        gt_lower = ground_truth.lower()
        
        # Extract measurements from ground truth
        gt_measurements = re.findall(r'(\d+\.?\d*)\s*(cm|mm)', gt_lower)
        gen_measurements = re.findall(r'(\d+\.?\d*)\s*(cm|mm)', gen_lower)
        
        # Extract anatomical locations
        gt_locations = self._extract_anatomical_locations(gt_lower)
        gen_locations = self._extract_anatomical_locations(gen_lower)
        
        # Check if same location mentioned
        location_overlap = bool(gt_locations & gen_locations)
        
        # Check if measurements are similar (data leakage indicator)
        measurement_match = False
        if gt_measurements and gen_measurements:
            for gt_val, gt_unit in gt_measurements:
                for gen_val, gen_unit in gen_measurements:
                    if gt_unit == gen_unit and abs(float(gt_val) - float(gen_val)) < 0.5:
                        measurement_match = True
                        break
        
        return {
            "imaging_location_overlap": location_overlap,
            "imaging_measurement_exact_match": measurement_match,
            "ground_truth_measurements": gt_measurements,
            "generated_measurements": gen_measurements,
            "ground_truth_locations": list(gt_locations),
            "generated_locations": list(gen_locations)
        }
    
    def _extract_anatomical_locations(self, text: str) -> set:
        """Extract anatomical locations from text."""
        locations = set()
        
        # Common anatomical terms
        patterns = [
            r'\b(left|right)\s+(adrenal|kidney|lung|lobe|ventricle|atrium)\b',
            r'\b(upper|lower|middle)\s+(lobe|pole)\b',
            r'\b(anterior|posterior|lateral|medial)\b',
            r'\b(liver|spleen|pancreas|gallbladder)\b',
            r'\b(abdomen|pelvis|chest|head|neck)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    locations.add(' '.join(match))
                else:
                    locations.add(match)
        
        return locations
    
    def _evaluate_lab(self, generated: str, ground_truth: str) -> Dict[str, Any]:
        """Evaluate lab-specific aspects."""
        # Extract numeric values
        gt_values = re.findall(r'(\d+\.?\d*)', ground_truth)
        gen_values = re.findall(r'(\d+\.?\d*)', generated)
        
        # Check for exact value matches (data leakage)
        value_match = any(
            abs(float(gv) - float(gtv)) < 0.01
            for gv in gen_values
            for gtv in gt_values
        ) if gt_values and gen_values else False
        
        return {
            "lab_value_exact_match": value_match,
            "ground_truth_values": gt_values,
            "generated_values": gen_values
        }
    
    def _evaluate_with_llm(self, generated: str, ground_truth: str, category: str,
                           diagnosis: Optional[str] = None, case_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Use LLM to evaluate the generated response with 2 focused metrics:
        1. Ground Truth Similarity (semantic comparison) — separate prompt, has GT
        2. Clinical Plausibility (diagnosis correlation) — separate prompt, no GT

        Args:
            generated: Generated response text
            ground_truth: Ground truth response
            category: Test category (lab, imaging, physical_exam)
            diagnosis: Patient diagnosis for context
            case_text: Clinical case text for context

        Returns:
            Dictionary with LLM evaluation scores and reasoning (merged from both calls)
        """
        similarity_result = self._evaluate_similarity_with_llm(
            generated=generated,
            ground_truth=ground_truth,
            category=category,
            diagnosis=diagnosis,
        )
        plausibility_result = self._evaluate_plausibility_with_llm(
            generated=generated,
            category=category,
            diagnosis=diagnosis,
            case_text=case_text,
        )

        # Merge into the same output shape as before so nothing downstream breaks
        evaluation = {}

        # ground_truth_similarity comes from the similarity call
        if "ground_truth_similarity" in similarity_result:
            evaluation["ground_truth_similarity"] = similarity_result["ground_truth_similarity"]

        # clinical_plausibility comes from the plausibility call
        if "clinical_plausibility" in plausibility_result:
            evaluation["clinical_plausibility"] = plausibility_result["clinical_plausibility"]

        # data_leakage, summary, red_flags come from the similarity call
        for key in ("data_leakage", "summary", "red_flags"):
            if key in similarity_result:
                evaluation[key] = similarity_result[key]

        # Merge _metadata (combine token counts)
        sim_meta = similarity_result.get("_metadata", {})
        plaus_meta = plausibility_result.get("_metadata", {})
        evaluation["_metadata"] = {
            "model": sim_meta.get("model", plaus_meta.get("model", "unknown")),
            "tokens_used": (sim_meta.get("tokens_used", 0) or 0) + (plaus_meta.get("tokens_used", 0) or 0),
        }

        # Surface any errors from either call
        errors = []
        if "error" in similarity_result:
            errors.append(f"similarity: {similarity_result['error']}")
        if "error" in plausibility_result:
            errors.append(f"plausibility: {plausibility_result['error']}")
        if errors:
            evaluation["error"] = "; ".join(errors)

        return evaluation

    def _evaluate_similarity_with_llm(self, generated: str, ground_truth: str,
                                       category: str,
                                       diagnosis: Optional[str] = None) -> Dict[str, Any]:
        """
        Separate LLM call for Ground Truth Similarity + Data Leakage.
        Prompt is identical to the original combined prompt minus the
        clinical_plausibility block.
        """
        prompt = f"""You are a medical AI evaluator. Evaluate the generated test result using 2 specific metrics:

CONTEXT:
- Test Category: {category}
- Patient Diagnosis: {diagnosis or "Not provided"}

GROUND TRUTH TEST RESULT:
{ground_truth}

GENERATED TEST RESULT:
{generated}

EVALUATION METRICS:

1. GROUND TRUTH SIMILARITY (1-5)
   Assess how semantically similar the generated result is to the ground truth.
   - Do they describe the same type of test?
   - Do they report similar findings or observations?
   - Are the medical concepts aligned?
   - Score 5: Highly similar in meaning, captures same clinical information
   - Score 3: Moderately similar, some overlap but notable differences
   - Score 1: Very different, describes different tests or findings
   
   NOTE: This is about semantic similarity, not exact wording. The generated response 
   should convey the same clinical information even if phrased differently.

2. DATA LEAKAGE ASSESSMENT (true/false)
   Determine if the generated result appears to contain information COPIED from the ground truth,
   rather than being independently generated from clinical reasoning.
   - true = The generated response contains specific details (exact values, exact phrasing,
     unusual word choices) that strongly suggest it saw or copied the ground truth
   - false = The generated response appears independently produced from clinical reasoning
   
   IMPORTANT DISTINCTIONS:
   - Sharing the TEST NAME (e.g. "CT angiography of head and neck") is NOT leakage — the test name
     is part of the request, so it will naturally appear in both GT and generated response
   - Arriving at the same CLINICAL CONCLUSION through reasoning is NOT leakage (e.g. generating
     "diminished strength in right arm" for hemiplegic migraine is expected clinical reasoning)
   - Leakage is when specific UNIQUE details appear that could only come from seeing the GT:
     exact numeric values, unusual phrasing, specific measurements, word-for-word copying
   - A normal/unremarkable result matching a normal GT is NOT leakage — it's correct reasoning

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown, no code blocks):
{{
    "ground_truth_similarity": {{
        "score": <1-5>,
        "reasoning": "<brief explanation>"
    }},
    "data_leakage": {{
        "leaked": <true or false>,
        "reasoning": "<brief explanation>"
    }},
    "summary": "<one sentence overall assessment>",
    "red_flags": ["<list any serious concerns>"] or []
}}

Be thorough but concise. Focus on objective medical quality assessment."""

        try:
            response = self.llm_client.text_completion(
                prompt=prompt,
                temperature=0.1,
                max_tokens=5000
            )

            response_text = response["text"].strip()

            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace("```json", "").replace("```", "").strip()

            evaluation = json.loads(response_text)

            evaluation["_metadata"] = {
                "model": response.get("model", "unknown"),
                "tokens_used": response.get("usage", {}).get("total_tokens", 0)
            }

            return evaluation

        except json.JSONDecodeError as e:
            print(f"Failed to parse similarity LLM response: {e}")
            return {"error": "Failed to parse LLM response"}
        except Exception as e:
            print(f"Similarity LLM evaluation failed: {e}")
            return {"error": str(e)}

    def _evaluate_plausibility_with_llm(self, generated: str, category: str,
                                         diagnosis: Optional[str] = None,
                                         case_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Separate LLM call for Clinical Plausibility only.
        No ground truth is provided — the judge sees only the generated result,
        the diagnosis, and the full case context (same-category tests masked).
        Prompt is identical to the original combined prompt minus the
        ground_truth_similarity block, data_leakage block, and GROUND TRUTH section.
        """
        case_context_section = ""
        if case_text:
            case_context_section = f"\nCASE CONTEXT (other clinical data for this patient):\n{case_text}\n"

        prompt = f"""You are a medical AI evaluator. Evaluate the generated test result using 1 specific metric:

CONTEXT:
- Test Category: {category}
- Patient Diagnosis: {diagnosis or "Not provided"}
{case_context_section}
GENERATED TEST RESULT:
{generated}

EVALUATION METRICS:

1. CLINICAL PLAUSIBILITY (1-5)
   Assess if the generated result is medically appropriate for the diagnosis.
   - Would a patient with "{diagnosis}" realistically have these test results?
   - Are the findings clinically consistent with the diagnosis?
   - Is the test appropriate for investigating this condition?
   - Score 5: Highly plausible, findings strongly consistent with diagnosis
   - Score 3: Somewhat plausible, could occur but not typical
   - Score 1: Implausible or medically inconsistent with diagnosis

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown, no code blocks):
{{
    "clinical_plausibility": {{
        "score": <1-5>,
        "reasoning": "<brief explanation>"
    }}
}}

Be thorough but concise. Focus on objective medical quality assessment."""

        try:
            response = self.llm_client.text_completion(
                prompt=prompt,
                temperature=0.1,
                max_tokens=5000
            )

            response_text = response["text"].strip()

            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
                response_text = response_text.replace("```json", "").replace("```", "").strip()

            evaluation = json.loads(response_text)

            evaluation["_metadata"] = {
                "model": response.get("model", "unknown"),
                "tokens_used": response.get("usage", {}).get("total_tokens", 0)
            }

            return evaluation

        except json.JSONDecodeError as e:
            print(f"Failed to parse plausibility LLM response: {e}")
            return {"error": "Failed to parse LLM response"}
        except Exception as e:
            print(f"Plausibility LLM evaluation failed: {e}")
            return {"error": str(e)}
    
    def summarize_results(self, results: list) -> Dict[str, Any]:
        """
        Summarize evaluation results across multiple samples.
        
        Args:
            results: List of evaluation result dictionaries
            
        Returns:
            Summary statistics
        """
        if not results:
            return {}
        
        total = len(results)
        
        summary = {
            "total_samples": total,
            "exact_matches": sum(1 for r in results if r.get("exact_match")),
            "data_leakage_detected": sum(1 for r in results if r.get("data_leakage_detected")),
            "contains_ground_truth": sum(1 for r in results if r.get("contains_ground_truth")),
            "avg_length_ratio": sum(r.get("length_ratio", 0) for r in results) / total,
        }
        
        # Numeric value comparison summary
        numeric_results = [r for r in results if r.get("mean_numeric_distance_percent") is not None]
        if numeric_results:
            summary["numeric_analysis"] = {
                "samples_with_numeric_values": len(numeric_results),
                "avg_numeric_distance_percent": sum(
                    r["mean_numeric_distance_percent"] for r in numeric_results
                ) / len(numeric_results),
                "total_numeric_matches": sum(
                    len(r.get("numeric_matches", [])) for r in results
                )
            }
        
        # Category-specific summaries
        imaging_results = [r for r in results if r.get("imaging_location_overlap") is not None]
        if imaging_results:
            summary["imaging"] = {
                "samples": len(imaging_results),
                "location_overlap": sum(1 for r in imaging_results if r.get("imaging_location_overlap")),
                "measurement_exact_match": sum(1 for r in imaging_results if r.get("imaging_measurement_exact_match"))
            }
        
        lab_results = [r for r in results if r.get("lab_value_exact_match") is not None]
        if lab_results:
            summary["lab"] = {
                "samples": len(lab_results),
                "value_exact_match": sum(1 for r in lab_results if r.get("lab_value_exact_match"))
            }
        
        # LLM evaluation summaries (2 metrics only)
        llm_results = [r for r in results if r.get("llm_evaluation") and not r["llm_evaluation"].get("error")]
        if llm_results:
            llm_summary = {
                "samples_evaluated": len(llm_results),
                "avg_ground_truth_similarity": sum(
                    r["llm_evaluation"]["ground_truth_similarity"]["score"] 
                    for r in llm_results
                ) / len(llm_results),
                "avg_clinical_plausibility": sum(
                    r["llm_evaluation"]["clinical_plausibility"]["score"] 
                    for r in llm_results
                ) / len(llm_results),
                "samples_with_red_flags": sum(
                    1 for r in llm_results 
                    if r["llm_evaluation"].get("red_flags")
                ),
                "llm_data_leakage_count": sum(
                    1 for r in llm_results
                    if r["llm_evaluation"].get("data_leakage", {}).get("leaked", False)
                )
            }
            llm_summary["llm_data_leakage_rate"] = llm_summary["llm_data_leakage_count"] / len(llm_results) * 100 if llm_results else 0
            summary["llm_evaluation"] = llm_summary
        
        # Calculate percentages
        summary["exact_match_rate"] = summary["exact_matches"] / total * 100
        summary["data_leakage_rate"] = summary["data_leakage_detected"] / total * 100
        summary["ground_truth_containment_rate"] = summary["contains_ground_truth"] / total * 100
        
        return summary
