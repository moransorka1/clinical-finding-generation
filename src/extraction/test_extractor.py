"""
Extract specific test/procedure names from clinical_data content using LLM.
"""
import json
import re
from typing import Dict, Any, List, Optional, Tuple


class TestExtractor:
    """Extract the specific test or procedure name from clinical data content."""
    
    # Regex patterns for units commonly found in clinical GT text
    _UNIT_PATTERNS = [
        # SI / lab units with slash notation: mmol/L, mg/dL, g/dL, mEq/L, ng/mL, etc.
        r'(?:per\s+)?(\w+/[a-zA-Zµμ]+)',
        # "per microliter", "per liter", "per deciliter", "per milliliter"
        r'(per\s+(?:micro)?liter|per\s+deciliter|per\s+milliliter)',
        # Percentage
        r'(\d[\d.]*\s*%)',
        # Pressure: mm Hg, cmH2O
        r'(mm\s*Hg|cmH2O|cm\s*H2O)',
        # Titer notation: 1:320
        r'(1:\d+)',
        # IU/L, U/L, mIU/L
        r'([mMIi]*[UuIi]/[a-zA-Z]+)',
        # Cells notation: cells/µL, cells/mm³
        r'(cells?/[a-zA-Zµμ³]+)',
        # ×10³/µL, ×10⁹/L, x10^3/uL etc.
        r'([×x]\s*10[\^³⁹⁶]*\s*/\s*[a-zA-Zµμ]+)',
    ]
    
    # Result qualifiers that should NEVER appear in a test/procedure request
    _RESULT_QUALIFIERS = re.compile(
        r'\b(?:normal|negative|positive|undetectable|elevated|decreased|'
        r'low|high|intact|grossly\s+intact|absent|present|unremarkable|'
        r'abnormal|confirmed|confirming|patent|stable|unchanged|suspicious|'
        r'trace|reactive|nonreactive|indeterminate)\b',
        re.IGNORECASE
    )
    
    def __init__(self, llm_client=None):
        """Initialize the test extractor with LLM client."""
        self.llm_client = llm_client
    
    def extract_test_name(self, content: str, category: str) -> str:
        """
        Extract the exact test/procedure that was performed from the clinical data content.
        
        Args:
            content: The clinical data content (e.g., "CT of the abdomen and pelvis revealed...")
            category: The category (imaging, laboratory_tests, procedures)
        
        Returns:
            The extracted test name (e.g., "CT of the abdomen and pelvis")
        """
        prompt = f"""Extract the EXACT and COMPLETE test or procedure name from the following clinical data.
Include ALL specific details about HOW the test was performed based on what the results indicate.

Category: {category}
Content: {content}

IMPORTANT: Be SPECIFIC about:
- For CSF analysis: specify method (lumbar puncture, ventricular tap, etc.)
- For CT/MRI: specify contrast usage (with contrast, without contrast, with and without contrast)
- For blood tests: specify the exact panel or specific test
- For procedures: specify the approach or technique

Return ONLY the complete, specific test/procedure name as it should be ordered.

Examples:
- "CT of the abdomen and pelvis revealed a left adrenal nodule" → "CT of the abdomen and pelvis without contrast"
- "CT head with contrast showed enhancement" → "CT of the head with contrast"
- "MRI brain with and without contrast demonstrated..." → "MRI of the brain with and without contrast"
- "CSF analysis via lumbar puncture showed 200 WBC" → "CSF analysis via lumbar puncture"
- "Complete blood count showed hemoglobin 8.2 g/dL" → "Complete blood count"
- "Serum electrolytes revealed sodium 128" → "Serum electrolytes"

Extract the complete, specific test/procedure name:"""

        try:
            if self.llm_client:
                response = self.llm_client.text_completion(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=1000
                )
                test_name = response['text'].strip()
                return test_name
            else:
                # No LLM client, use simple extraction
                return self._simple_extract(content, category)
        except Exception as e:
            print(f"⚠️  Error extracting test name: {e}")
            # Fallback: try simple extraction
            return self._simple_extract(content, category)
    
    def _simple_extract(self, content: str, category: str) -> str:
        """Fallback simple extraction without LLM."""
        # Try to get text before common result indicators
        indicators = [
            ' revealed', ' showed', ' demonstrated', ' indicated', ' found',
            ' was ', ' were ', ' shows ', ' show ', ' appeared ',
            ' confirmed ', ' confirming ',
        ]
        
        content_lower = content.lower()
        for indicator in indicators:
            if indicator in content_lower:
                idx = content_lower.index(indicator)
                return self._clean_test_name(content[:idx].strip())
        
        # No indicator found — strip result qualifiers from the full text
        cleaned = self._clean_test_name(content)
        
        # If still very long, return first sentence
        if len(cleaned) > 200:
            sentences = cleaned.split('.')
            return sentences[0].strip()
        
        return cleaned

    def _clean_test_name(self, name: str) -> str:
        """Remove result values and qualitative outcomes from an extracted test name."""
        # Remove result qualifiers (normal, negative, undetectable, etc.)
        cleaned = self._RESULT_QUALIFIERS.sub('', name)
        # Remove standalone numeric values with optional units that look like results
        # e.g., "Hemoglobin 8.5 g/dl" → "Hemoglobin"
        cleaned = re.sub(
            r'\s+<?[\d,.]+\s*(?:%|g/d[lL]|mg/d[lL]|mmol/[lL](?:iter)?|ng/[a-zA-Z]+|'
            r'mEq/[lL]|U/[lL](?:iter)?|IU/[lL]|per\s+\S+|/[µμu]?[lLmM]|mm/hr|'
            r'mm\s*Hg|sec|cm|mm)\b',
            '', cleaned
        )
        # Remove orphaned numeric values (e.g., "Hemoglobin 8" or "sodium 128")
        cleaned = re.sub(r'\s+<?[\d,.]+(?:\s*$|\s*,)', ',', cleaned)
        # Clean up punctuation artifacts
        cleaned = re.sub(r',\s*,+', ',', cleaned)        # double commas
        cleaned = re.sub(r'\s*,\s*$', '', cleaned)        # trailing comma
        cleaned = re.sub(r'^\s*,\s*', '', cleaned)        # leading comma
        cleaned = re.sub(r'\s*;\s*$', '', cleaned)        # trailing semicolon
        cleaned = re.sub(r'^\s*;\s*', '', cleaned)        # leading semicolon
        cleaned = re.sub(r'\s{2,}', ' ', cleaned)         # multiple spaces
        cleaned = cleaned.strip().strip(',').strip(';').strip()
        return cleaned if cleaned else name  # fallback to original if over-cleaned

    def extract_units(self, content: str, category: str) -> List[str]:
        """
        Extract measurement units mentioned in ground-truth content.
        
        Uses regex to pull units like 'mmol/L', 'mg/dL', 'per microliter', '%', 'mm Hg'
        directly from the GT text. No LLM call needed — units are explicit in clinical text.
        
        Args:
            content: The ground-truth raw_content string
            category: The category (imaging, laboratory_tests, etc.)
            
        Returns:
            Deduplicated list of unit strings found, e.g. ['mmol/L', '%', 'per microliter']
        """
        if category not in ('laboratory_tests', 'physical_examination_or_assessments'):
            return []
        
        units_found: List[str] = []
        seen_lower: set = set()
        
        for pattern in self._UNIT_PATTERNS:
            for match in re.finditer(pattern, content):
                unit = match.group(1) if match.lastindex else match.group(0)
                unit = unit.strip()
                # Skip pure numbers that accidentally matched the % pattern
                if unit.replace('.', '').replace('%', '').strip().isdigit():
                    unit = '%'
                # Deduplicate (case-insensitive)
                if unit.lower() not in seen_lower and len(unit) > 0:
                    seen_lower.add(unit.lower())
                    units_found.append(unit)
        
        return units_found

    def extract_test_name_and_units(self, content: str, category: str) -> Tuple[str, List[str]]:
        """
        Extract both test name and measurement units from GT content in a single LLM call.
        
        Args:
            content: The ground-truth raw_content string
            category: The category
            
        Returns:
            Tuple of (test_name, units_list)
        """
        prompt = f"""You are a clinical lab order interpreter. Given clinical data containing test results, extract each individual test/procedure as a structured list.

Category: {category}
Content: {content}

RULES:
1. Return a JSON object with a "tests" array. Each element has "name" (what a doctor would order) and "unit" (measurement unit, or null if none).
2. Test names must NEVER contain numeric values or result qualifiers (normal, negative, positive, undetectable, elevated, decreased, intact, absent, present, unremarkable, patent, confirmed, etc.).
3. For imaging: include contrast info. For CSF: include collection method.
4. For physical exams: describe what was ASSESSED, not the findings.
5. Extract EVERY test mentioned in the content — do not skip any.

Return this exact JSON structure:
{{"tests": [{{"name": "test name", "unit": "unit or null"}}]}}

Examples:

Content: "Hemoglobin 8.5 g/dl, Hematocrit 25.4%, White-cell count 16,200 per μl, Platelet count 5000 per μl, Sodium 138 mmol/liter, Creatinine 1.07 mg/dl, LDH 1495 U/liter, Reticulocytes 5.8%"
Output: {{"tests": [{{"name": "Hemoglobin", "unit": "g/dl"}}, {{"name": "Hematocrit", "unit": "%"}}, {{"name": "White blood cell count", "unit": "per μl"}}, {{"name": "Platelet count", "unit": "per μl"}}, {{"name": "Sodium", "unit": "mmol/liter"}}, {{"name": "Creatinine", "unit": "mg/dl"}}, {{"name": "Lactate dehydrogenase", "unit": "U/liter"}}, {{"name": "Reticulocytes", "unit": "%"}}]}}

Content: "Blood ethanol level undetectable, urine toxicologic screening negative, screening for human chorionic gonadotropin negative, blood thyrotropin level normal"
Output: {{"tests": [{{"name": "Blood ethanol level", "unit": null}}, {{"name": "Urine toxicologic screening", "unit": null}}, {{"name": "Human chorionic gonadotropin screening", "unit": null}}, {{"name": "Blood thyrotropin level", "unit": null}}]}}

Content: "CT angiography of the head and neck showed no evidence of stenosis or occlusion"
Output: {{"tests": [{{"name": "CT angiography of the head and neck with contrast", "unit": null}}]}}

Content: "CSF analysis revealed 67 nucleated cells per microliter, 48% neutrophils, protein 55 mg/dL"
Output: {{"tests": [{{"name": "CSF analysis via lumbar puncture: nucleated cells", "unit": "per microliter"}}, {{"name": "CSF analysis via lumbar puncture: neutrophils", "unit": "%"}}, {{"name": "CSF analysis via lumbar puncture: protein", "unit": "mg/dL"}}]}}

Content: "Peripheral-blood smear showed numerous schistocytes per high-power field, very few platelets, and reticulocytosis"
Output: {{"tests": [{{"name": "Peripheral-blood smear", "unit": "per high-power field"}}]}}

Content: "Strength and sensation appeared grossly intact, but could not perform rapidly alternating movements or tests of coordination"
Output: {{"tests": [{{"name": "Neurological examination: motor strength, sensation, coordination, and rapid alternating movements", "unit": null}}]}}

Content: "Aphasic; speech described as mumbling, able to answer simple questions intermittently"
Output: {{"tests": [{{"name": "Aphasia and speech assessment", "unit": null}}]}}

Return ONLY the JSON object:"""

        try:
            if self.llm_client:
                response = self.llm_client.text_completion(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=4000
                )
                response_text = response['text'].strip()
                
                # Parse JSON response
                # Strip markdown code blocks if present
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].split('```')[0].strip()
                
                # Fix common LLM JSON errors before parsing
                # 1. Trailing commas before ] or }
                response_text = re.sub(r',\s*([}\]])', r'\1', response_text)
                # 2. Single quotes → double quotes (but not inside values)
                # 3. Unquoted null/true/false are valid JSON, leave them
                
                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError:
                    # Last resort: try to extract the JSON object with a regex
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        cleaned = re.sub(r',\s*([}\]])', r'\1', json_match.group(0))
                        result = json.loads(cleaned)
                    else:
                        raise
                tests = result.get('tests', [])
                
                if not tests:
                    raise ValueError("Empty tests array in response")
                
                # Build test_name: join all unique test names
                names = []
                seen_names = set()
                units = []
                seen_units = set()
                
                for t in tests:
                    name = t.get('name', '').strip()
                    unit = t.get('unit')
                    
                    if name and name.lower() not in seen_names:
                        seen_names.add(name.lower())
                        names.append(name)
                    
                    if unit and str(unit).strip().lower() not in seen_units and str(unit).lower() != 'null':
                        seen_units.add(str(unit).strip().lower())
                        units.append(str(unit).strip())
                
                test_name = ', '.join(names) if names else ''
                
                if test_name:
                    return test_name, units
                else:
                    raise ValueError("No test names extracted")
            else:
                # No LLM — fall back to simple extraction + regex units
                test_name = self._simple_extract(content, category)
                units = self.extract_units(content, category)
                return test_name, units
                
        except Exception as e:
            print(f"⚠️  Error in extract_test_name_and_units: {e}")
            # Fallback: use individual methods
            test_name = self._simple_extract(content, category)
            units = self.extract_units(content, category)
            return test_name, units
