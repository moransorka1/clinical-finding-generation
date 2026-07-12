"""
Synthetic Response Generator for CliniClue
Generates realistic responses when requested information is not explicitly available in the case data.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException

# Use local database models
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.models.case import Case, FinalDiagnosis

logger = logging.getLogger(__name__)


class SyntheticResponseGenerator:
    def __init__(self, db: Session, llm_client):
        self.db = db
        self.llm_client = llm_client

    async def generate_synthetic_response(
        self,
        user_query: str,
        case_id: int,
        chat_history: List[Dict[str, Any]],
        partial_response: Optional[str] = None,
        cpt_codes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a synthetic response when requested information is not available.
        
        TWO-STEP APPROACH:
        1. Ask LLM what results would occur for this patient's diagnosis
        2. Format and sanitize the response to remove diagnostic conclusions
        
        Args:
            user_query: The original user query
            case_id: The case ID
            chat_history: List of previous chat messages
            partial_response: Existing partial response to enhance (optional)
            cpt_codes: CPT codes and descriptions to guide response content (optional)
            
        Returns:
            Dict containing assistant_response and formatting_type
        """
        try:
            # Get case text and final diagnosis
            case = self.db.query(Case).filter(Case.id == case_id).first()
            if not case:
                raise HTTPException(status_code=404, detail="Case not found")
            
            case_text = str(case.text)
            
            # Get final diagnosis
            case_diagnosis = self.db.query(FinalDiagnosis).filter(FinalDiagnosis.case_id == case_id).first()
            if not case_diagnosis:
                raise HTTPException(status_code=404, detail="Case diagnosis not found")
            
            diagnosis_info = {
                "primary_diagnosis": case_diagnosis.primary_diagnosis,
                "supporting_evidence": case_diagnosis.supporting_evidence,
                "differential_diagnoses": case_diagnosis.differential_diagnoses
            }
            
            # STEP 1: Generate realistic results for this diagnosis
            logger.info(f"🔬 STEP 1: Generating realistic results for diagnosis...")
            raw_results = await self._step1_generate_realistic_results(
                user_query=user_query,
                diagnosis_info=diagnosis_info,
                case_text=case_text,
                chat_history=chat_history,
                cpt_codes=cpt_codes
            )
            
            # STEP 2: Format and sanitize the results
            logger.info(f"🧹 STEP 2: Formatting and sanitizing results...")
            final_response = await self._step2_format_and_sanitize(
                raw_results=raw_results,
                user_query=user_query,
                diagnosis_info=diagnosis_info,
                cpt_codes=cpt_codes
            )
            
            # Validation removed - ChatResponseValidator 3-stage pipeline handles all validation
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating synthetic response: {str(e)}")
            # Force a retry rather than provide a rule-violating fallback
            raise HTTPException(status_code=500, detail="Error generating response - please try again")

    async def _step1_generate_realistic_results(
        self,
        user_query: str,
        diagnosis_info: Dict[str, Any],
        case_text: str,
        chat_history: List[Dict[str, Any]],
        cpt_codes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        STEP 1: Generate realistic test results for a patient with this diagnosis.
        
        This step explicitly asks: "If a patient has [diagnosis], what would their
        [test/imaging/exam] results show?"
        
        Returns raw medical findings without sanitization.
        """
        # Get previous lab values for consistency
        previous_labs = self._extract_previous_lab_values(chat_history)
        previous_labs_text = ""
        if previous_labs:
            previous_labs_text = f"\n\nPREVIOUSLY REVEALED VALUES (MAINTAIN CONSISTENCY):\n{previous_labs}"
        
        # Format CPT codes to understand what tests are being requested
        cpt_guidance = self._format_cpt_guidance(cpt_codes) if cpt_codes else ""

        prompt = f"""You are a medical pathophysiology expert. Generate realistic test results for a specific diagnosis.

PATIENT'S ACTUAL DIAGNOSIS:
{diagnosis_info['primary_diagnosis']}

CASE CONTEXT (full case report for clinical context):
{case_text}

RECENT CHAT HISTORY:{previous_labs_text}

USER REQUEST:
"{user_query}"

{cpt_guidance}

YOUR TASK:
Generate realistic, medically accurate results that a patient with the mentioned conditions would have for the requested test/imaging/examination.

GENERATION GUIDELINES:

1. Pathophysiological Accuracy — values consistent with the patient condition and disease stage/severity

2. Test-Specific Results — lab values with units, imaging findings, physical exam scoped strictly

3. Strict Scope Limitation — ONLY generate what is explicitly in USER REQUEST

4. Include Diagnostic Markers — show what would ACTUALLY be seen (e.g. "Positive" for HIV test)

5. Realistic Reference Ranges — include ranges, flag abnormals

OUTPUT FORMAT (JSON):
{{
  "test_type": "lab_results|imaging|physical_exam|specialized_test",
  "diagnosis_being_modeled": "...",
  "raw_findings": {{
    "finding_1": {{"value": "...", "unit": "...", "reference_range": "..."}},
    ...
  }},
  "pathophysiology_note": "internal use only"
}}"""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.llm_client.chat_completion(
            messages=messages,
            temperature=0.3,  # Lower temperature for medical accuracy
            max_tokens=6000  # Increased to prevent truncation
        )
        
        response_text = response.get('text', '') if isinstance(response, dict) else str(response)
        logger.info(f"📋 STEP 1 Raw Results: {response_text[:500]}...")
        
        # Parse JSON response
        return self._parse_json_response(response_text)

    async def _step2_format_and_sanitize(
        self,
        raw_results: Dict[str, Any],
        user_query: str,
        diagnosis_info: Dict[str, Any],
        cpt_codes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        STEP 2: Format the raw results and remove diagnostic conclusions.
        
        This step:
        - Formats results in the appropriate style (table/list/narrative)
        - Removes phrases like "consistent with", "diagnostic of", "indicates"
        - Keeps direct findings like "Positive for HIV" or "tumor seen in imaging"
        - Removes disease name UNLESS it's part of a test name (e.g., "Anti-NMDA receptor")
        """
        
        prompt = f"""You are formatting medical test results for presentation to a clinician.

RAW FINDINGS (from medical simulation):
{json.dumps(raw_results, indent=2)}

USER'S ORIGINAL REQUEST:
"{user_query}"

DIAGNOSIS TO PROTECT (remove mentions of this):
{diagnosis_info['primary_diagnosis']}

🎯 YOUR TASK:
Format these findings into a professional clinical response while removing diagnostic conclusions.

FORMATTING RULES:

1. **Keep Objective Findings**:
   ✅ "Anti-NMDA receptor antibodies: Positive (titer 1:320)"
   ✅ "HIV antibody: Positive"
   ✅ "CT shows 3.2cm mass in right upper lobe"
   ✅ "WBC: 18,500/μL (Reference: 4,000-11,000/μL)"

2. **Remove Diagnostic Conclusions**:
   ❌ "Results consistent with NMDAR encephalitis"
   ❌ "Findings indicate HIV infection"
   ❌ "Suggestive of bacterial pneumonia"
   ❌ "Diagnostic of [condition]"

3. **Handle Test Names with Disease Terms**:
   ✅ "Anti-NMDA receptor antibodies: Positive" (test name includes "NMDA")
   ✅ "Lyme antibody: Positive" (test name is "Lyme antibody")
   ❌ "Patient has Lyme disease" (diagnostic conclusion)

4. **Imaging Findings**:
   ✅ Include anatomical findings ("mass", "lesion", "consolidation", "hemorrhage")
   ✅ Include measurements and locations
   ❌ Do NOT add "consistent with [diagnosis]" or "typical of [disease]"

5. **Choose Format Type**:
   - "table" → Lab results with multiple values (use | separators and \\n for newlines)
   - "list" → Multiple items or findings
   - "narrative" → Imaging descriptions, exam findings

TABLE FORMAT RULES (CRITICAL):
- Use pipe separators: Test | Result | Reference Range
- Use \\n for row breaks (literal backslash-n, not actual newlines)
- Include header: Test | Result | Reference Range\\n--- | --- | ---\\n
- Example: "Test | Result | Reference Range\\n--- | --- | ---\\nWBC | 18,500/μL | 4,000-11,000/μL\\nHemoglobin | 11.2 g/dL | 12.0-15.5 g/dL"

OUTPUT FORMAT (JSON):
{{
  "assistant_response": "formatted response text (with \\n for newlines if table)",
  "formatting_type": "table"|"list"|"narrative",
  "sanitization_log": ["removed phrase 1", "removed phrase 2"]
}}

EXAMPLES:

GOOD - Lab Results (table format):
{{
  "assistant_response": "Test | Result | Reference Range\\n--- | --- | ---\\nWBC | 18,500/μL | 4,000-11,000/μL\\nNeutrophils | 85% | 40-70%\\nHemoglobin | 11.2 g/dL | 12.0-15.5 g/dL",
  "formatting_type": "table"
}}

GOOD - Antibody Test (narrative format):
{{
  "assistant_response": "Anti-NMDA receptor antibodies:\\n- Serum: Positive (titer 1:320)\\n- CSF: Positive (titer 1:80)",
  "formatting_type": "list"
}}

GOOD - Imaging (narrative format):
{{
  "assistant_response": "CT of the chest shows a 3.2 cm mass in the right upper lobe with irregular margins. No significant mediastinal lymphadenopathy. Small right pleural effusion present.",
  "formatting_type": "narrative"
}}

BAD - Reveals diagnosis:
{{
  "assistant_response": "Results are consistent with HIV infection.",
  "formatting_type": "narrative"
}}

Format the findings now, removing any diagnostic conclusions while keeping objective results."""

        messages = [{"role": "user", "content": prompt}]
        
        response = self.llm_client.chat_completion(
            messages=messages,
            temperature=0.1,  # Very low temperature for precise formatting
            max_tokens=6000  # Increased to prevent truncation
        )
        
        response_text = response.get('text', '') if isinstance(response, dict) else str(response)
        logger.info(f"📋 STEP 2 Sanitized Response: {response_text[:500]}...")
        
        # Parse JSON response
        result = self._parse_json_response(response_text)
        
        # Log sanitization
        if result.get('sanitization_log'):
            logger.info(f"🧹 Removed phrases: {result['sanitization_log']}")
        
        return {
            "assistant_response": result.get('assistant_response', ''),
            "formatting_type": result.get('formatting_type', 'narrative')
        }

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks and truncation."""
        try:
            # Remove markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            response_text = response_text.strip()
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            
            # CRITICAL FIX: Try to extract partial JSON using recovery strategies
            try:
                cleaned_text = response_text.strip()
                if '```json' in cleaned_text:
                    cleaned_text = cleaned_text.split('```json')[1].split('```')[0]
                elif '```' in cleaned_text:
                    cleaned_text = cleaned_text.split('```')[1].split('```')[0]
                cleaned_text = cleaned_text.strip()
                
                # Try to fix common truncation issues by closing unclosed structures
                if any(key in cleaned_text for key in ['test_type', 'raw_findings', 'assistant_response']):
                    # Try multiple closing patterns
                    attempts = [
                        cleaned_text,  # Original
                        cleaned_text + '"}}',  # Close string + 2 levels (most common)
                        cleaned_text + '"}}}',  # Close string + 3 levels
                        cleaned_text + '"}',  # Close string + 1 level
                        cleaned_text + '}}',  # Close 2 levels
                        cleaned_text + '}}}',  # Close 3 levels
                        cleaned_text + '}',  # Close 1 level
                        cleaned_text.rstrip(',') + '}}',  # Remove trailing comma and close
                        cleaned_text.rstrip(',') + '}',  # Remove trailing comma and close once
                    ]
                    
                    for attempt in attempts:
                        try:
                            result = json.loads(attempt)
                            logger.warning(f"✅ Recovered partial JSON by closing unclosed structures")
                            return result
                        except:
                            continue
                
                logger.error(f"❌ Could not recover partial JSON - returning empty dict")
                logger.error(f"   Truncated text: {cleaned_text[-200:] if len(cleaned_text) > 200 else cleaned_text}")
                return {}
            except Exception as recovery_error:
                logger.error(f"❌ JSON recovery failed: {recovery_error}")
                return {}


    def _format_cpt_guidance(self, cpt_codes: Optional[Dict[str, Any]]) -> str:
        """Format CPT codes into guidance for synthetic response generation."""
        if not cpt_codes or not cpt_codes.get('selected_codes'):
            return "No specific CPT billing codes provided - generate standard response based on user query."
        
        guidance = "The following CPT codes have been selected for billing this requests:\n"
        
        for request, code_info in cpt_codes.get('selected_codes', {}).items():
            cpt_code = code_info.get('cpt_code', 'Unknown')
            description = code_info.get('description', 'No description')
            price = code_info.get('price', 0)
            
            guidance += f"\n• Medical Request: '{request}'\n"
            guidance += f"  CPT Code: {cpt_code}\n"
            guidance += f"  🔍 PROCEDURE DESCRIPTION: {description}\n"
            guidance += f"  Price: ${price:.2f}\n"
            
            # Emphasize using the description for content generation
            guidance += f"\n  🎯 GENERATE CONTENT FOR: {description}\n"
            guidance += f"  → The CPT description above tells you EXACTLY what medical procedure/test was performed\n"
            guidance += f"  → Generate findings/results specific to this procedure only\n"
            guidance += f"  → Do NOT generate data for other types of medical procedures\n"
        
        guidance += f"\n🚨 CRITICAL BILLING COMPLIANCE 🚨\n"
        guidance += f"Your response MUST match the CPT codes above EXACTLY. do not add any other tests or data that is not related to the billed procedures.\n"
        guidance += f"Only generate data that corresponds to the billed procedures.\n"
        return guidance

    def _extract_previous_lab_values(self, chat_history: List[Dict[str, Any]]) -> str:
        """Extract previous lab values from chat history using LLM for consistency."""
        if not chat_history:
            return ""
        
        # Get assistant messages that might contain lab results
        assistant_messages = []
        for msg in chat_history:
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                # Only include messages that likely contain lab data
                if any(keyword in content.lower() for keyword in ['test', 'result', 'lab', 'cortisol', 'acth', 'glucose', 'sodium', 'potassium', 'creatinine', 'hemoglobin', 'wbc']):
                    assistant_messages.append(content)
        
        if not assistant_messages:
            return ""
        
        # Use LLM to extract lab values
        extraction_prompt = f"""
        Analyze the following chat messages and extract any laboratory test results that were previously provided.
        
        CHAT MESSAGES:
        {chr(10).join(assistant_messages)}
        
        TASK:
        Extract all laboratory test results mentioned in these messages. Focus on:
        - Complete blood count (hemoglobin, WBC, platelets, etc.)
        - Chemistry panels (glucose, electrolytes, liver function, etc.)
        - Endocrine tests (cortisol, ACTH, thyroid function, etc.)
        - Cardiac markers (troponin, CK, etc.)
        - Inflammatory markers (ESR, CRP, procalcitonin, etc.)
        - Coagulation studies (PT, PTT, INR, etc.)
        - Specialty tests (antibodies, tumor markers, etc.)
        
        FORMAT:
        For each lab value found, extract:
        - Test name
        - Result value with units
        - Normal range (if provided)
        
        RESPONSE FORMAT:
        Return a simple list format like:
        - Test Name: Value (Normal Range)
        
        EXAMPLE:
        - Morning Cortisol: 28 μg/dL (Normal: 6-23)
        - ACTH: 15 pg/mL (Normal: 10-60)
        - 24-Hour Urine Cortisol: 400 μg (Normal: 3.5-45)
        
        RULES:
        - Only extract actual numerical lab values, not descriptions
        - Include units when available
        - If no lab values are found, respond with "No lab values found"
        - Be precise with test names and values
        """
        
        try:
            # Debug: Show which LLM model is being used for lab extraction
            model_name = getattr(self.llm_client, 'default_text_model', 'Unknown')
            logger.info(f"🤖 LAB EXTRACTION LLM MODEL: {model_name}")
            
            response = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000    # Increased significantly for Vertex AI thoughts mechanism
            )
            
            result = response.get('text', '') if isinstance(response, dict) else str(response)
            
            # Clean up the response
            result = result.strip()
            if result.lower() == "no lab values found" or not result:
                return ""
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting lab values with LLM: {str(e)}")
            return ""

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response to extract assistant_response and formatting_type."""
        try:
            # Clean up the response text - remove markdown blocks if present
            cleaned_text = response_text.strip()
            
            # Remove ```json and ``` blocks
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]  # Remove ```json
            elif cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]   # Remove ```
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]  # Remove ending ```
            
            cleaned_text = cleaned_text.strip()
            
            # Fix common JSON issues before parsing
            cleaned_text = self._fix_json_formatting(cleaned_text)
            
            # Log the cleaned text for debugging
            logger.debug(f"Attempting to parse JSON: {cleaned_text[:200]}...")
            
            # Parse JSON response
            import json
            response_data = json.loads(cleaned_text)
            
            # Extract fields with defaults
            result = {
                "assistant_response": response_data.get("assistant_response", ""),
                "formatting_type": response_data.get("formatting_type", "narrative")
            }
            
            # Validate formatting_type
            valid_formats = ["table", "list", "narrative"]
            if result["formatting_type"] not in valid_formats:
                result["formatting_type"] = "narrative"
            
            logger.debug(f"Successfully parsed synthetic response: {len(result['assistant_response'])} chars")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in synthetic response: {str(e)}")
            logger.error(f"Problematic text: {response_text[:500]}...")
            
            # Try to extract assistant_response from malformed JSON using regex
            return self._extract_from_malformed_json(response_text)
            
        except Exception as e:
            logger.error(f"Error parsing synthetic response: {str(e)}")
            logger.error(f"Response text: {response_text[:200]}...")
            
            # If all else fails, try to use the raw response if it looks like medical content
            cleaned_response = response_text.strip()
            if len(cleaned_response) > 20 and not cleaned_response.startswith('{'):
                # Raw response might be the actual medical content
                logger.warning("Using raw response as assistant_response")
                return {
                    "assistant_response": cleaned_response,
                    "formatting_type": "narrative"
                }
            
            # Re-raise the exception to force retry rather than provide bad fallback
            raise Exception(f"Could not parse synthetic response: {str(e)}")

    def _fix_json_formatting(self, text: str) -> str:
        """Fix common JSON formatting issues in the LLM response."""
        import re
        
        # Find the assistant_response field and fix newlines within it
        def fix_assistant_response(match):
            field_content = match.group(1)
            # Escape unescaped newlines
            field_content = re.sub(r'(?<!\\)\n', '\\n', field_content)
            # Escape unescaped quotes
            field_content = re.sub(r'(?<!\\)"', '\\"', field_content)
            return f'"assistant_response": "{field_content}"'
        
        # Fix the assistant_response field specifically
        text = re.sub(
            r'"assistant_response"\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
            fix_assistant_response,
            text,
            flags=re.DOTALL
        )
        
        return text

    def _extract_from_malformed_json(self, response_text: str) -> Dict[str, Any]:
        """Extract data from malformed JSON using regex."""
        import re
        
        # Try to find assistant_response field even in malformed JSON
        assistant_match = re.search(
            r'"assistant_response"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', 
            response_text, 
            re.DOTALL
        )
        
        if assistant_match:
            assistant_response = assistant_match.group(1)
            # Unescape basic JSON escape sequences
            assistant_response = assistant_response.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            
            # Try to find formatting_type
            format_match = re.search(r'"formatting_type"\s*:\s*"([^"]*)"', response_text)
            formatting_type = format_match.group(1) if format_match else "narrative"
            
            # Validate formatting_type
            valid_formats = ["table", "list", "narrative"]
            if formatting_type not in valid_formats:
                formatting_type = "narrative"
            
            logger.info(f"Recovered assistant_response from malformed JSON: {len(assistant_response)} chars")
            return {
                "assistant_response": assistant_response,
                "formatting_type": formatting_type
            }
        
        # If regex extraction fails, try to use the raw response if it looks like medical content
        cleaned_response = response_text.strip()
        if len(cleaned_response) > 20 and not cleaned_response.startswith('{'):
            # Raw response might be the actual medical content without JSON structure
            logger.warning("Using raw response as medical content")
            return {
                "assistant_response": cleaned_response,
                "formatting_type": "narrative"
            }
        
        # Re-raise to force retry rather than provide rule-violating content
        raise Exception("Could not extract valid medical content from response") 