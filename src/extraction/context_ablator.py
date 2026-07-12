"""
Context Ablator - Removes ground truth data from case text for evaluation.

This module removes specific clinical findings from case narratives to test
whether the synthetic response generator can accurately recreate them.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AblationResult:
    """Result of context ablation."""
    original_text: str
    ablated_text: str
    removed_content: str
    removal_positions: List[Tuple[int, int]]  # (start, end) positions
    ablation_method: str
    success: bool
    notes: str = ""


class ContextAblator:
    """Removes ground truth findings from case text for evaluation."""
    
    def __init__(self, llm_client=None):
        """
        Initialize ContextAblator.
        
        Args:
            llm_client: Optional LLM client for intelligent ablation. If provided,
                       will use LLM-based surgical ablation instead of rule-based.
        """
        self.ablation_patterns = []
        self.llm_client = llm_client
    
    def ablate_exact_match(
        self,
        case_text: str,
        content_to_remove: str,
        context_chars: int = 100
    ) -> AblationResult:
        """
        Remove exact matching content from case text.
        
        Args:
            case_text: Full case narrative
            content_to_remove: Exact text to remove
            context_chars: Characters of context to keep around removal
            
        Returns:
            AblationResult with ablated text
        """
        # Find exact match
        match_pos = case_text.find(content_to_remove)
        
        if match_pos == -1:
            return AblationResult(
                original_text=case_text,
                ablated_text=case_text,
                removed_content=content_to_remove,
                removal_positions=[],
                ablation_method="exact_match",
                success=False,
                notes=f"Content not found in case text"
            )
        
        # Remove the content
        ablated = case_text[:match_pos] + case_text[match_pos + len(content_to_remove):]
        
        return AblationResult(
            original_text=case_text,
            ablated_text=ablated,
            removed_content=content_to_remove,
            removal_positions=[(match_pos, match_pos + len(content_to_remove))],
            ablation_method="exact_match",
            success=True,
            notes=f"Removed exact match at position {match_pos}"
        )
    
    def ablate_fuzzy_match(
        self,
        case_text: str,
        content_to_remove: str,
        similarity_threshold: float = 0.8
    ) -> AblationResult:
        """
        Remove similar content using fuzzy matching.
        
        Args:
            case_text: Full case narrative
            content_to_remove: Text to find and remove (approximately)
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            AblationResult with ablated text
        """
        # For now, fall back to exact match
        # TODO: Implement fuzzy matching with difflib or similar
        return self.ablate_exact_match(case_text, content_to_remove)
    
    def ablate_sentence_containing(
        self,
        case_text: str,
        keyword: str
    ) -> AblationResult:
        """
        Remove entire sentence(s) containing a keyword or phrase.
        
        Args:
            case_text: Full case narrative
            keyword: Keyword to search for
            
        Returns:
            AblationResult with ablated text
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', case_text)
        
        removed_sentences = []
        kept_sentences = []
        removal_positions = []
        
        current_pos = 0
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                removed_sentences.append(sentence)
                removal_positions.append((current_pos, current_pos + len(sentence)))
            else:
                kept_sentences.append(sentence)
            current_pos += len(sentence) + 1  # +1 for space
        
        if not removed_sentences:
            return AblationResult(
                original_text=case_text,
                ablated_text=case_text,
                removed_content=keyword,
                removal_positions=[],
                ablation_method="sentence_containing",
                success=False,
                notes=f"No sentences containing '{keyword}' found"
            )
        
        ablated = ' '.join(kept_sentences)
        removed = ' '.join(removed_sentences)
        
        return AblationResult(
            original_text=case_text,
            ablated_text=ablated,
            removed_content=removed,
            removal_positions=removal_positions,
            ablation_method="sentence_containing",
            success=True,
            notes=f"Removed {len(removed_sentences)} sentence(s) containing '{keyword}'"
        )
    
    def ablate_paragraph_containing(
        self,
        case_text: str,
        keyword: str
    ) -> AblationResult:
        """
        Remove entire paragraph(s) containing a keyword or phrase.
        
        Args:
            case_text: Full case narrative
            keyword: Keyword to search for
            
        Returns:
            AblationResult with ablated text
        """
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', case_text)
        
        removed_paragraphs = []
        kept_paragraphs = []
        removal_positions = []
        
        current_pos = 0
        for para in paragraphs:
            if keyword.lower() in para.lower():
                removed_paragraphs.append(para)
                removal_positions.append((current_pos, current_pos + len(para)))
            else:
                kept_paragraphs.append(para)
            current_pos += len(para) + 2  # +2 for double newline
        
        if not removed_paragraphs:
            return AblationResult(
                original_text=case_text,
                ablated_text=case_text,
                removed_content=keyword,
                removal_positions=[],
                ablation_method="paragraph_containing",
                success=False,
                notes=f"No paragraphs containing '{keyword}' found"
            )
        
        ablated = '\n\n'.join(kept_paragraphs)
        removed = '\n\n'.join(removed_paragraphs)
        
        return AblationResult(
            original_text=case_text,
            ablated_text=ablated,
            removed_content=removed,
            removal_positions=removal_positions,
            ablation_method="paragraph_containing",
            success=True,
            notes=f"Removed {len(removed_paragraphs)} paragraph(s) containing '{keyword}'"
        )
    
    def ablate_clinical_data_entry(
        self,
        case_text: str,
        raw_content: str,
        category: str
    ) -> AblationResult:
        """
        Comprehensive ablation that removes the ground truth AND all references to it.
        
        This prevents data leakage by removing:
        1. The original finding
        2. All subsequent mentions in discussion/reasoning
        3. Figure captions
        4. Any text containing key terms from the finding
        
        Args:
            case_text: Full case narrative
            raw_content: Ground truth content to remove
            category: Category of clinical data
            
        Returns:
            AblationResult with comprehensive ablation
        """
        # First, try exact match
        result = self.ablate_exact_match(case_text, raw_content)
        if result.success:
            ablated_text = result.ablated_text
            method = "exact_match"
            initial_removed = result.removed_content
        else:
            # Try sentence-level removal for initial finding
            if category == 'imaging':
                for modality in ['CT', 'MRI', 'X-ray', 'ultrasound', 'PET']:
                    if modality.lower() in raw_content.lower():
                        result = self.ablate_sentence_containing(case_text, modality)
                        if result.success:
                            ablated_text = result.ablated_text
                            method = f"sentence_containing_{modality}"
                            initial_removed = result.removed_content
                            break
                else:
                    ablated_text = case_text
                    method = "none"
                    initial_removed = ""
            else:
                ablated_text = case_text
                method = "none"
                initial_removed = ""
        
        # Now perform comprehensive removal of ALL references
        # Extract key medical terms from the ground truth
        key_terms = self._extract_key_terms(raw_content)
        
        # Remove all sentences containing any of these key terms
        sentences = ablated_text.split('.')
        filtered_sentences = []
        removed_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            contains_key_term = False
            
            for term in key_terms:
                if term.lower() in sentence_lower:
                    contains_key_term = True
                    removed_sentences.append(sentence.strip())
                    break
            
            if not contains_key_term and sentence.strip():
                filtered_sentences.append(sentence)
        
        final_ablated_text = '. '.join(filtered_sentences)
        
        # Combine removed content
        all_removed = initial_removed
        if removed_sentences:
            all_removed += '\n\n[Additional references removed:]\n' + '. '.join(removed_sentences[:10])
        
        return AblationResult(
            original_text=case_text,
            ablated_text=final_ablated_text,
            removed_content=all_removed,
            removal_positions=[],
            ablation_method=f"{method}_comprehensive",
            success=True,
            notes=f"Comprehensive ablation: removed {len(removed_sentences)} sentences containing key terms: {', '.join(key_terms[:5])}"
        )
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key medical terms from ground truth text.
        
        Returns terms that are likely to uniquely identify this finding:
        - Anatomical terms (e.g., "adrenal", "liver", "kidney")
        - Measurements (e.g., "1.9 cm", "2.5 mm")
        - Medical descriptors (e.g., "nodule", "mass", "lesion")
        """
        import re
        
        key_terms = set()
        text_lower = text.lower()
        
        # Extract measurements (numbers + units)
        measurements = re.findall(r'\d+\.?\d*\s*(?:cm|mm|mg|ml|mmol|g/dl|hounsfield)', text_lower)
        key_terms.update(measurements)
        
        # Common anatomical terms
        anatomical_words = [
            'adrenal', 'liver', 'spleen', 'kidney', 'pancreas', 'lung', 'heart',
            'brain', 'thyroid', 'prostate', 'ovary', 'uterus', 'bladder',
            'stomach', 'colon', 'intestine', 'gallbladder', 'aorta', 'vena cava',
            'thalamus', 'cerebellum', 'cortex', 'medulla', 'pelvis', 'abdomen'
        ]
        
        for word in anatomical_words:
            if word in text_lower:
                key_terms.add(word)
        
        # Medical descriptors
        descriptors = [
            'nodule', 'mass', 'lesion', 'tumor', 'cyst', 'adenoma', 'carcinoma',
            'infiltrate', 'consolidation', 'opacity', 'effusion', 'edema',
            'hemorrhage', 'infarct', 'ischemia', 'thrombus', 'embolus'
        ]
        
        for desc in descriptors:
            if desc in text_lower:
                key_terms.add(desc)
        
        # Laterality
        if 'left' in text_lower:
            key_terms.add('left')
        if 'right' in text_lower:
            key_terms.add('right')
        if 'bilateral' in text_lower:
            key_terms.add('bilateral')
        
        return list(key_terms)
    
    def ablate_with_llm(
        self,
        case_text: str,
        raw_content: str,
        category: str,
        test_name: str
    ) -> AblationResult:
        """
        Use LLM to surgically remove ONLY the specific test finding and its references.
        
        This method preserves:
        - The final diagnosis (crucial for synthetic response generation)
        - Other test results
        - Clinical reasoning not directly about this specific finding
        
        Args:
            case_text: Full case narrative
            raw_content: The specific test finding to remove
            category: Category (imaging, laboratory_tests, etc.)
            test_name: Name of the test (CT, MRI, CBC, etc.)
            
        Returns:
            AblationResult with surgically ablated text
        """
        if not self.llm_client:
            # Fall back to rule-based if no LLM available
            return self.ablate_clinical_data_entry(case_text, raw_content, category)
        
        # Create ablation prompt for the LLM
        ablation_prompt = f"""You are a medical text editor. Your task is to remove a SPECIFIC test finding from a clinical case report while preserving all other information.

CRITICAL INSTRUCTIONS:
1. Remove ONLY the specific test finding mentioned below
2. Remove ALL references to this specific finding (in discussion, figures, reasoning)
3. PRESERVE the final diagnosis - this is essential for the case
4. PRESERVE all other test results and clinical information
5. PRESERVE the overall narrative flow and clinical reasoning

TEST TO REMOVE:
Category: {category}
Test Name: {test_name}
Finding: {raw_content}

WHAT TO REMOVE:
- The exact finding: "{raw_content}"
- Any sentences that discuss this specific finding
- Figure captions describing this finding
- Clinical reasoning that references this specific finding
- Any measurements, anatomical locations, or descriptors specific to this finding

WHAT TO PRESERVE:
- The final diagnosis (e.g., "Cushing's syndrome", "Diabetes mellitus")
- Other test results (labs, other imaging, physical exam findings)
- General clinical reasoning not tied to this specific finding
- Patient demographics and history
- Treatment plans and outcomes

CASE TEXT:
{case_text}

OUTPUT FORMAT:
Return ONLY a JSON object (no markdown, no explanation) with:
{{
    "sentences_to_remove": [
        "exact sentence 1 to remove",
        "exact sentence 2 to remove",
        "exact sentence 3 to remove"
    ],
    "notes": "brief explanation of what was identified for removal and what was preserved"
}}

IMPORTANT: 
- Return sentences EXACTLY as they appear in the case text
- Include complete sentences, not fragments
- The sentences will be programmatically removed, so precision is critical

Be surgical and precise. Only identify sentences that directly relate to the specific test finding mentioned above."""

        try:
            # Call LLM using text_completion
            llm_response = self.llm_client.text_completion(
                prompt=ablation_prompt,
                temperature=0.1,  # Low temperature for precise editing
                max_tokens=4000
            )
            
            # Parse response
            import json
            response_text = llm_response.get('text', '{}')
            
            # Log the raw response for debugging
            print(f"🔍 LLM raw response length: {len(response_text)} chars")
            print(f"🔍 LLM response preview: {response_text[:500]}...")
            
            # Try to extract JSON from the response (may have markdown formatting)
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                if json_end == -1:  # Truncated - try to salvage
                    # Find last complete sentence entry
                    last_quote = response_text.rfind('",')
                    if last_quote > json_start:
                        response_text = response_text[json_start:last_quote+1] + '\n    ],\n    "notes": "Truncated"\n}'
                    else:
                        response_text = response_text[json_start:]
                else:
                    response_text = response_text[json_start:json_end].strip()
            elif '```' in response_text:
                json_start = response_text.find('```') + 3
                json_end = response_text.find('```', json_start)
                if json_end == -1:
                    last_quote = response_text.rfind('",')
                    if last_quote > json_start:
                        response_text = response_text[json_start:last_quote+1] + '\n    ],\n    "notes": "Truncated"\n}'
                    else:
                        response_text = response_text[json_start:]
                else:
                    response_text = response_text[json_start:json_end].strip()
            
            print(f"🔍 Extracted JSON: {response_text[:300]}...")
            
            result_data = json.loads(response_text)
            
            sentences_to_remove = result_data.get('sentences_to_remove', [])
            notes = result_data.get('notes', 'LLM-based surgical ablation')
            
            # Programmatically remove the identified sentences
            ablated_text = case_text
            removed_content_list = []
            
            for sentence in sentences_to_remove:
                # Normalize whitespace in both sentence and text for matching
                # (PDF artifacts can cause double spaces)
                sentence_normalized = re.sub(r'\s+', ' ', sentence).strip()
                
                # Try exact match first
                if sentence in ablated_text:
                    ablated_text = ablated_text.replace(sentence, '')
                    removed_content_list.append(sentence)
                else:
                    # Try with normalized whitespace
                    # Find and replace the sentence with normalized spacing
                    pattern = re.escape(sentence_normalized).replace(r'\ ', r'\s+')
                    match = re.search(pattern, ablated_text)
                    if match:
                        ablated_text = ablated_text[:match.start()] + ablated_text[match.end():]
                        removed_content_list.append(match.group())
                    else:
                        print(f"⚠️  Could not find sentence to remove: {sentence[:100]}...")
            
            # Clean up extra whitespace
            ablated_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', ablated_text)
            ablated_text = re.sub(r'  +', ' ', ablated_text)
            ablated_text = ablated_text.strip()
            
            removed_content = '\n\n'.join(removed_content_list) if removed_content_list else raw_content
            
            return AblationResult(
                original_text=case_text,
                ablated_text=ablated_text,
                removed_content=removed_content,
                removal_positions=[],  # LLM doesn't provide positions
                ablation_method="llm_surgical",
                success=True,
                notes=notes
            )
            
        except Exception as e:
            # Fall back to rule-based on error
            print(f"LLM ablation failed: {e}. Falling back to rule-based.")
            return self.ablate_clinical_data_entry(case_text, raw_content, category)
    
    def validate_ablation(
        self,
        ablation_result: AblationResult,
        min_removal_length: int = 10
    ) -> Tuple[bool, str]:
        """
        Validate that ablation was successful and meaningful.
        
        Args:
            ablation_result: Result to validate
            min_removal_length: Minimum length of removed content
            
        Returns:
            (is_valid, reason)
        """
        if not ablation_result.success:
            return False, "Ablation failed"
        
        if len(ablation_result.removed_content) < min_removal_length:
            return False, f"Removed content too short ({len(ablation_result.removed_content)} chars)"
        
        if ablation_result.ablated_text == ablation_result.original_text:
            return False, "No change in text after ablation"
        
        # Check that we didn't remove too much (>50% of original)
        removal_ratio = 1 - (len(ablation_result.ablated_text) / len(ablation_result.original_text))
        if removal_ratio > 0.5:
            return False, f"Removed too much text ({removal_ratio*100:.1f}%)"
        
        return True, "Ablation valid"


# Example usage
if __name__ == "__main__":
    ablator = ContextAblator()
    
    # Test case
    case_text = """A 70-year-old man presented with weakness and anorexia.
    
    CT of the abdomen revealed a left adrenal nodule measuring 1.9 cm.
    
    Laboratory results showed hypokalemia with potassium of 3.1 mmol/L."""
    
    # Test ablation
    result = ablator.ablate_clinical_data_entry(
        case_text=case_text,
        raw_content="CT of the abdomen revealed a left adrenal nodule measuring 1.9 cm",
        category="imaging"
    )
    
    print(f"Success: {result.success}")
    print(f"Method: {result.ablation_method}")
    print(f"Notes: {result.notes}")
    print(f"\nRemoved: {result.removed_content}")
    print(f"\nAblated text:\n{result.ablated_text}")
    
    # Validate
    is_valid, reason = ablator.validate_ablation(result)
    print(f"\nValidation: {is_valid} - {reason}")
