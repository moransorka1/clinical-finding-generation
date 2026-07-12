"""
Case Reconstructor - Build case context from structured data.

Instead of ablating the full case text, this module reconstructs the case
from structured components (HPI + clinical data) while excluding the ground truth test.

This eliminates the fragile text-matching problem in ablation.
"""

import json
import re
from typing import Dict, List, Optional
from sqlalchemy import text as sql_text


class CaseReconstructor:
    """Reconstructs case context from structured data, excluding ground truth test."""
    
    # Categories where leaked info is primarily numeric → masking works
    NUMERIC_CATEGORIES = {'laboratory_tests'}
    # Categories where leaked info is qualitative text → must remove entirely
    TEXT_CATEGORIES = {'imaging', 'physical_examination_or_assessments', 'procedures'}
    
    def __init__(self, db_session, llm_client=None, same_category_strategy='hybrid'):
        """
        Initialize reconstructor.
        
        Args:
            db_session: SQLAlchemy database session
            llm_client: Optional LLM client for HPI ablation (removes sentences
                        that leak the excluded test finding)
            same_category_strategy: How to handle other items in the same category
                as the excluded test. Options:
                - 'hybrid': (default) mask numerics for labs, remove for imaging/physical_exam
                - 'mask': Replace numeric values with [VALUE] placeholders (Option A)
                - 'remove': Remove ALL same-category items entirely (Option B)
                - 'none': Leave same-category items as-is (legacy behavior)
        """
        self.db_session = db_session
        self.llm_client = llm_client
        self.same_category_strategy = same_category_strategy
    
    def reconstruct_case_without_test(
        self,
        case_id: int,
        exclude_test_content: str,
        exclude_category: str = None
    ) -> Dict:
        """
        Reconstruct case context excluding a specific test result.
        
        Args:
            case_id: The case ID
            exclude_test_content: The exact test content to exclude
            exclude_category: Optional category filter (e.g., 'imaging', 'laboratory_tests')
        
        Returns:
            Dictionary with:
                - hpi: History of present illness
                - clinical_data: Structured clinical data by category
                - full_context: Combined narrative text
                - excluded_item: Info about what was excluded
        """
        # Get HPI (History of Present Illness)
        hpi_result = self.db_session.execute(
            sql_text("""
                SELECT hpi_raw, description, title
                FROM cases 
                WHERE id = :case_id
            """),
            {"case_id": case_id}
        ).fetchone()
        
        if not hpi_result:
            raise ValueError(f"Case {case_id} not found")
        
        hpi = hpi_result.hpi_raw or ""
        case_description = hpi_result.description or ""
        case_title = hpi_result.title or ""
        
        # Get all clinical data
        clinical_data_results = self.db_session.execute(
            sql_text("""
                SELECT id, category, content, sequence_order
                FROM case_clinical_data
                WHERE case_id = :case_id
                ORDER BY sequence_order
            """),
            {"case_id": case_id}
        ).fetchall()
        
        # Organize by category and exclude matching test
        clinical_data_by_category = {}
        excluded_item = None
        total_items = 0
        excluded_count = 0
        
        for row in clinical_data_results:
            category = row.category
            content = row.content
            
            # Check if this is the test to exclude
            is_excluded = False
            if exclude_test_content:
                # Normalize for comparison (strip whitespace, lowercase)
                content_normalized = ' '.join(content.lower().split())
                exclude_normalized = ' '.join(exclude_test_content.lower().split())
                
                if content_normalized == exclude_normalized:
                    is_excluded = True
                elif exclude_category and category == exclude_category:
                    # If category matches, check for substring match
                    if exclude_normalized in content_normalized or content_normalized in exclude_normalized:
                        is_excluded = True
            
            if is_excluded:
                excluded_item = {
                    'id': row.id,
                    'category': category,
                    'content': content,
                    'sequence_order': row.sequence_order
                }
                excluded_count += 1
                continue
            
            # Add to structured data
            if category not in clinical_data_by_category:
                clinical_data_by_category[category] = []
            
            clinical_data_by_category[category].append({
                'content': content,
                'sequence_order': row.sequence_order
            })
            total_items += 1
        
        # Ablate HPI: remove sentences that leak the excluded finding
        hpi_ablated = hpi
        hpi_sentences_removed = []
        if excluded_item and hpi:
            hpi_ablated, hpi_sentences_removed = self._ablate_hpi(
                hpi=hpi,
                excluded_content=excluded_item['content'],
                excluded_category=excluded_item.get('category', '')
            )
            if hpi_sentences_removed:
                print(f"   🧹 HPI ablation: removed {len(hpi_sentences_removed)} sentence(s) that referenced excluded finding")
        
        # Ablate clinical data items that leak the excluded finding
        # (e.g., patient_condition items that describe the same symptom)
        context_items_removed = []
        if excluded_item and self.llm_client:
            clinical_data_by_category, context_items_removed = self._ablate_context_items(
                clinical_data_by_category=clinical_data_by_category,
                excluded_content=excluded_item['content'],
                excluded_category=excluded_item.get('category', '')
            )
            if context_items_removed:
                total_items -= len(context_items_removed)
                print(f"   🧹 Context ablation: removed {len(context_items_removed)} item(s) that leaked excluded finding")
        
        # Same-category numeric masking / removal to prevent cross-entry contamination
        same_category_masked_count = 0
        same_category_removed_count = 0
        same_category_strategy_used = self.same_category_strategy
        
        if excluded_item and exclude_category and exclude_category in clinical_data_by_category:
            # Resolve effective strategy for this category
            if self.same_category_strategy == 'hybrid':
                if exclude_category in self.NUMERIC_CATEGORIES:
                    effective_strategy = 'mask'
                elif exclude_category in self.TEXT_CATEGORIES:
                    effective_strategy = 'remove'
                else:
                    effective_strategy = 'remove'  # safe default for unknown categories
                same_category_strategy_used = f'hybrid:{effective_strategy}'
            else:
                effective_strategy = self.same_category_strategy
            
            if effective_strategy == 'mask' and self.llm_client:
                # Option A: Mask numeric values in same-category items (best for labs)
                clinical_data_by_category, same_category_masked_count = self._mask_same_category_numerics(
                    clinical_data_by_category=clinical_data_by_category,
                    excluded_category=exclude_category
                )
                if same_category_masked_count > 0:
                    print(f"   🔢 Numeric masking: masked values in {same_category_masked_count} same-category item(s)")
                    
            elif effective_strategy == 'remove':
                # Option B: Remove ALL same-category items (needed for imaging/physical_exam)
                removed_same_cat = clinical_data_by_category.pop(exclude_category, [])
                same_category_removed_count = len(removed_same_cat)
                total_items -= same_category_removed_count
                # Tag each removed item with its category before extending
                for item in removed_same_cat:
                    item['category'] = exclude_category
                context_items_removed.extend(removed_same_cat)
                if same_category_removed_count > 0:
                    print(f"   🗑️ Same-category removal: removed all {same_category_removed_count} {exclude_category} item(s)")

        # Mask vital-sign numerics when excluding imaging / physical-exam / procedures.
        # ECG / echo / Holter results often mirror vital-sign values (HR, BP, temp),
        # so leaving raw vital-sign numbers in the context enables indirect leakage.
        vital_signs_masked_count = 0
        if (excluded_item
                and exclude_category in self.TEXT_CATEGORIES
                and 'vital_signs' in clinical_data_by_category):
            for item in clinical_data_by_category['vital_signs']:
                original = item['content']
                item['content'] = self._regex_mask_numerics(original)
                if item['content'] != original:
                    vital_signs_masked_count += 1
            if vital_signs_masked_count > 0:
                print(f"   🔢 Vital-sign masking: masked numerics in {vital_signs_masked_count} vital-sign item(s)")

        # Build full context narrative
        full_context = self._build_narrative(
            hpi=hpi_ablated,
            case_description=case_description,
            case_title=case_title,
            clinical_data_by_category=clinical_data_by_category
        )
        
        return {
            'hpi': hpi_ablated,
            'hpi_original': hpi,
            'hpi_sentences_removed': hpi_sentences_removed,
            'context_items_removed': context_items_removed,
            'same_category_strategy': same_category_strategy_used,
            'same_category_masked_count': same_category_masked_count,
            'same_category_removed_count': same_category_removed_count,
            'vital_signs_masked_count': vital_signs_masked_count,
            'case_description': case_description,
            'case_title': case_title,
            'clinical_data': clinical_data_by_category,
            'full_context': full_context,
            'excluded_item': excluded_item,
            'stats': {
                'total_items': total_items,
                'excluded_count': excluded_count,
                'included_count': total_items
            }
        }
    
    def _mask_same_category_numerics(
        self,
        clinical_data_by_category: Dict[str, List[Dict]],
        excluded_category: str
    ) -> tuple:
        """
        Mask numeric values in same-category items to prevent cross-entry contamination.
        
        When testing one lab entry, all OTHER lab entries' numeric values are replaced
        with [VALUE] placeholders. This preserves test names and qualitative info while
        preventing the LLM from anchoring on related numeric values.
        
        Args:
            clinical_data_by_category: The structured clinical data dict
            excluded_category: Category of the excluded item (items in this category get masked)
            
        Returns:
            Tuple of (updated_clinical_data_by_category, count_of_masked_items)
        """
        if excluded_category not in clinical_data_by_category:
            return clinical_data_by_category, 0
        
        items = clinical_data_by_category[excluded_category]
        if not items:
            return clinical_data_by_category, 0
        
        # Build numbered list for the LLM
        items_text = '\n'.join(
            f'{i+1}. {it["content"]}'
            for i, it in enumerate(items)
        )
        
        prompt = f"""Replace ALL specific numeric values in these clinical items with [VALUE] placeholders.

RULES:
1. Replace numbers, percentages, ratios, decimals, and counts with [VALUE]
2. Keep test/study names, units, qualitative descriptions, and anatomical terms intact
3. Keep words like "normal", "elevated", "positive", "negative", "unremarkable" intact
4. Replace date-like numbers too (ages, durations with numbers)
5. Output ONLY the masked items, one per line, with the same numbering

ITEMS:
{items_text}

MASKED ITEMS:"""

        try:
            response = self.llm_client.text_completion(
                prompt=prompt,
                temperature=0.0,
                max_tokens=4000
            )
            response_text = response['text'].strip()
            
            # Strip markdown fences if present
            if '```' in response_text:
                if '```text' in response_text:
                    response_text = response_text.split('```text')[1].split('```')[0].strip()
                elif '```' in response_text:
                    parts = response_text.split('```')
                    if len(parts) >= 3:
                        response_text = parts[1].strip()
                    else:
                        response_text = parts[0].strip()
            
            # Parse numbered lines back into items
            masked_lines = {}
            for line in response_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Match "1. content" or "1) content"
                match = re.match(r'^(\d+)[.)]\s*(.+)$', line)
                if match:
                    idx = int(match.group(1))
                    content = match.group(2).strip()
                    masked_lines[idx] = content
            
            # Update items with masked content
            masked_count = 0
            for i, item in enumerate(items):
                idx = i + 1
                if idx in masked_lines:
                    original = item['content']
                    masked = masked_lines[idx]
                    # Verify masking actually happened (contains [VALUE])
                    if '[VALUE]' in masked:
                        item['content'] = masked
                        masked_count += 1
                    elif original != masked:
                        # LLM changed something but didn't use [VALUE] — 
                        # check if numbers were actually present
                        has_numbers = bool(re.search(r'\d', original))
                        if not has_numbers:
                            # No numbers to mask — this is fine
                            pass
                        else:
                            # Numbers present but not masked — use regex fallback
                            item['content'] = self._regex_mask_numerics(original)
                            if item['content'] != original:
                                masked_count += 1
                else:
                    # LLM didn't return this item — apply regex fallback
                    original = item['content']
                    if re.search(r'\d', original):
                        item['content'] = self._regex_mask_numerics(original)
                        if item['content'] != original:
                            masked_count += 1
            
            return clinical_data_by_category, masked_count
            
        except Exception as e:
            print(f"   ⚠️ Numeric masking LLM failed: {e}, falling back to regex")
            return self._regex_mask_same_category(clinical_data_by_category, excluded_category)
    
    def _regex_mask_numerics(self, text: str) -> str:
        """
        Regex fallback: replace numeric values in text with [VALUE].
        
        Handles integers, decimals, percentages, fractions, and comma-separated numbers.
        Preserves non-numeric text.
        """
        # Replace percentages first (e.g., "90%", "33.2%")
        text = re.sub(r'\d[\d,]*\.?\d*\s*%', '[VALUE]%', text)
        # Replace numbers with commas (e.g., "14,300", "20,200")
        text = re.sub(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?', '[VALUE]', text)
        # Replace decimal numbers (e.g., "6.4", "2.7")
        text = re.sub(r'\d+\.\d+', '[VALUE]', text)
        # Replace remaining standalone integers (but not inside [VALUE])
        text = re.sub(r'(?<!\[)\b\d+\b(?!\])', '[VALUE]', text)
        # Clean up multiple consecutive [VALUE] markers
        text = re.sub(r'\[VALUE\]\s*[–-]\s*\[VALUE\]', '[VALUE]', text)
        return text
    
    def _regex_mask_same_category(
        self,
        clinical_data_by_category: Dict[str, List[Dict]],
        excluded_category: str
    ) -> tuple:
        """
        Regex fallback for masking all same-category numerics when LLM fails.
        """
        if excluded_category not in clinical_data_by_category:
            return clinical_data_by_category, 0
        
        items = clinical_data_by_category[excluded_category]
        masked_count = 0
        for item in items:
            original = item['content']
            item['content'] = self._regex_mask_numerics(original)
            if item['content'] != original:
                masked_count += 1
        
        return clinical_data_by_category, masked_count
    
    def _ablate_context_items(
        self,
        clinical_data_by_category: Dict[str, List[Dict]],
        excluded_content: str,
        excluded_category: str
    ) -> tuple:
        """
        Remove clinical data items (e.g., patient_condition) that leak the excluded finding.
        
        Uses a single LLM call to check all remaining items against the excluded content.
        
        Args:
            clinical_data_by_category: The structured clinical data dict
            excluded_content: The ground-truth content being excluded
            excluded_category: Category of the excluded item
            
        Returns:
            Tuple of (filtered_clinical_data_by_category, list_of_removed_items)
        """
        # Collect all items into a flat list for the LLM to evaluate
        all_items = []
        for category, items in clinical_data_by_category.items():
            for item in items:
                all_items.append({
                    'category': category,
                    'content': item['content'],
                    'sequence_order': item['sequence_order']
                })
        
        if not all_items:
            return clinical_data_by_category, []
        
        # Build a numbered list for the LLM
        items_text = '\n'.join(
            f'{i+1}. [{it["category"]}] {it["content"]}'
            for i, it in enumerate(all_items)
        )
        
        prompt = f"""You are reviewing a clinical case for data leakage. A specific test finding has been EXCLUDED from the case to test whether an AI can regenerate it. However, other items in the case context may describe the SAME finding using different words, which would leak the answer.

EXCLUDED FINDING:
Category: {excluded_category}
Content: {excluded_content}

REMAINING CASE ITEMS:
{items_text}

TASK: Identify which items (by number) describe the SAME clinical finding as the excluded content, even using synonyms. For example:
- "weakness and numbness in the right hand" leaks "diminished strength and sensation in the right arm" (same finding, different words)
- "fever of 102°F" leaks "temperature 38.9°C" (same measurement, different units)
- "patient has headache" does NOT leak "CT showed hemorrhage" (symptom ≠ finding)
- "history of migraine" does NOT leak "diminished strength in right arm" (unrelated)

RULES:
1. Only flag items that describe the SAME specific finding
2. Do NOT flag items that merely motivated ordering the test
3. Be conservative — when in doubt, keep the item

RESPONSE FORMAT — two lines, nothing else:
Line 1: a JSON array of item numbers to remove, e.g. [1, 3] or [] if none
Line 2: brief reasoning in plain text

Example response if items 2 and 5 leak:
[2, 5]
Items 2 and 5 describe the same motor deficit using synonymous terms.

Example response if nothing leaks:
[]
No items describe the excluded finding."""

        try:
            response = self.llm_client.text_completion(
                prompt=prompt,
                temperature=0.1,
                max_tokens=500
            )
            response_text = response['text'].strip()
            
            # Parse: first line is the JSON array, rest is reasoning
            lines = response_text.split('\n', 1)
            array_line = lines[0].strip()
            reasoning = lines[1].strip() if len(lines) > 1 else ''
            
            # Clean the array line — strip markdown fences if present
            if '```' in array_line:
                # LLM wrapped in markdown — extract from full text instead
                clean = response_text
                if '```json' in clean:
                    clean = clean.split('```json')[1].split('```')[0].strip()
                elif '```' in clean:
                    clean = clean.split('```')[1].split('```')[0].strip()
                array_line = clean.split('\n')[0].strip()
                reasoning = clean.split('\n', 1)[1].strip() if '\n' in clean else ''
            
            # Fix trailing commas in array
            array_line = re.sub(r',\s*\]', ']', array_line)
            
            items_to_remove = set(json.loads(array_line))
            
            if not items_to_remove:
                return clinical_data_by_category, []
            
            if reasoning:
                print(f"   🧹 Context ablation reasoning: {reasoning}")
            
            # Build set of (category, content) to remove
            remove_set = set()
            removed_items = []
            for idx in items_to_remove:
                if 1 <= idx <= len(all_items):
                    item = all_items[idx - 1]
                    remove_set.add((item['category'], item['content']))
                    removed_items.append(item)
            
            # Filter clinical_data_by_category
            filtered = {}
            for category, items in clinical_data_by_category.items():
                kept = [it for it in items if (category, it['content']) not in remove_set]
                if kept:
                    filtered[category] = kept
            
            return filtered, removed_items
            
        except Exception as e:
            print(f"   ⚠️ Context ablation LLM failed: {e}")
            return clinical_data_by_category, []
    
    def _ablate_hpi(
        self,
        hpi: str,
        excluded_content: str,
        excluded_category: str
    ) -> tuple:
        """
        Remove HPI sentences that describe the same finding as the excluded test.
        
        Uses LLM when available (synonym-aware: "weakness" ↔ "diminished strength"),
        falls back to keyword overlap.
        
        Args:
            hpi: The full HPI text
            excluded_content: The ground-truth content being excluded
            excluded_category: Category of the excluded item
            
        Returns:
            Tuple of (ablated_hpi, list_of_removed_sentences)
        """
        if not hpi or not excluded_content:
            return hpi, []
        
        if self.llm_client:
            return self._ablate_hpi_with_llm(hpi, excluded_content, excluded_category)
        else:
            return self._ablate_hpi_keywords(hpi, excluded_content)
    
    def _ablate_hpi_with_llm(
        self,
        hpi: str,
        excluded_content: str,
        excluded_category: str
    ) -> tuple:
        """
        Use LLM to identify which HPI sentences leak the excluded finding.
        
        The LLM understands medical synonyms (weakness ↔ diminished strength,
        numbness ↔ decreased sensation) that keyword matching would miss.
        """
        prompt = f"""You are reviewing a clinical case for data leakage. A specific test finding has been excluded from the case context, but the HPI (History of Present Illness) may still describe the same finding using different words.

EXCLUDED TEST FINDING:
Category: {excluded_category}
Content: {excluded_content}

HPI TEXT:
{hpi}

TASK: Identify any HPI sentences that describe the SAME clinical finding as the excluded test, even if using different words or synonyms. For example:
- "weakness in the right arm" leaks "diminished strength in the right upper extremity"
- "patient reports numbness" leaks "decreased sensation to light touch"
- "fever of 102F" leaks "temperature 38.9°C"
- "severe headache" does NOT leak "CT showed hemorrhage" (symptom vs finding)

RULES:
1. Only flag sentences that directly describe the SAME finding as the excluded content
2. Do NOT flag general symptoms that merely motivated ordering the test
3. Be conservative — when in doubt, keep the sentence
4. A headache motivating an MRI is NOT leakage. But "right arm weakness" in the HPI IS leakage when the excluded finding is "diminished strength in the right arm"

Return ONLY a JSON array of the exact sentences to remove, nothing else.
If no sentences leak the finding, return an empty array: []

Example outputs:
["weakness in the right arm developed over two hours", "numbness spread to her fingers"]
[]

Return ONLY the JSON array:"""
        
        try:
            response = self.llm_client.text_completion(
                prompt=prompt,
                temperature=0.1,
                max_tokens=2000
            )
            response_text = response['text'].strip()
            
            # Strip markdown code blocks
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            # Fix trailing commas and collapse whitespace
            response_text = re.sub(r',\s*([}\]])', r'\1', response_text)
            response_text = ' '.join(response_text.split())
            
            result = json.loads(response_text)
            # Accept both array and object formats
            if isinstance(result, list):
                sentences_to_remove = result
            else:
                sentences_to_remove = result.get('sentences_to_remove', [])
            reasoning = ''
            
            if not sentences_to_remove:
                return hpi, []
            
            # Remove identified sentences from HPI
            ablated = hpi
            actually_removed = []
            for sentence in sentences_to_remove:
                sentence = sentence.strip()
                if not sentence:
                    continue
                # Try exact match
                if sentence in ablated:
                    ablated = ablated.replace(sentence, '')
                    actually_removed.append(sentence)
                else:
                    # Try normalized whitespace match
                    pattern = re.escape(re.sub(r'\s+', ' ', sentence)).replace(r'\ ', r'\s+')
                    match = re.search(pattern, ablated)
                    if match:
                        ablated = ablated[:match.start()] + ablated[match.end():]
                        actually_removed.append(match.group())
            
            # Clean up whitespace artifacts
            ablated = re.sub(r'\n\s*\n\s*\n+', '\n\n', ablated)
            ablated = re.sub(r'- \s*\n', '', ablated)  # empty bullet points
            ablated = ablated.strip()
            
            if actually_removed:
                print(f"   🧹 HPI ablation reasoning: {reasoning}")
            
            return ablated, actually_removed
            
        except Exception as e:
            print(f"   ⚠️ LLM HPI ablation failed: {e}, falling back to keywords")
            return self._ablate_hpi_keywords(hpi, excluded_content)
    
    def _ablate_hpi_keywords(self, hpi: str, excluded_content: str) -> tuple:
        """
        Fallback: remove HPI lines that share multiple key terms with excluded content.
        
        Requires 3+ matching terms (case-insensitive, words >3 chars) to avoid
        over-ablation.
        """
        # Extract meaningful words from excluded content (>3 chars, no stopwords)
        stopwords = {
            'with', 'that', 'this', 'from', 'were', 'been', 'have', 'also',
            'which', 'their', 'than', 'each', 'other', 'into', 'more',
            'some', 'such', 'only', 'over', 'after', 'before', 'between',
            'showed', 'revealed', 'demonstrated', 'indicated', 'found',
            'normal', 'within', 'without', 'approximately', 'consistent',
        }
        excluded_words = set()
        for word in re.findall(r'[a-zA-Z]{4,}', excluded_content.lower()):
            if word not in stopwords:
                excluded_words.add(word)
        
        if len(excluded_words) < 2:
            return hpi, []
        
        # Split HPI into lines (bullet points or sentences)
        lines = hpi.split('\n')
        kept = []
        removed = []
        
        for line in lines:
            line_words = set(re.findall(r'[a-zA-Z]{4,}', line.lower()))
            overlap = line_words & excluded_words
            # Require 3+ matching terms to remove
            if len(overlap) >= 3:
                removed.append(line.strip())
            else:
                kept.append(line)
        
        if removed:
            return '\n'.join(kept), removed
        return hpi, []
    
    def _build_narrative(
        self,
        hpi: str,
        case_description: str,
        case_title: str,
        clinical_data_by_category: Dict[str, List[Dict]]
    ) -> str:
        """
        Build a narrative from structured components.
        
        Args:
            hpi: History of present illness
            case_description: Case description
            case_title: Case title
            clinical_data_by_category: Structured clinical data
        
        Returns:
            Combined narrative text
        """
        sections = []
        
        # Add case title if available
        if case_title:
            sections.append(f"CASE: {case_title}\n")
        
        # Add HPI
        if hpi:
            sections.append("HISTORY OF PRESENT ILLNESS:")
            sections.append(hpi)
            sections.append("")
        
        # Define section order and labels
        section_order = [
            ('patient_background_or_history', 'PATIENT BACKGROUND'),
            ('medications', 'MEDICATIONS'),
            ('vital_signs', 'VITAL SIGNS'),
            ('physical_examination_or_assessments', 'PHYSICAL EXAMINATION'),
            ('laboratory_tests', 'LABORATORY TESTS'),
            ('imaging', 'IMAGING STUDIES'),
            ('procedures', 'PROCEDURES'),
            ('diagnoses', 'DIAGNOSES'),
            ('treatment_plans', 'TREATMENT PLANS')
        ]
        
        # Add clinical data by section
        for category, label in section_order:
            if category in clinical_data_by_category:
                items = clinical_data_by_category[category]
                if items:
                    sections.append(f"{label}:")
                    for item in items:
                        sections.append(f"- {item['content']}")
                    sections.append("")
        
        # Add any remaining categories not in the standard order
        for category, items in clinical_data_by_category.items():
            if not any(cat == category for cat, _ in section_order):
                if items:
                    label = category.replace('_', ' ').upper()
                    sections.append(f"{label}:")
                    for item in items:
                        sections.append(f"- {item['content']}")
                    sections.append("")
        
        return "\n".join(sections)
    
    def get_case_context_length(self, case_id: int) -> Dict[str, int]:
        """
        Get lengths of different case components.
        
        Args:
            case_id: The case ID
        
        Returns:
            Dictionary with component lengths
        """
        hpi_result = self.db_session.execute(
            sql_text("SELECT LENGTH(hpi_raw) as hpi_len, LENGTH(text) as full_text_len FROM cases WHERE id = :case_id"),
            {"case_id": case_id}
        ).fetchone()
        
        clinical_data_count = self.db_session.execute(
            sql_text("SELECT COUNT(*) as count FROM case_clinical_data WHERE case_id = :case_id"),
            {"case_id": case_id}
        ).fetchone()
        
        return {
            'hpi_length': hpi_result.hpi_len or 0,
            'full_text_length': hpi_result.full_text_len or 0,
            'clinical_data_items': clinical_data_count.count
        }
