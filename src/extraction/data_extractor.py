"""
Ground truth data extractor for clinical cases.

This module extracts laboratory tests, imaging studies, and physical examination
findings from the cliniclue.db database to create a ground truth dataset for
evaluating synthetic response generators.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.db_utils import (
    get_case_by_id,
    get_clinical_data_by_case,
    get_all_clinical_data,
    get_cases_with_lab_data,
    get_cases_with_imaging_data,
    ClinicalCase,
    ClinicalDataPoint
)


@dataclass
class GroundTruthEntry:
    """Represents a single ground truth data point for evaluation."""
    id: str  # Unique identifier (e.g., "case_1_lab_3")
    case_id: int
    case_specialty: str
    case_source: str
    primary_diagnosis: str
    category: str  # 'laboratory_tests', 'imaging', 'physical_examination_or_assessments'
    raw_content: str
    case_text: str  # Full case narrative
    case_text_length: int
    
    # Parsed information (if applicable)
    test_name: Optional[str] = None
    value: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    abnormality: Optional[str] = None  # 'high', 'low', 'normal', 'abnormal'
    
    # Metadata
    extraction_timestamp: str = None
    
    def __post_init__(self):
        if self.extraction_timestamp is None:
            self.extraction_timestamp = datetime.now().isoformat()


class GroundTruthExtractor:
    """Extracts ground truth data from clinical cases."""
    
    def __init__(self):
        self.extracted_entries: List[GroundTruthEntry] = []
        
    def extract_all(
        self,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
        min_case_text_length: int = 1000
    ) -> List[GroundTruthEntry]:
        """
        Extract all ground truth data points from the database.
        
        Args:
            categories: Categories to extract (default: lab, imaging, exam)
            limit: Maximum number of entries to extract
            min_case_text_length: Minimum case text length to include
            
        Returns:
            List of GroundTruthEntry objects
        """
        if categories is None:
            categories = [
                'laboratory_tests',
                'imaging',
                'physical_examination_or_assessments'
            ]
        
        print(f"Extracting ground truth data for categories: {categories}")
        
        # Get all clinical data points
        data_points = get_all_clinical_data(categories=categories, limit=limit)
        print(f"Found {len(data_points)} clinical data points")
        
        # Group by case_id
        cases_map: Dict[int, List[ClinicalDataPoint]] = {}
        for dp in data_points:
            if dp.case_id not in cases_map:
                cases_map[dp.case_id] = []
            cases_map[dp.case_id].append(dp)
        
        print(f"Data spans {len(cases_map)} unique cases")
        
        # Process each case
        for case_id, case_data_points in cases_map.items():
            # Get case details
            case = get_case_by_id(case_id)
            if not case:
                print(f"Warning: Could not retrieve case {case_id}")
                continue
                
            # Filter by case text length
            if len(case.text) < min_case_text_length:
                continue
            
            # Extract each data point
            for dp in case_data_points:
                entry = self._create_entry(case, dp)
                self.extracted_entries.append(entry)
        
        print(f"Extracted {len(self.extracted_entries)} ground truth entries")
        return self.extracted_entries
    
    def extract_by_case_ids(
        self,
        case_ids: List[int],
        categories: Optional[List[str]] = None
    ) -> List[GroundTruthEntry]:
        """
        Extract ground truth data for specific cases.
        
        Args:
            case_ids: List of case IDs to extract
            categories: Categories to extract
            
        Returns:
            List of GroundTruthEntry objects
        """
        if categories is None:
            categories = [
                'laboratory_tests',
                'imaging',
                'physical_examination_or_assessments'
            ]
        
        print(f"Extracting ground truth for {len(case_ids)} cases")
        
        for case_id in case_ids:
            case = get_case_by_id(case_id)
            if not case:
                print(f"Warning: Could not retrieve case {case_id}")
                continue
            
            data_points = get_clinical_data_by_case(case_id, categories)
            
            for dp in data_points:
                entry = self._create_entry(case, dp)
                self.extracted_entries.append(entry)
        
        print(f"Extracted {len(self.extracted_entries)} ground truth entries")
        return self.extracted_entries
    
    def _create_entry(
        self,
        case: ClinicalCase,
        data_point: ClinicalDataPoint
    ) -> GroundTruthEntry:
        """Create a GroundTruthEntry from a case and data point."""
        
        # Generate unique ID
        entry_id = f"case_{case.case_id}_{data_point.category}_{data_point.id}"
        
        # Create base entry
        entry = GroundTruthEntry(
            id=entry_id,
            case_id=case.case_id,
            case_specialty=case.specialty,
            case_source=case.source,
            primary_diagnosis=case.primary_diagnosis or "Unknown",
            category=data_point.category,
            raw_content=data_point.content,
            case_text=case.text,
            case_text_length=len(case.text)
        )
        
        # Parse additional information based on category
        if data_point.category == 'laboratory_tests':
            self._parse_lab_test(entry)
        elif data_point.category == 'imaging':
            self._parse_imaging(entry)
        elif data_point.category == 'physical_examination_or_assessments':
            self._parse_examination(entry)
        
        return entry
    
    def _parse_lab_test(self, entry: GroundTruthEntry):
        """
        Parse laboratory test information from raw content.
        
        Common patterns:
        - "Test name: value unit (reference range)"
        - "Test name of value unit"
        - "Elevated/decreased test name at value"
        """
        content = entry.raw_content
        
        # Try to extract test name (usually at the beginning or after certain keywords)
        # This is a simplified parser - can be enhanced
        
        # Pattern 1: "Test: value unit"
        match = re.search(r'([^:]+):\s*([\d.]+)\s*([a-zA-Z/]+)', content, re.IGNORECASE)
        if match:
            entry.test_name = match.group(1).strip()
            entry.value = match.group(2).strip()
            entry.unit = match.group(3).strip()
        
        # Pattern 2: "value unit" without test name
        if not entry.value:
            match = re.search(r'([\d.]+)\s*([a-zA-Z/]+)', content)
            if match:
                entry.value = match.group(1).strip()
                entry.unit = match.group(2).strip()
        
        # Detect abnormality keywords
        content_lower = content.lower()
        if any(word in content_lower for word in ['elevated', 'high', 'increased', 'above']):
            entry.abnormality = 'high'
        elif any(word in content_lower for word in ['low', 'decreased', 'reduced', 'below']):
            entry.abnormality = 'low'
        elif any(word in content_lower for word in ['normal', 'within normal']):
            entry.abnormality = 'normal'
        elif 'abnormal' in content_lower:
            entry.abnormality = 'abnormal'
    
    def _parse_imaging(self, entry: GroundTruthEntry):
        """Parse imaging study information from raw content."""
        content = entry.raw_content
        
        # Extract modality (CT, MRI, X-ray, etc.)
        modalities = ['CT', 'MRI', 'X-ray', 'ultrasound', 'PET', 'angiography']
        for modality in modalities:
            if modality.lower() in content.lower():
                entry.test_name = modality
                break
        
        # Detect abnormality
        content_lower = content.lower()
        abnormal_keywords = [
            'showed', 'revealed', 'demonstrated', 'mass', 'lesion',
            'abnormal', 'finding', 'nodule', 'opacity'
        ]
        if any(keyword in content_lower for keyword in abnormal_keywords):
            entry.abnormality = 'abnormal'
        elif any(word in content_lower for word in ['normal', 'unremarkable', 'negative']):
            entry.abnormality = 'normal'
    
    def _parse_examination(self, entry: GroundTruthEntry):
        """Parse physical examination finding from raw content."""
        content = entry.raw_content
        
        # Detect abnormality
        content_lower = content.lower()
        if any(word in content_lower for word in ['normal', 'unremarkable', 'negative']):
            entry.abnormality = 'normal'
        else:
            # Assume any documented finding is potentially abnormal
            entry.abnormality = 'abnormal'
    
    def get_statistics(self) -> Dict:
        """Get statistics about extracted data."""
        if not self.extracted_entries:
            return {}
        
        stats = {
            'total_entries': len(self.extracted_entries),
            'unique_cases': len(set(e.case_id for e in self.extracted_entries)),
            'by_category': {},
            'by_specialty': {},
            'by_source': {},
            'by_abnormality': {},
        }
        
        for entry in self.extracted_entries:
            # Count by category
            cat = entry.category
            stats['by_category'][cat] = stats['by_category'].get(cat, 0) + 1
            
            # Count by specialty
            spec = entry.case_specialty
            stats['by_specialty'][spec] = stats['by_specialty'].get(spec, 0) + 1
            
            # Count by source
            src = entry.case_source
            stats['by_source'][src] = stats['by_source'].get(src, 0) + 1
            
            # Count by abnormality
            if entry.abnormality:
                abn = entry.abnormality
                stats['by_abnormality'][abn] = stats['by_abnormality'].get(abn, 0) + 1
        
        return stats
    
    def save_to_json(self, output_path: Path):
        """Save extracted data to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': {
                'extraction_timestamp': datetime.now().isoformat(),
                'total_entries': len(self.extracted_entries),
                'statistics': self.get_statistics()
            },
            'entries': [asdict(entry) for entry in self.extracted_entries]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.extracted_entries)} entries to {output_path}")
    
    def load_from_json(self, input_path: Path) -> List[GroundTruthEntry]:
        """Load extracted data from JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.extracted_entries = [
            GroundTruthEntry(**entry) for entry in data['entries']
        ]
        
        print(f"Loaded {len(self.extracted_entries)} entries from {input_path}")
        return self.extracted_entries


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Ground Truth Data Extractor")
    print("=" * 80)
    
    extractor = GroundTruthExtractor()
    
    # Extract first 100 entries for testing
    entries = extractor.extract_all(limit=100)
    
    print("\n" + "=" * 80)
    print("Statistics:")
    print("=" * 80)
    stats = extractor.get_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\n" + "=" * 80)
    print("Sample Entries (first 3):")
    print("=" * 80)
    for i, entry in enumerate(entries[:3]):
        print(f"\nEntry {i+1}: {entry.id}")
        print(f"  Case: #{entry.case_id} ({entry.case_specialty})")
        print(f"  Category: {entry.category}")
        print(f"  Content: {entry.raw_content[:100]}...")
        if entry.test_name:
            print(f"  Test: {entry.test_name}")
        if entry.value:
            print(f"  Value: {entry.value} {entry.unit or ''}")
        if entry.abnormality:
            print(f"  Abnormality: {entry.abnormality}")
    
    # Save to file
    output_dir = Path(__file__).parent.parent.parent / "data"
    output_path = output_dir / "ground_truth_sample.json"
    extractor.save_to_json(output_path)
    
    print(f"\n✅ Complete! Extracted {len(entries)} ground truth entries")
