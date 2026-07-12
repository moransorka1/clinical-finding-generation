"""
Database utilities for accessing the cliniclue.db SQLite database.

This module provides functions for querying clinical cases, extracting
ground truth data, and accessing case metadata.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.orm import Session
import os
import sys

# Add project root to path to import connection
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the existing database connection
from src.database.connection import SessionLocal, get_db


# Database path (relative to project root)
DB_PATH = Path(__file__).parent.parent.parent / "db" / "cliniclue.db"


def get_sqlalchemy_session() -> Session:
    """
    Get a SQLAlchemy session for ORM operations.
    Used by experiment wrapper and generator.
    """
    return SessionLocal()


@dataclass
class ClinicalCase:
    """Represents a clinical case from the database."""
    case_id: int
    specialty: str
    text: str
    source: str
    primary_diagnosis: Optional[str] = None
    differential_diagnoses: Optional[List[str]] = None
    

@dataclass
class ClinicalDataPoint:
    """Represents a single clinical data point (lab, imaging, exam)."""
    id: int
    case_id: int
    category: str  # 'laboratory_tests', 'imaging', 'physical_examination_or_assessments'
    content: str
    parsed_values: Optional[Dict] = None


class DatabaseConnection:
    """Context manager for database connections."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = None
        
    def __enter__(self):
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()


def get_case_count() -> int:
    """Get total number of cases in database."""
    with DatabaseConnection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM cases")
        return cursor.fetchone()[0]


def get_case_by_id(case_id: int) -> Optional[ClinicalCase]:
    """
    Retrieve a single case by ID with its diagnosis.
    
    Args:
        case_id: The case ID
        
    Returns:
        ClinicalCase object or None if not found
    """
    with DatabaseConnection() as conn:
        query = """
        SELECT 
            c.id,
            c.specialty,
            c.text,
            c.source,
            d.primary_diagnosis,
            d.differential_diagnoses
        FROM cases c
        LEFT JOIN case_final_diagnoses d ON c.id = d.case_id
        WHERE c.id = ?
        """
        cursor = conn.execute(query, (case_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
            
        return ClinicalCase(
            case_id=row['id'],
            specialty=row['specialty'],
            text=row['text'],
            source=row['source'],
            primary_diagnosis=row['primary_diagnosis'],
            differential_diagnoses=(
                row['differential_diagnoses'].split(';') 
                if row['differential_diagnoses'] else None
            )
        )


def get_cases_by_specialty(specialty: str, limit: Optional[int] = None) -> List[ClinicalCase]:
    """
    Retrieve cases for a specific specialty.
    
    Args:
        specialty: Medical specialty name
        limit: Maximum number of cases to return
        
    Returns:
        List of ClinicalCase objects
    """
    with DatabaseConnection() as conn:
        query = """
        SELECT 
            c.id,
            c.specialty,
            c.text,
            c.source,
            d.primary_diagnosis,
            d.differential_diagnoses
        FROM cases c
        LEFT JOIN case_final_diagnoses d ON c.id = d.case_id
        WHERE c.specialty = ?
        """
        
        if limit:
            query += f" LIMIT {limit}"
            
        cursor = conn.execute(query, (specialty,))
        rows = cursor.fetchall()
        
        return [
            ClinicalCase(
                case_id=row['id'],
                specialty=row['specialty'],
                text=row['text'],
                source=row['source'],
                primary_diagnosis=row['primary_diagnosis'],
                differential_diagnoses=(
                    row['differential_diagnoses'].split(';') 
                    if row['differential_diagnoses'] else None
                )
            )
            for row in rows
        ]


def get_clinical_data_by_case(
    case_id: int, 
    categories: Optional[List[str]] = None
) -> List[ClinicalDataPoint]:
    """
    Retrieve all clinical data points for a case.
    
    Args:
        case_id: The case ID
        categories: Filter by categories (e.g., ['laboratory_tests', 'imaging'])
                   If None, returns all categories
        
    Returns:
        List of ClinicalDataPoint objects
    """
    with DatabaseConnection() as conn:
        query = """
        SELECT id, case_id, category, content
        FROM case_clinical_data
        WHERE case_id = ?
        """
        
        params = [case_id]
        
        if categories:
            placeholders = ','.join('?' * len(categories))
            query += f" AND category IN ({placeholders})"
            params.extend(categories)
            
        query += " ORDER BY category, id"
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        return [
            ClinicalDataPoint(
                id=row['id'],
                case_id=row['case_id'],
                category=row['category'],
                content=row['content']
            )
            for row in rows
        ]


def get_all_clinical_data(
    categories: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> List[ClinicalDataPoint]:
    """
    Retrieve all clinical data points across all cases.
    
    Args:
        categories: Filter by categories
        limit: Maximum number of data points to return
        
    Returns:
        List of ClinicalDataPoint objects
    """
    with DatabaseConnection() as conn:
        query = """
        SELECT id, case_id, category, content
        FROM case_clinical_data
        """
        
        params = []
        
        if categories:
            placeholders = ','.join('?' * len(categories))
            query += f" WHERE category IN ({placeholders})"
            params.extend(categories)
            
        query += " ORDER BY case_id, category, id"
        
        if limit:
            query += f" LIMIT {limit}"
            
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        return [
            ClinicalDataPoint(
                id=row['id'],
                case_id=row['case_id'],
                category=row['category'],
                content=row['content']
            )
            for row in rows
        ]


def get_specialty_distribution() -> Dict[str, int]:
    """
    Get count of cases by specialty.
    
    Returns:
        Dictionary mapping specialty name to count
    """
    with DatabaseConnection() as conn:
        query = """
        SELECT specialty, COUNT(*) as count
        FROM cases
        GROUP BY specialty
        ORDER BY count DESC
        """
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        
        return {row['specialty']: row['count'] for row in rows}


def get_category_distribution() -> Dict[str, int]:
    """
    Get count of clinical data points by category.
    
    Returns:
        Dictionary mapping category to count
    """
    with DatabaseConnection() as conn:
        query = """
        SELECT category, COUNT(*) as count
        FROM case_clinical_data
        GROUP BY category
        ORDER BY count DESC
        """
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        
        return {row['category']: row['count'] for row in rows}


def get_cases_with_lab_data(min_lab_count: int = 1) -> List[Tuple[int, str, int]]:
    """
    Get cases that have at least min_lab_count laboratory tests.
    
    Args:
        min_lab_count: Minimum number of lab tests required
        
    Returns:
        List of tuples (case_id, specialty, lab_count)
    """
    with DatabaseConnection() as conn:
        query = """
        SELECT 
            c.id,
            c.specialty,
            COUNT(d.id) as lab_count
        FROM cases c
        INNER JOIN case_clinical_data d ON c.id = d.case_id
        WHERE d.category = 'laboratory_tests'
        GROUP BY c.id, c.specialty
        HAVING COUNT(d.id) >= ?
        ORDER BY lab_count DESC
        """
        cursor = conn.execute(query, (min_lab_count,))
        rows = cursor.fetchall()
        
        return [(row['id'], row['specialty'], row['lab_count']) for row in rows]


def get_cases_with_imaging_data(min_imaging_count: int = 1) -> List[Tuple[int, str, int]]:
    """
    Get cases that have at least min_imaging_count imaging studies.
    
    Args:
        min_imaging_count: Minimum number of imaging studies required
        
    Returns:
        List of tuples (case_id, specialty, imaging_count)
    """
    with DatabaseConnection() as conn:
        query = """
        SELECT 
            c.id,
            c.specialty,
            COUNT(d.id) as imaging_count
        FROM cases c
        INNER JOIN case_clinical_data d ON c.id = d.case_id
        WHERE d.category = 'imaging'
        GROUP BY c.id, c.specialty
        HAVING COUNT(d.id) >= ?
        ORDER BY imaging_count DESC
        """
        cursor = conn.execute(query, (min_imaging_count,))
        rows = cursor.fetchall()
        
        return [(row['id'], row['specialty'], row['imaging_count']) for row in rows]


def execute_custom_query(query: str, params: Optional[Tuple] = None) -> List[sqlite3.Row]:
    """
    Execute a custom SQL query.
    
    Args:
        query: SQL query string
        params: Query parameters
        
    Returns:
        List of Row objects
    """
    with DatabaseConnection() as conn:
        cursor = conn.execute(query, params or ())
        return cursor.fetchall()


# Quick validation
if __name__ == "__main__":
    print(f"Database path: {DB_PATH}")
    print(f"Database exists: {DB_PATH.exists()}")
    
    if DB_PATH.exists():
        print(f"\nTotal cases: {get_case_count()}")
        print(f"\nSpecialty distribution:")
        for specialty, count in get_specialty_distribution().items():
            print(f"  {specialty}: {count}")
        
        print(f"\nCategory distribution:")
        for category, count in get_category_distribution().items():
            print(f"  {category}: {count}")
        
        print(f"\nCases with ≥3 lab tests: {len(get_cases_with_lab_data(min_lab_count=3))}")
        print(f"Cases with ≥3 imaging studies: {len(get_cases_with_imaging_data(min_imaging_count=3))}")
        
        # Test single case retrieval
        case = get_case_by_id(1)
        if case:
            print(f"\nSample case (ID=1):")
            print(f"  Specialty: {case.specialty}")
            print(f"  Source: {case.source}")
            print(f"  Diagnosis: {case.primary_diagnosis}")
            print(f"  Text length: {len(case.text)} chars")
