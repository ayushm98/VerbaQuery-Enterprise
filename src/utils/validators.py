from pathlib import Path
from typing import List


def validate_pdf_file(file_path: Path) -> bool:
    """
    Validate PDF file exists and has correct extension.

    Args:
        file_path: Path to PDF file

    Returns:
        True if valid PDF file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a PDF
    """
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"File must be a PDF, got: {file_path.suffix}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"PDF file is empty: {file_path}")

    return True


def validate_query(query: str, min_length: int = 3, max_length: int = 500) -> str:
    """
    Validate and sanitize user query.

    Args:
        query: User input query
        min_length: Minimum allowed query length
        max_length: Maximum allowed query length

    Returns:
        Sanitized query string

    Raises:
        ValueError: If query is invalid
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    query = query.strip()

    if len(query) < min_length:
        raise ValueError(f"Query too short (min {min_length} characters)")

    if len(query) > max_length:
        raise ValueError(f"Query too long (max {max_length} characters)")

    return query
