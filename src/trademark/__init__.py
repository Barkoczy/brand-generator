"""Trademark clearance module for brand name validation.

This module provides functionality for checking brand name candidates
against trademark databases (primarily EUIPO) to assess collision risk.

Usage:
    from src.trademark import TrademarkClearanceChecker, create_checker_from_settings

    # Create checker (uses mock if no API credentials)
    checker = create_checker_from_settings()

    # Check single name
    result = checker.check_name("MYNAME", nice_classes=[9, 35, 42])
    print(f"Risk: {result.risk_level}")  # HIGH, MEDIUM, or LOW

    # Check batch
    batch_result = checker.check_batch(["NAME1", "NAME2"], nice_classes=[9, 42])
    print(f"Safe names: {batch_result.get_safe_candidates()}")
"""

from src.trademark.clearance_checker import (
    TrademarkClearanceChecker,
    create_checker_from_settings,
)
from src.trademark.models import (
    BatchClearanceResult,
    ClearanceResult,
    RiskLevel,
    TrademarkConflict,
)
from src.trademark.similarity import (
    NameSimilarityResult,
    combined_similarity,
    is_confusingly_similar,
    phonetic_similarity,
    string_similarity,
    visual_similarity,
)

__all__ = [
    # Checker
    "TrademarkClearanceChecker",
    "create_checker_from_settings",
    # Models
    "BatchClearanceResult",
    "ClearanceResult",
    "RiskLevel",
    "TrademarkConflict",
    # Similarity
    "NameSimilarityResult",
    "combined_similarity",
    "is_confusingly_similar",
    "phonetic_similarity",
    "string_similarity",
    "visual_similarity",
]
