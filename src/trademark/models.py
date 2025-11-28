"""Data models for trademark clearance functionality."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

# Risk level type for trademark clearance results
RiskLevel = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class TrademarkConflict:
    """Represents a potential trademark conflict found in database search.

    Attributes:
        name: Name of the conflicting trademark
        application_number: Official application/registration number
        status: Current status (REGISTERED, APPLIED, PENDING, EXPIRED, etc.)
        nice_classes: List of Nice classification classes
        owner: Name of the trademark owner (if available)
        similarity_score: Similarity score to candidate name (0.0-1.0)
        filing_date: Date when trademark was filed (if available)
        registration_date: Date when trademark was registered (if available)
    """

    name: str
    application_number: str
    status: str
    nice_classes: list[int]
    owner: str | None = None
    similarity_score: float = 0.0
    filing_date: datetime | None = None
    registration_date: datetime | None = None

    @property
    def is_active(self) -> bool:
        """Check if trademark is currently active/valid."""
        active_statuses = {"REGISTERED", "APPLIED", "PENDING", "PUBLISHED"}
        return self.status.upper() in active_statuses

    @property
    def similarity_percent(self) -> int:
        """Return similarity score as percentage."""
        return int(self.similarity_score * 100)


@dataclass
class ClearanceResult:
    """Result of trademark clearance check for a candidate name.

    Attributes:
        candidate_name: The name that was checked
        conflicts: List of found trademark conflicts
        risk_level: Overall risk assessment (HIGH, MEDIUM, LOW)
        overlapping_classes: Nice classes with potential conflicts
        recommendation: Human-readable recommendation text
        checked_classes: Nice classes that were checked
        search_timestamp: When the search was performed
        source: Data source (e.g., "EUIPO", "WIPO", "USPTO")
    """

    candidate_name: str
    conflicts: list[TrademarkConflict] = field(default_factory=list)
    risk_level: RiskLevel = "LOW"
    overlapping_classes: list[int] = field(default_factory=list)
    recommendation: str = ""
    checked_classes: list[int] = field(default_factory=list)
    search_timestamp: datetime = field(default_factory=datetime.now)
    source: str = "EUIPO"

    @property
    def conflict_count(self) -> int:
        """Number of conflicts found."""
        return len(self.conflicts)

    @property
    def has_high_risk_conflict(self) -> bool:
        """Check if any conflict has very high similarity (>0.9)."""
        return any(c.similarity_score > 0.9 for c in self.conflicts)

    @property
    def active_conflicts(self) -> list[TrademarkConflict]:
        """Return only active/valid trademark conflicts."""
        return [c for c in self.conflicts if c.is_active]

    @property
    def max_similarity(self) -> float:
        """Maximum similarity score among all conflicts."""
        if not self.conflicts:
            return 0.0
        return max(c.similarity_score for c in self.conflicts)

    def get_conflicts_by_class(self, nice_class: int) -> list[TrademarkConflict]:
        """Get conflicts that overlap with a specific Nice class."""
        return [c for c in self.conflicts if nice_class in c.nice_classes]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "candidate_name": self.candidate_name,
            "risk_level": self.risk_level,
            "conflict_count": self.conflict_count,
            "overlapping_classes": self.overlapping_classes,
            "recommendation": self.recommendation,
            "checked_classes": self.checked_classes,
            "max_similarity": self.max_similarity,
            "source": self.source,
            "conflicts": [
                {
                    "name": c.name,
                    "application_number": c.application_number,
                    "status": c.status,
                    "nice_classes": c.nice_classes,
                    "owner": c.owner,
                    "similarity_score": c.similarity_score,
                }
                for c in self.conflicts
            ],
        }


@dataclass
class BatchClearanceResult:
    """Result of batch trademark clearance check for multiple names.

    Attributes:
        results: List of individual clearance results
        total_checked: Total number of names checked
        high_risk_count: Number of names with HIGH risk
        medium_risk_count: Number of names with MEDIUM risk
        low_risk_count: Number of names with LOW risk
    """

    results: list[ClearanceResult] = field(default_factory=list)

    @property
    def total_checked(self) -> int:
        """Total number of names checked."""
        return len(self.results)

    @property
    def high_risk_count(self) -> int:
        """Number of names with HIGH risk."""
        return sum(1 for r in self.results if r.risk_level == "HIGH")

    @property
    def medium_risk_count(self) -> int:
        """Number of names with MEDIUM risk."""
        return sum(1 for r in self.results if r.risk_level == "MEDIUM")

    @property
    def low_risk_count(self) -> int:
        """Number of names with LOW risk."""
        return sum(1 for r in self.results if r.risk_level == "LOW")

    def get_by_risk(self, risk_level: RiskLevel) -> list[ClearanceResult]:
        """Get results filtered by risk level."""
        return [r for r in self.results if r.risk_level == risk_level]

    def get_safe_candidates(self) -> list[str]:
        """Get list of candidate names with LOW risk."""
        return [r.candidate_name for r in self.results if r.risk_level == "LOW"]

    def to_summary_dict(self) -> dict:
        """Get summary statistics as dictionary."""
        return {
            "total_checked": self.total_checked,
            "high_risk": self.high_risk_count,
            "medium_risk": self.medium_risk_count,
            "low_risk": self.low_risk_count,
            "safe_candidates": self.get_safe_candidates(),
        }
