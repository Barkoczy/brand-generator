"""Trademark clearance checker for brand name validation.

This module provides the main business logic for checking candidate
brand names against trademark databases and assessing collision risk.
"""

from typing import Callable

from src.trademark.euipo_client import EuipoApiError, EuipoClient, MockEuipoClient
from src.trademark.models import (
    BatchClearanceResult,
    ClearanceResult,
    RiskLevel,
    TrademarkConflict,
)
from src.trademark.similarity import combined_similarity

# Status values considered "active" (potential conflict)
ACTIVE_STATUSES = frozenset({"REGISTERED", "APPLIED", "PENDING", "PUBLISHED", "OPPOSITION"})


class TrademarkClearanceChecker:
    """Checker for trademark clearance and collision risk assessment.

    This class combines API searches with similarity analysis to determine
    the risk level for registering a candidate brand name.

    Usage:
        checker = TrademarkClearanceChecker(
            client_id="your-id",
            client_secret="your-secret",
        )
        result = checker.check_name("MYNAME", nice_classes=[9, 42])
        print(f"Risk: {result.risk_level}")
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        sandbox: bool = True,
        similarity_threshold: float = 0.7,
        use_mock: bool = False,
    ):
        """Initialize clearance checker.

        Args:
            client_id: EUIPO OAuth client ID (optional if using mock)
            client_secret: EUIPO OAuth client secret (optional if using mock)
            sandbox: Use EUIPO sandbox environment
            similarity_threshold: Minimum similarity to consider as conflict
            use_mock: Use mock client instead of real API
        """
        self._similarity_threshold = similarity_threshold

        if use_mock or not (client_id and client_secret):
            self._client: EuipoClient | MockEuipoClient = MockEuipoClient()
            self._using_mock = True
        else:
            self._client = EuipoClient(
                client_id=client_id,
                client_secret=client_secret,
                sandbox=sandbox,
            )
            self._using_mock = False

    @property
    def is_using_mock(self) -> bool:
        """Check if using mock client."""
        return self._using_mock

    def check_name(
        self,
        candidate_name: str,
        nice_classes: list[int],
    ) -> ClearanceResult:
        """Check a candidate name for trademark conflicts.

        Args:
            candidate_name: The brand name to check
            nice_classes: Nice classification classes to search

        Returns:
            ClearanceResult with conflicts and risk assessment
        """
        conflicts: list[TrademarkConflict] = []
        overlapping_classes: set[int] = set()

        try:
            # Search for potentially conflicting trademarks
            raw_results = self._client.search_trademarks(
                name=candidate_name,
                nice_classes=nice_classes,
                status_filter=list(ACTIVE_STATUSES),
                max_results=200,
            )

            # Process results
            for item in raw_results:
                conflict = self._process_trademark_result(
                    candidate_name, item, nice_classes
                )
                if conflict:
                    conflicts.append(conflict)
                    overlapping_classes.update(
                        set(conflict.nice_classes) & set(nice_classes)
                    )

        except EuipoApiError:
            # If API fails, return result with warning
            return ClearanceResult(
                candidate_name=candidate_name,
                conflicts=[],
                risk_level="MEDIUM",
                overlapping_classes=[],
                recommendation="Nepodařilo se zkontrolovat databázi ochranných známek. "
                "Doporučujeme manuální kontrolu na TMview.",
                checked_classes=nice_classes,
                source="EUIPO (error)",
            )

        # Determine risk level
        risk = self._determine_risk(conflicts)

        # Build recommendation
        recommendation = self._build_recommendation(candidate_name, risk, conflicts)

        return ClearanceResult(
            candidate_name=candidate_name,
            conflicts=conflicts,
            risk_level=risk,
            overlapping_classes=sorted(overlapping_classes),
            recommendation=recommendation,
            checked_classes=nice_classes,
            source="MOCK" if self._using_mock else "EUIPO",
        )

    def _process_trademark_result(
        self,
        candidate_name: str,
        item: dict,
        nice_classes: list[int],
    ) -> TrademarkConflict | None:
        """Process a single trademark result and determine if it's a conflict.

        Args:
            candidate_name: The candidate name being checked
            item: Raw API result
            nice_classes: Nice classes we're checking against

        Returns:
            TrademarkConflict if it's a relevant conflict, None otherwise
        """
        # Parse the trademark data
        conflict = self._client.parse_trademark_result(item)

        # Calculate similarity
        sim_result = combined_similarity(candidate_name, conflict.name)
        conflict.similarity_score = sim_result.combined_score

        # Filter by similarity threshold
        if sim_result.combined_score < self._similarity_threshold * 0.7:
            # Very low similarity - skip
            return None

        # Check class overlap
        candidate_classes = set(nice_classes)
        trademark_classes = set(conflict.nice_classes)
        overlap = candidate_classes & trademark_classes

        # If no class overlap and similarity is not very high, skip
        if not overlap and sim_result.combined_score < 0.85:
            return None

        return conflict

    def _determine_risk(self, conflicts: list[TrademarkConflict]) -> RiskLevel:
        """Determine overall risk level based on conflicts.

        Risk levels:
        - HIGH: Very similar active trademark in same classes
        - MEDIUM: Moderately similar trademark or same name in related classes
        - LOW: No significant conflicts found

        Args:
            conflicts: List of found conflicts

        Returns:
            Risk level string
        """
        if not conflicts:
            return "LOW"

        # Check for active conflicts only
        active_conflicts = [c for c in conflicts if c.status.upper() in ACTIVE_STATUSES]
        if not active_conflicts:
            return "LOW"

        max_similarity = max(c.similarity_score for c in active_conflicts)

        # Very high similarity (>90%) = HIGH risk
        if max_similarity > 0.9:
            return "HIGH"

        # Check for exact or near-exact matches
        exact_match = any(c.similarity_score >= 0.95 for c in active_conflicts)
        if exact_match:
            return "HIGH"

        # High similarity (>80%) = HIGH risk
        if max_similarity > 0.8:
            return "HIGH"

        # Medium similarity (>70%) = MEDIUM risk
        if max_similarity > 0.7:
            return "MEDIUM"

        # Multiple moderate conflicts = MEDIUM risk
        moderate_conflicts = [c for c in active_conflicts if c.similarity_score > 0.5]
        if len(moderate_conflicts) >= 3:
            return "MEDIUM"

        return "LOW"

    def _build_recommendation(
        self,
        candidate_name: str,
        risk: RiskLevel,
        conflicts: list[TrademarkConflict],
    ) -> str:
        """Build human-readable recommendation based on results.

        Args:
            candidate_name: The checked name
            risk: Determined risk level
            conflicts: Found conflicts

        Returns:
            Recommendation text in Czech
        """
        if risk == "HIGH":
            active = [c for c in conflicts if c.status.upper() in ACTIVE_STATUSES]
            if active:
                top_conflict = max(active, key=lambda c: c.similarity_score)
                return (
                    f"Nalezena velmi podobná ochranná známka '{top_conflict.name}' "
                    f"(podobnost {top_conflict.similarity_percent}%, "
                    f"status: {top_conflict.status}). "
                    f"Riziko kolize je vysoké. Tento název pravděpodobně není vhodný "
                    f"pro registraci ochranné známky v daných třídách."
                )
            return (
                "Nalezeny velmi podobné ochranné známky v překrývajících se třídách. "
                "Riziko kolize je vysoké, tento název pravděpodobně není vhodný pro registraci."
            )

        if risk == "MEDIUM":
            return (
                "Nalezeny podobné ochranné známky v překrývajících se třídách. "
                "Riziko opozice nebo kolize je střední. Doporučujeme konzultaci "
                "s patentovým zástupcem a případnou úpravu názvu."
            )

        # LOW risk
        if conflicts:
            return (
                "Byly nalezeny jen vzdálenější podobné známky. "
                "Název vypadá z hlediska kolizí relativně bezpečně, "
                "ale finální posouzení by měl provést odborník."
            )

        return (
            "Nebyla nalezena žádná zjevná kolize v daných třídách Nice. "
            "Šance na registraci z hlediska relativních důvodů vypadá dobře, "
            "ale absolutní důvody (popisnost, klamavost) a strategii "
            "je stále vhodné konzultovat s odborníkem."
        )

    def check_batch(
        self,
        candidate_names: list[str],
        nice_classes: list[int],
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> BatchClearanceResult:
        """Check multiple candidate names in batch.

        Args:
            candidate_names: List of names to check
            nice_classes: Nice classes to check against
            progress_callback: Optional callback(name, current, total)

        Returns:
            BatchClearanceResult with all results
        """
        results: list[ClearanceResult] = []
        total = len(candidate_names)

        for i, name in enumerate(candidate_names, 1):
            if progress_callback:
                progress_callback(name, i, total)

            result = self.check_name(name, nice_classes)
            results.append(result)

        return BatchClearanceResult(results=results)

    def test_connection(self) -> tuple[bool, str]:
        """Test API connection.

        Returns:
            Tuple of (success, message)
        """
        return self._client.test_connection()


def create_checker_from_settings(
    client_id: str | None = None,
    client_secret: str | None = None,
    sandbox: bool = True,
    similarity_threshold: float = 0.7,
) -> TrademarkClearanceChecker:
    """Factory function to create a clearance checker.

    If credentials are not provided, creates a mock checker
    suitable for development and testing.

    Args:
        client_id: EUIPO client ID (or from env var EUIPO_CLIENT_ID)
        client_secret: EUIPO client secret (or from env var EUIPO_CLIENT_SECRET)
        sandbox: Use sandbox environment
        similarity_threshold: Similarity threshold for conflicts

    Returns:
        Configured TrademarkClearanceChecker
    """
    import os

    client_id = client_id or os.getenv("EUIPO_CLIENT_ID")
    client_secret = client_secret or os.getenv("EUIPO_CLIENT_SECRET")

    use_mock = not (client_id and client_secret)

    return TrademarkClearanceChecker(
        client_id=client_id,
        client_secret=client_secret,
        sandbox=sandbox,
        similarity_threshold=similarity_threshold,
        use_mock=use_mock,
    )
