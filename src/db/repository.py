"""SQLite repository for storing candidate evaluations."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.config import Settings
from src.generator.candidate_generator import Candidate
from src.llm.claude_scorer import LLMEvaluation
from src.scoring.heuristic_scorer import HeuristicScore


@dataclass
class StoredCandidate:
    """A candidate stored in the database with full evaluation data."""

    id: int
    name: str
    pattern: str
    heuristic_score: float | None
    llm_score: int | None
    category: str | None
    pros: list[str]
    cons: list[str]
    flags: list[str]
    recommendation: str | None
    status: str  # new, favorite, rejected
    created_at: datetime
    updated_at: datetime
    # Trademark clearance fields
    trademark_risk: str | None = None  # HIGH, MEDIUM, LOW
    trademark_checked_at: datetime | None = None
    trademark_conflicts: list[dict] | None = None

    @property
    def best_score(self) -> float:
        """Return the best available score (LLM preferred over heuristic)."""
        if self.llm_score is not None:
            return float(self.llm_score)
        if self.heuristic_score is not None:
            return self.heuristic_score
        return 0.0

    @property
    def is_trademark_safe(self) -> bool:
        """Check if trademark clearance passed (LOW risk)."""
        return self.trademark_risk == "LOW"

    @property
    def trademark_conflict_count(self) -> int:
        """Number of trademark conflicts found."""
        return len(self.trademark_conflicts) if self.trademark_conflicts else 0


class CandidateRepository:
    """SQLite repository for managing candidate names.

    Provides CRUD operations for storing, retrieving, and filtering
    brand name candidates with their evaluation scores.
    """

    def __init__(self, settings: Settings):
        """Initialize repository with settings.

        Args:
            settings: Configuration settings with database path
        """
        self.db_path = Path(settings.database.path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    pattern TEXT NOT NULL,
                    heuristic_score REAL,
                    llm_score INTEGER,
                    category TEXT,
                    pros TEXT,
                    cons TEXT,
                    flags TEXT,
                    recommendation TEXT,
                    status TEXT DEFAULT 'new',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    trademark_risk TEXT,
                    trademark_checked_at TIMESTAMP,
                    trademark_conflicts TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_candidates_name ON candidates(name)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_candidates_llm_score ON candidates(llm_score)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status)
            """)

            # Add columns to existing tables (migration)
            self._migrate_trademark_columns(conn)

            # Create trademark index after migration ensures column exists
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_candidates_trademark_risk
                ON candidates(trademark_risk)
            """)

            conn.commit()

    def _migrate_trademark_columns(self, conn: sqlite3.Connection) -> None:
        """Add trademark columns if they don't exist (migration for existing DBs)."""
        cursor = conn.execute("PRAGMA table_info(candidates)")
        columns = {row["name"] for row in cursor.fetchall()}

        if "trademark_risk" not in columns:
            conn.execute("ALTER TABLE candidates ADD COLUMN trademark_risk TEXT")
        if "trademark_checked_at" not in columns:
            conn.execute("ALTER TABLE candidates ADD COLUMN trademark_checked_at TIMESTAMP")
        if "trademark_conflicts" not in columns:
            conn.execute("ALTER TABLE candidates ADD COLUMN trademark_conflicts TEXT")

    def _row_to_stored_candidate(self, row: sqlite3.Row) -> StoredCandidate:
        """Convert database row to StoredCandidate."""
        # Handle trademark_checked_at (may not exist in old DBs)
        trademark_checked_at = None
        try:
            if row["trademark_checked_at"]:
                trademark_checked_at = datetime.fromisoformat(row["trademark_checked_at"])
        except (KeyError, IndexError):
            pass

        # Handle trademark_conflicts JSON
        trademark_conflicts = None
        try:
            if row["trademark_conflicts"]:
                trademark_conflicts = json.loads(row["trademark_conflicts"])
        except (KeyError, IndexError):
            pass

        # Handle trademark_risk
        trademark_risk = None
        try:
            trademark_risk = row["trademark_risk"]
        except (KeyError, IndexError):
            pass

        return StoredCandidate(
            id=row["id"],
            name=row["name"],
            pattern=row["pattern"],
            heuristic_score=row["heuristic_score"],
            llm_score=row["llm_score"],
            category=row["category"],
            pros=json.loads(row["pros"]) if row["pros"] else [],
            cons=json.loads(row["cons"]) if row["cons"] else [],
            flags=json.loads(row["flags"]) if row["flags"] else [],
            recommendation=row["recommendation"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            trademark_risk=trademark_risk,
            trademark_checked_at=trademark_checked_at,
            trademark_conflicts=trademark_conflicts,
        )

    def save_candidate(
        self,
        candidate: Candidate,
        heuristic: HeuristicScore | None = None,
        llm_eval: LLMEvaluation | None = None,
    ) -> int:
        """Save a candidate with optional evaluation scores.

        Args:
            candidate: The candidate to save
            heuristic: Optional heuristic score
            llm_eval: Optional LLM evaluation

        Returns:
            The ID of the saved candidate
        """
        with self._get_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO candidates (name, pattern, heuristic_score, llm_score, category,
                                           pros, cons, flags, recommendation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.name,
                        candidate.pattern,
                        heuristic.score if heuristic else None,
                        llm_eval.score if llm_eval else None,
                        llm_eval.category if llm_eval else None,
                        json.dumps(llm_eval.pros) if llm_eval else None,
                        json.dumps(llm_eval.cons) if llm_eval else None,
                        json.dumps(llm_eval.flags) if llm_eval else None,
                        llm_eval.recommendation if llm_eval else None,
                    ),
                )
                conn.commit()
                return cursor.lastrowid or 0
            except sqlite3.IntegrityError:
                # Name already exists, update instead
                return self.update_candidate(candidate.name, heuristic, llm_eval)

    def update_candidate(
        self,
        name: str,
        heuristic: HeuristicScore | None = None,
        llm_eval: LLMEvaluation | None = None,
    ) -> int:
        """Update an existing candidate with new evaluation scores.

        Args:
            name: Name of the candidate to update
            heuristic: Optional heuristic score
            llm_eval: Optional LLM evaluation

        Returns:
            The ID of the updated candidate
        """
        with self._get_connection() as conn:
            updates = ["updated_at = CURRENT_TIMESTAMP"]
            params: list = []

            if heuristic:
                updates.append("heuristic_score = ?")
                params.append(heuristic.score)

            if llm_eval:
                updates.append("llm_score = ?")
                updates.append("category = ?")
                updates.append("pros = ?")
                updates.append("cons = ?")
                updates.append("flags = ?")
                updates.append("recommendation = ?")
                params.extend([
                    llm_eval.score,
                    llm_eval.category,
                    json.dumps(llm_eval.pros),
                    json.dumps(llm_eval.cons),
                    json.dumps(llm_eval.flags),
                    llm_eval.recommendation,
                ])

            params.append(name)

            conn.execute(
                f"UPDATE candidates SET {', '.join(updates)} WHERE name = ?",
                params,
            )
            conn.commit()

            cursor = conn.execute("SELECT id FROM candidates WHERE name = ?", (name,))
            row = cursor.fetchone()
            return row["id"] if row else 0

    def set_status(self, name: str, status: str) -> None:
        """Set the status of a candidate (new, favorite, rejected).

        Args:
            name: Name of the candidate
            status: New status
        """
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE candidates SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE name = ?",
                (status, name),
            )
            conn.commit()

    def get_by_name(self, name: str) -> StoredCandidate | None:
        """Get a candidate by name.

        Args:
            name: Name to search for

        Returns:
            StoredCandidate or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM candidates WHERE name = ?", (name,))
            row = cursor.fetchone()
            return self._row_to_stored_candidate(row) if row else None

    def exists(self, name: str) -> bool:
        """Check if a candidate name already exists.

        Args:
            name: Name to check

        Returns:
            True if exists
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT 1 FROM candidates WHERE name = ?", (name,))
            return cursor.fetchone() is not None

    def get_top(self, limit: int = 50, min_score: int | None = None) -> list[StoredCandidate]:
        """Get top candidates by LLM score.

        Args:
            limit: Maximum number of results
            min_score: Minimum LLM score filter

        Returns:
            List of top candidates
        """
        query = "SELECT * FROM candidates WHERE llm_score IS NOT NULL"
        params: list = []

        if min_score is not None:
            query += " AND llm_score >= ?"
            params.append(min_score)

        query += " ORDER BY llm_score DESC, heuristic_score DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_stored_candidate(row) for row in cursor.fetchall()]

    def get_by_status(self, status: str) -> list[StoredCandidate]:
        """Get candidates by status.

        Args:
            status: Status to filter by

        Returns:
            List of matching candidates
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM candidates WHERE status = ? ORDER BY llm_score DESC",
                (status,),
            )
            return [self._row_to_stored_candidate(row) for row in cursor.fetchall()]

    def get_unevaluated(self, limit: int = 100) -> list[StoredCandidate]:
        """Get candidates without LLM evaluation.

        Args:
            limit: Maximum number of results

        Returns:
            List of unevaluated candidates
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM candidates
                WHERE llm_score IS NULL
                ORDER BY heuristic_score DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [self._row_to_stored_candidate(row) for row in cursor.fetchall()]

    def count(self, status: str | None = None) -> int:
        """Count candidates, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            Count of matching candidates
        """
        with self._get_connection() as conn:
            if status:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM candidates WHERE status = ?", (status,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM candidates")
            return cursor.fetchone()[0]

    def count_above_score(self, min_score: int) -> int:
        """Count candidates with LLM score at or above threshold.

        Args:
            min_score: Minimum score threshold

        Returns:
            Count of matching candidates
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM candidates WHERE llm_score >= ?", (min_score,)
            )
            return cursor.fetchone()[0]

    def get_all_scored(
        self, limit: int = 1000, min_heuristic: float | None = None
    ) -> list[StoredCandidate]:
        """Get all scored candidates (LLM or heuristic).

        Args:
            limit: Maximum number of results
            min_heuristic: Minimum heuristic score filter

        Returns:
            List of candidates sorted by best score
        """
        query = """
            SELECT * FROM candidates
            WHERE (llm_score IS NOT NULL OR heuristic_score IS NOT NULL)
        """
        params: list = []

        if min_heuristic is not None:
            query += " AND (llm_score >= ? OR heuristic_score >= ?)"
            params.extend([min_heuristic, min_heuristic])

        query += """
            ORDER BY
                CASE WHEN llm_score IS NOT NULL THEN llm_score ELSE 0 END DESC,
                CASE WHEN heuristic_score IS NOT NULL THEN heuristic_score ELSE 0 END DESC
            LIMIT ?
        """
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_stored_candidate(row) for row in cursor.fetchall()]

    def save_trademark_result(
        self,
        name: str,
        risk_level: str,
        conflicts: list[dict] | None = None,
    ) -> None:
        """Save trademark clearance result for a candidate.

        Args:
            name: Name of the candidate
            risk_level: Risk level (HIGH, MEDIUM, LOW)
            conflicts: Optional list of conflict dictionaries
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE candidates
                SET trademark_risk = ?,
                    trademark_checked_at = CURRENT_TIMESTAMP,
                    trademark_conflicts = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE name = ?
                """,
                (risk_level, json.dumps(conflicts) if conflicts else None, name),
            )
            conn.commit()

    def get_unchecked_trademarks(self, limit: int = 100) -> list[StoredCandidate]:
        """Get candidates without trademark clearance check.

        Args:
            limit: Maximum number of results

        Returns:
            List of candidates without trademark check
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM candidates
                WHERE trademark_risk IS NULL
                ORDER BY
                    CASE WHEN llm_score IS NOT NULL THEN llm_score ELSE 0 END DESC,
                    CASE WHEN heuristic_score IS NOT NULL THEN heuristic_score ELSE 0 END DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [self._row_to_stored_candidate(row) for row in cursor.fetchall()]

    def get_trademark_safe(self, limit: int = 50) -> list[StoredCandidate]:
        """Get candidates with LOW trademark risk.

        Args:
            limit: Maximum number of results

        Returns:
            List of candidates with LOW trademark risk
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM candidates
                WHERE trademark_risk = 'LOW'
                ORDER BY
                    CASE WHEN llm_score IS NOT NULL THEN llm_score ELSE 0 END DESC,
                    CASE WHEN heuristic_score IS NOT NULL THEN heuristic_score ELSE 0 END DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [self._row_to_stored_candidate(row) for row in cursor.fetchall()]

    def count_by_trademark_risk(self) -> dict[str, int]:
        """Count candidates by trademark risk level.

        Returns:
            Dictionary with counts: {"HIGH": n, "MEDIUM": n, "LOW": n, "unchecked": n}
        """
        with self._get_connection() as conn:
            result = {
                "HIGH": 0,
                "MEDIUM": 0,
                "LOW": 0,
                "unchecked": 0,
            }

            cursor = conn.execute(
                "SELECT trademark_risk, COUNT(*) FROM candidates GROUP BY trademark_risk"
            )
            for row in cursor.fetchall():
                risk = row[0]
                count = row[1]
                if risk is None:
                    result["unchecked"] = count
                elif risk in result:
                    result[risk] = count

            return result
