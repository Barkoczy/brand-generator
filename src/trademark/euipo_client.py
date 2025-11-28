"""EUIPO API client for trademark database searches.

This module provides a client for the EUIPO (European Union Intellectual
Property Office) Trademark Search API to check for existing trademarks
that might conflict with candidate brand names.

API Documentation: https://dev.euipo.europa.eu/product/trademark-search_100

Note: The actual endpoint URLs and response structures should be verified
against the official EUIPO API documentation and Swagger specification.
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

from src.trademark.models import TrademarkConflict


@dataclass
class EuipoToken:
    """OAuth token for EUIPO API authentication.

    Attributes:
        access_token: The bearer token for API requests
        expires_at: Unix timestamp when token expires
        token_type: Type of token (usually "Bearer")
    """

    access_token: str
    expires_at: float
    token_type: str = "Bearer"

    @property
    def is_expired(self) -> bool:
        """Check if token is expired (with 60s buffer)."""
        return time.time() >= (self.expires_at - 60)


class EuipoApiError(Exception):
    """Exception raised for EUIPO API errors."""

    def __init__(self, message: str, status_code: int | None = None, response: Any = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)


class EuipoClient:
    """Client for EUIPO Trademark Search API.

    This client handles authentication (OAuth 2.0) and provides methods
    for searching the EUIPO trademark database.

    Usage:
        client = EuipoClient(
            client_id="your-client-id",
            client_secret="your-client-secret",
            sandbox=True  # Use sandbox for testing
        )
        results = client.search_trademarks("MYNAME", nice_classes=[9, 42])
    """

    # API endpoints from official EUIPO documentation
    # https://dev.euipo.europa.eu/security
    # https://dev.euipo.europa.eu/product/trademark-search_100/api/trademark-search
    SANDBOX_AUTH_URL = "https://auth-sandbox.euipo.europa.eu"
    PROD_AUTH_URL = "https://euipo.europa.eu/cas-server-webapp"

    SANDBOX_API_URL = "https://api-sandbox.euipo.europa.eu"
    PROD_API_URL = "https://api.euipo.europa.eu"

    # Token endpoint (OpenID Connect)
    TOKEN_PATH = "/oidc/accessToken"

    # Trademark search endpoint (base path for API)
    SEARCH_PATH = "/trademark-search/trademarks"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        sandbox: bool = True,
        timeout: int = 30,
    ):
        """Initialize EUIPO client.

        Args:
            client_id: OAuth client ID from EUIPO API Portal
            client_secret: OAuth client secret from EUIPO API Portal
            sandbox: Use sandbox environment (default True)
            timeout: Request timeout in seconds
        """
        self._client_id = client_id
        self._client_secret = client_secret
        self._sandbox = sandbox
        self._timeout = timeout
        self._session = requests.Session()
        self._token: EuipoToken | None = None

    @property
    def auth_url(self) -> str:
        """Get auth URL based on environment."""
        return self.SANDBOX_AUTH_URL if self._sandbox else self.PROD_AUTH_URL

    @property
    def api_url(self) -> str:
        """Get API URL based on environment."""
        return self.SANDBOX_API_URL if self._sandbox else self.PROD_API_URL

    @property
    def is_configured(self) -> bool:
        """Check if client has valid credentials configured."""
        return bool(self._client_id and self._client_secret)

    def _get_token(self) -> str:
        """Get valid access token, refreshing if necessary.

        Returns:
            Valid access token string

        Raises:
            EuipoApiError: If token acquisition fails
        """
        # Return cached token if still valid
        if self._token and not self._token.is_expired:
            return self._token.access_token

        # Request new token from auth server
        token_url = f"{self.auth_url}{self.TOKEN_PATH}"

        try:
            response = self._session.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "scope": "uid",
                },
                timeout=self._timeout,
            )
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            status = None
            if hasattr(e, "response") and e.response is not None:
                status = getattr(e.response, "status_code", None)
            raise EuipoApiError(
                f"Failed to obtain access token: {e}",
                status_code=status,
            ) from e

        data = response.json()
        access_token = data.get("access_token")
        expires_in = data.get("expires_in", 3600)

        if not access_token:
            raise EuipoApiError("No access token in response", response=data)

        self._token = EuipoToken(
            access_token=access_token,
            expires_at=time.time() + expires_in,
            token_type=data.get("token_type", "Bearer"),
        )

        return self._token.access_token

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        token = self._get_token()
        return {
            "Authorization": f"Bearer {token}",
            "X-IBM-Client-Id": self._client_id,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _build_rsql_query(
        self,
        name: str,
        nice_classes: list[int] | None = None,
        status_filter: list[str] | None = None,
    ) -> str:
        """Build RSQL query string for trademark search.

        EUIPO API uses RSQL (RESTful Service Query Language) for filtering.
        See: https://dev.euipo.europa.eu/product/trademark-search_100/api/trademark-search

        Args:
            name: Trademark name to search for (supports * wildcard)
            nice_classes: Filter by Nice classification classes
            status_filter: Filter by status codes

        Returns:
            RSQL query string
        """
        conditions = []

        # Search by verbal element (word mark) with wildcard
        # Use * as wildcard for partial matching
        escaped_name = name.replace("'", "\\'")
        conditions.append(f"wordMarkSpecification.verbalElement==*{escaped_name}*")

        # Filter by Nice classes using =in= operator
        if nice_classes:
            classes_str = ",".join(str(c) for c in nice_classes)
            conditions.append(f"niceClasses=in=({classes_str})")

        # Filter by status using =in= operator
        if status_filter:
            statuses_str = ",".join(status_filter)
            conditions.append(f"status=in=({statuses_str})")

        # Join conditions with AND (semicolon in RSQL)
        return ";".join(conditions)

    def search_trademarks(
        self,
        name: str,
        nice_classes: list[int] | None = None,
        status_filter: list[str] | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Search for trademarks matching the given criteria.

        Uses RSQL query syntax as per EUIPO API specification.
        See: https://dev.euipo.europa.eu/product/trademark-search_100/api/trademark-search

        Args:
            name: Trademark name to search for
            nice_classes: Filter by Nice classification classes
            status_filter: Filter by status (e.g., ["REGISTERED", "APPLICATION_PUBLISHED"])
            max_results: Maximum number of results to return (10-100)

        Returns:
            List of trademark data dictionaries

        Raises:
            EuipoApiError: If API request fails
        """
        if not self.is_configured:
            raise EuipoApiError("EUIPO client not configured - missing credentials")

        url = f"{self.api_url}{self.SEARCH_PATH}"

        # Build RSQL query
        rsql_query = self._build_rsql_query(name, nice_classes, status_filter)

        # API parameters per documentation
        params: dict[str, Any] = {
            "page": 0,
            "size": max(10, min(max_results, 100)),  # API requires 10-100
            "query": rsql_query,
        }

        try:
            response = self._session.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=self._timeout,
            )
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            status = None
            resp_text = None
            if hasattr(e, "response") and e.response is not None:
                status = getattr(e.response, "status_code", None)
                try:
                    resp_text = e.response.text[:500]
                except Exception:
                    pass
            raise EuipoApiError(
                f"Trademark search failed: {e}" + (f" Response: {resp_text}" if resp_text else ""),
                status_code=status,
            ) from e

        data = response.json()

        # API returns { trademarks: [...], size, totalElements, totalPages, page }
        return data.get("trademarks", [])

    def get_trademark_detail(self, application_number: str) -> dict[str, Any] | None:
        """Get detailed information about a specific trademark.

        Args:
            application_number: The trademark application number

        Returns:
            Trademark detail dictionary or None if not found

        Raises:
            EuipoApiError: If API request fails
        """
        if not self.is_configured:
            raise EuipoApiError("EUIPO client not configured - missing credentials")

        url = f"{self.api_url}{self.SEARCH_PATH}/{application_number}"

        try:
            response = self._session.get(
                url,
                headers=self._get_headers(),
                timeout=self._timeout,
            )

            if response.status_code == 404:
                return None

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            status = None
            if hasattr(e, "response") and e.response is not None:
                status = getattr(e.response, "status_code", None)
            raise EuipoApiError(
                f"Failed to get trademark detail: {e}",
                status_code=status,
            ) from e

    def parse_trademark_result(self, data: dict[str, Any]) -> TrademarkConflict:
        """Parse API response data into TrademarkConflict object.

        Based on actual EUIPO API response structure from:
        https://dev.euipo.europa.eu/product/trademark-search_100/api/trademark-search

        Args:
            data: Raw API response dictionary

        Returns:
            TrademarkConflict object
        """
        # Extract verbal element from wordMarkSpecification
        word_spec = data.get("wordMarkSpecification", {})
        name = word_spec.get("verbalElement", "") if isinstance(word_spec, dict) else ""

        # Application number is the primary identifier
        app_no = data.get("applicationNumber", "")

        # Status field
        status = data.get("status", "UNKNOWN")

        # Extract owner from applicants list
        owner = None
        applicants = data.get("applicants", [])
        if applicants and isinstance(applicants, list):
            first_applicant = applicants[0]
            if isinstance(first_applicant, dict):
                # Applicant info may have name or identifier
                owner = first_applicant.get("name") or first_applicant.get("identifier")

        # Nice classes - API returns list of integers directly
        nice_classes: list[int] = []
        raw_classes = data.get("niceClasses", [])
        for c in raw_classes:
            if isinstance(c, int):
                nice_classes.append(c)
            elif isinstance(c, str) and c.isdigit():
                nice_classes.append(int(c))

        # Dates - ISO format strings (YYYY-MM-DD)
        filing_date = None
        reg_date = None

        filing_str = data.get("applicationDate")
        if filing_str:
            try:
                filing_date = datetime.fromisoformat(filing_str)
            except (ValueError, AttributeError):
                pass

        reg_str = data.get("registrationDate")
        if reg_str:
            try:
                reg_date = datetime.fromisoformat(reg_str)
            except (ValueError, AttributeError):
                pass

        return TrademarkConflict(
            name=name,
            application_number=str(app_no),
            status=status.upper() if status else "UNKNOWN",
            nice_classes=nice_classes,
            owner=owner,
            similarity_score=0.0,  # Will be calculated separately
            filing_date=filing_date,
            registration_date=reg_date,
        )

    def test_connection(self) -> tuple[bool, str]:
        """Test API connection and authentication.

        Returns:
            Tuple of (success, message)
        """
        if not self.is_configured:
            return False, "Client not configured - missing credentials"

        try:
            # Try to get a token
            self._get_token()

            # Try a simple search
            self.search_trademarks("TEST", max_results=1)

            return True, "Connection successful"

        except EuipoApiError as e:
            return False, f"API error: {e.message}"
        except Exception as e:
            return False, f"Connection failed: {e!s}"


class MockEuipoClient:
    """Mock EUIPO client for testing and demo purposes.

    Returns simulated results without making actual API calls.
    Useful for development and when API credentials are not available.
    """

    def __init__(self) -> None:
        """Initialize mock client."""
        # Simulated trademark database
        self._mock_trademarks = [
            {
                "name": "NEXUS",
                "applicationNumber": "018000001",
                "status": "REGISTERED",
                "niceClasses": [9, 35, 42],
                "owner": "Example Corp",
            },
            {
                "name": "NEXTRA",
                "applicationNumber": "018000002",
                "status": "REGISTERED",
                "niceClasses": [9, 38, 42],
                "owner": "Tech Holdings Ltd",
            },
            {
                "name": "ZONIFY",
                "applicationNumber": "018000003",
                "status": "APPLIED",
                "niceClasses": [9, 35],
                "owner": "Digital Solutions Inc",
            },
            {
                "name": "VALUJO",
                "applicationNumber": "018000004",
                "status": "REGISTERED",
                "niceClasses": [35, 36, 42],
                "owner": "Finance Global SA",
            },
            {
                "name": "RIVENO",
                "applicationNumber": "018000005",
                "status": "APPLIED",
                "niceClasses": [9, 16, 42],
                "owner": "Creative Labs",
            },
        ]

    @property
    def is_configured(self) -> bool:
        """Mock client is always configured."""
        return True

    def search_trademarks(
        self,
        name: str,
        nice_classes: list[int] | None = None,
        status_filter: list[str] | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Search mock trademark database.

        Returns trademarks that have similar names (simple substring match).
        """
        from src.trademark.similarity import string_similarity

        results = []
        name_lower = name.lower()

        for tm in self._mock_trademarks:
            tm_name = tm["name"].lower()

            # Check name similarity
            sim = string_similarity(name, tm_name)
            if sim < 0.3 and name_lower not in tm_name and tm_name not in name_lower:
                continue

            # Check class filter
            if nice_classes:
                overlap = set(tm.get("niceClasses", [])) & set(nice_classes)
                if not overlap:
                    continue

            # Check status filter
            if status_filter:
                if tm.get("status", "").upper() not in [s.upper() for s in status_filter]:
                    continue

            results.append(tm)

            if len(results) >= max_results:
                break

        return results

    def parse_trademark_result(self, data: dict[str, Any]) -> TrademarkConflict:
        """Parse mock data into TrademarkConflict."""
        return TrademarkConflict(
            name=data.get("name", ""),
            application_number=data.get("applicationNumber", ""),
            status=data.get("status", "UNKNOWN"),
            nice_classes=data.get("niceClasses", []),
            owner=data.get("owner"),
            similarity_score=0.0,
        )

    def test_connection(self) -> tuple[bool, str]:
        """Mock connection test always succeeds."""
        return True, "Mock client - no real connection"
