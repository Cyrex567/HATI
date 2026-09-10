"""Shared, fail-closed provenance rules for the Athena counterfactual."""
from datetime import datetime, timezone

# Conservative: exclude the entire landing day, not an assumed touchdown time.
DEFAULT_BEFORE = "2025-03-06T00:00:00Z"
PROCESSING_VERSION = "2026-09-audit-echo-v1"


def utc_time(value: str) -> datetime:
    t = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError("acquisition/cutoff timestamps must include a timezone")
    return t.astimezone(timezone.utc)


def predates(value: str, before: str = DEFAULT_BEFORE) -> bool:
    cutoff = utc_time(before)
    try:
        return utc_time(value) < cutoff
    except (ValueError, TypeError, AttributeError):
        return False
