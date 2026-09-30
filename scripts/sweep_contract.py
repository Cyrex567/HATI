"""Shared, fail-closed provenance rules for the Athena counterfactual."""
from datetime import datetime, timezone

# Conservative: exclude the entire landing day, not an assumed touchdown time.
DEFAULT_BEFORE = "2025-03-06T00:00:00Z"
# Held-out post-landing validation (audit item 2): frames that show the landed
# spacecraft. It starts after the landing day, so no frame can straddle the
# touchdown, and such frames never enter the pre-landing detector.
POST_LANDING_AFTER = "2025-03-07T00:00:00Z"
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


def post_landing_window(after: str, before: str) -> None:
    """Refuse a post-landing window that could reach back into the counterfactual."""
    if utc_time(after) < utc_time(DEFAULT_BEFORE):
        raise ValueError(f"post-landing window must start at or after {DEFAULT_BEFORE}")
    if utc_time(before) <= utc_time(after):
        raise ValueError("post-landing window must end after it starts")


def within(value: str, after: str, before: str) -> bool:
    """after <= acquisition < before; missing or naive timestamps fail."""
    start, end = utc_time(after), utc_time(before)
    try:
        return start <= utc_time(value) < end
    except (ValueError, TypeError, AttributeError):
        return False
