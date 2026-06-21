from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


def normalize_article_text(s: str, max_len: int) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ")
    s = " ".join(str(s).split())
    # Fix cases like: "m i l l i o n" → "million"
    s = re.sub(r"\b([A-Za-z])\s+(?=[A-Za-z]\b)", r"\1", s)
    return s[:max_len]


def company_tokens_from_name(name: str) -> list[str]:
    """Extract alphanumeric tokens so "Amazon.com" matches "Amazon" headlines."""
    u = str(name).upper()
    tokens = re.findall(r"[A-Z0-9]{2,}", u)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def finlight_pub_dt(art: Any) -> datetime:
    pub_date = (
        getattr(art, "published_at", None)
        or getattr(art, "publishDate", None)
        or getattr(art, "publishedAt", None)
        or getattr(art, "date", None)
        or getattr(art, "created_at", None)
        or getattr(art, "createdAt", None)
        or getattr(art, "published_date", None)
        or getattr(art, "publish_date", None)
    )

    if isinstance(pub_date, str):
        try:
            dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)
    elif isinstance(pub_date, datetime):
        dt = pub_date
    else:
        return datetime.min.replace(tzinfo=timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_finlight_relevant(art: Any, *, symbol_upper: str, company_tokens: list[str]) -> bool:
    title = (getattr(art, "title", "") or "").upper()
    summary_raw = getattr(art, "summary", None) or getattr(art, "content", None) or ""
    summary = summary_raw[:500].upper()

    for attr_name in ("tickers", "symbols", "ticker", "symbol"):
        v = getattr(art, attr_name, None)
        if isinstance(v, str) and symbol_upper in v.upper():
            return True
        if isinstance(v, (list, tuple)) and any(symbol_upper in str(x).upper() for x in v):
            return True

    title_has_our_symbol = symbol_upper in title
    title_has_company_token = any(tok in title for tok in company_tokens)
    if title_has_our_symbol or title_has_company_token:
        return True

    if symbol_upper in summary and not title_has_company_token:
        return False
    return False


def alpha_pub_dt(art: Any) -> datetime:
    t = (art.get("time_published", "") or "").strip()
    if not t:
        return datetime.min.replace(tzinfo=timezone.utc)

    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%d%H%M%S"):
        try:
            dt = datetime.strptime(t, fmt)
            if ZoneInfo is not None:
                dt = dt.replace(tzinfo=ZoneInfo("America/New_York"))
                return dt.astimezone(timezone.utc)
            # Fallback: treat as UTC if ZoneInfo isn't available.
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return datetime.min.replace(tzinfo=timezone.utc)


def is_alpha_vantage_relevant(art: Any, *, symbol_upper: str, company_tokens: list[str]) -> bool:
    title = (art.get("title", "") or "").upper()
    if symbol_upper in title:
        return True
    return any(tok in title for tok in company_tokens)
