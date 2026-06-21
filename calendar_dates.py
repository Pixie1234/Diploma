# ============================================================
# src/calendar_utils.py
# US Market Trading Day Calendar Utilities
# ============================================================
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

US_MARKET_CALENDAR = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def get_next_trading_days(start_date, n_days):
    """Generate exactly n_days real US market trading days after start_date."""
    return pd.date_range(
        start=start_date + US_MARKET_CALENDAR,
        periods=n_days,
        freq=US_MARKET_CALENDAR
    )


def get_last_trading_day(date=None):
    """Return the most recent trading day on or before the given date."""
    if date is None:
        date = pd.Timestamp.today()
    candidate = pd.Timestamp(date).normalize()
    holidays = USFederalHolidayCalendar().holidays()
    while candidate.weekday() >= 5 or candidate in holidays:
        candidate -= pd.Timedelta(days=1)
    return candidate


def assign_to_trading_day(publish_date):
    """
    Roll any weekend/holiday news date forward to next real trading day.
    Saturday -> Monday, Sunday -> Monday, Holiday -> Next trading day.
    """
    if publish_date is None:
        return None
    d = pd.Timestamp(publish_date).normalize()
    holidays = USFederalHolidayCalendar().holidays(
        start=d - pd.Timedelta(days=1),
        end=d + pd.Timedelta(days=7)
    )
    while d.weekday() >= 5 or d in holidays:
        d += pd.Timedelta(days=1)
    return d


def is_trading_day(date):
    """Check if a given date is a valid US market trading day."""
    d = pd.Timestamp(date).normalize()
    if d.weekday() >= 5:
        return False
    holidays = USFederalHolidayCalendar().holidays(start=d, end=d)
    return d not in holidays