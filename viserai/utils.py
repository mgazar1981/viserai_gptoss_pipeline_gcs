from __future__ import annotations
from datetime import date, timedelta
import pandas_market_calendars as mcal

NYSE = mcal.get_calendar("NYSE")

def previous_trading_day(d: date) -> date:
    start = d - timedelta(days=10)
    sched = NYSE.schedule(start_date=start, end_date=d)
    days = [x for x in sched.index.date if x < d]
    return days[-1] if days else (d - timedelta(days=1))

def next_trading_day(d: date) -> date:
    end = d + timedelta(days=10)
    sched = NYSE.schedule(start_date=d, end_date=end)
    days = [x for x in sched.index.date if x > d]
    return days[0] if days else (d + timedelta(days=1))

def is_trading_day(d: date) -> bool:
    sched = NYSE.schedule(start_date=d, end_date=d)
    return len(sched) > 0
