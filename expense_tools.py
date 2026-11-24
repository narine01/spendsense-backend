"""Calculations the model can request instead of doing arithmetic itself.

Everything here is plain Python over the expenses the app sent with the question,
so it is deterministic and unit-testable. The model only chooses WHICH calculation
to run and with which filters; the numbers come from this code.
"""

import datetime as dt
from collections import defaultdict
from typing import Callable, Iterable


def _parse_day(value: str) -> dt.date | None:
    value = (value or "").strip()
    if not value:
        return None
    return dt.date.fromisoformat(value)  # raises ValueError on a bad format


def filter_expenses(
    expenses: Iterable,
    category: str = "",
    start_date: str = "",
    end_date: str = "",
    text: str = "",
) -> list:
    """Keep expenses matching every non-empty filter. Dates are inclusive."""
    start = _parse_day(start_date)
    end = _parse_day(end_date)
    wanted_category = (category or "").strip().lower()
    needle = (text or "").strip().lower()

    result = []
    for e in expenses:
        if start and e.date < start:
            continue
        if end and e.date > end:
            continue
        if wanted_category and e.category.strip().lower() != wanted_category:
            continue
        if needle and needle not in e.title.lower() and needle not in e.category.lower():
            continue
        result.append(e)
    return result


def _money(value: float) -> float:
    return round(float(value), 2)


def make_tools(expenses: list) -> list[Callable]:
    """Build the tool functions for one request. The model sees each function's
    name, signature and docstring, so keep those clear and simple."""

    def total_spent(category: str = "", start_date: str = "", end_date: str = "") -> dict:
        """Total amount spent in rupees and the number of expenses.

        Args:
            category: Only count this category (exact name, case-insensitive). Empty means all categories.
            start_date: First day to include, as YYYY-MM-DD. Empty means no lower limit.
            end_date: Last day to include, as YYYY-MM-DD. Empty means no upper limit.
        """
        try:
            rows = filter_expenses(expenses, category, start_date, end_date)
        except ValueError:
            return {"error": "Dates must be in YYYY-MM-DD format."}
        return {"total_rupees": _money(sum(e.amount for e in rows)), "count": len(rows)}

    def spending_by_category(start_date: str = "", end_date: str = "") -> dict:
        """Total spent per category in rupees, largest first.

        Args:
            start_date: First day to include, as YYYY-MM-DD. Empty means no lower limit.
            end_date: Last day to include, as YYYY-MM-DD. Empty means no upper limit.
        """
        try:
            rows = filter_expenses(expenses, "", start_date, end_date)
        except ValueError:
            return {"error": "Dates must be in YYYY-MM-DD format."}
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for e in rows:
            totals[e.category] += e.amount
            counts[e.category] += 1
        ordered = sorted(totals.items(), key=lambda item: item[1], reverse=True)
        return {
            "categories": [
                {"category": name, "total_rupees": _money(total), "count": counts[name]}
                for name, total in ordered
            ]
        }

    def top_expenses(
        limit: int = 3, category: str = "", start_date: str = "", end_date: str = ""
    ) -> dict:
        """The largest individual expenses, biggest first.

        Args:
            limit: How many to return, between 1 and 10.
            category: Only this category. Empty means all categories.
            start_date: First day to include, as YYYY-MM-DD. Empty means no lower limit.
            end_date: Last day to include, as YYYY-MM-DD. Empty means no upper limit.
        """
        try:
            rows = filter_expenses(expenses, category, start_date, end_date)
        except ValueError:
            return {"error": "Dates must be in YYYY-MM-DD format."}
        limit = max(1, min(int(limit), 10))
        rows = sorted(rows, key=lambda e: e.amount, reverse=True)[:limit]
        return {
            "expenses": [
                {
                    "title": e.title,
                    "amount_rupees": _money(e.amount),
                    "category": e.category,
                    "date": e.date.isoformat(),
                }
                for e in rows
            ]
        }

    def find_expenses(text: str, limit: int = 10, start_date: str = "", end_date: str = "") -> dict:
        """Find expenses whose title or category contains some text (for example a shop name),
        with the total of ALL matches.

        Args:
            text: Text to look for, case-insensitive.
            limit: Maximum number of rows to return, between 1 and 20 (the total still covers all matches).
            start_date: First day to include, as YYYY-MM-DD. Empty means no lower limit.
            end_date: Last day to include, as YYYY-MM-DD. Empty means no upper limit.
        """
        try:
            rows = filter_expenses(expenses, "", start_date, end_date, text)
        except ValueError:
            return {"error": "Dates must be in YYYY-MM-DD format."}
        limit = max(1, min(int(limit), 20))
        shown = sorted(rows, key=lambda e: e.date, reverse=True)[:limit]
        return {
            "match_count": len(rows),
            "total_rupees": _money(sum(e.amount for e in rows)),
            "expenses": [
                {
                    "title": e.title,
                    "amount_rupees": _money(e.amount),
                    "category": e.category,
                    "date": e.date.isoformat(),
                }
                for e in shown
            ],
        }

    return [total_spent, spending_by_category, top_expenses, find_expenses]
