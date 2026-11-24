import datetime as dt

from expense_tools import make_tools
from models import ExpenseItem


def D(s):
    return dt.date.fromisoformat(s)


EXPENSES = [
    ExpenseItem(title="Swiggy dinner", amount=450.5, category="Food", date=D("2026-11-02")),
    ExpenseItem(title="Uber", amount=220, category="Transport", date=D("2026-11-05")),
    ExpenseItem(title="Zomato", amount=300, category="Food", date=D("2026-10-28")),
    ExpenseItem(title="Rent", amount=15000, category="Bills", date=D("2026-10-01")),
]


def tools():
    return {fn.__name__: fn for fn in make_tools(EXPENSES)}


def test_total_all():
    assert tools()["total_spent"]() == {"total_rupees": 15970.5, "count": 4}


def test_total_category_case_insensitive_and_dates():
    r = tools()["total_spent"](category="food", start_date="2026-11-01", end_date="2026-11-30")
    assert r == {"total_rupees": 450.5, "count": 1}


def test_total_bad_date():
    assert "error" in tools()["total_spent"](start_date="yesterday")


def test_by_category_sorted():
    r = tools()["spending_by_category"]()
    assert [c["category"] for c in r["categories"]] == ["Bills", "Food", "Transport"]
    assert r["categories"][1]["total_rupees"] == 750.5


def test_top_expenses_limit_clamped():
    r = tools()["top_expenses"](limit=2)
    assert [e["title"] for e in r["expenses"]] == ["Rent", "Swiggy dinner"]
    assert len(tools()["top_expenses"](limit=999)["expenses"]) == 4
    assert len(tools()["top_expenses"](limit=0)["expenses"]) == 1


def test_find_expenses_totals_all_matches():
    r = tools()["find_expenses"](text="zomato")
    assert r["match_count"] == 1 and r["total_rupees"] == 300
    assert r["expenses"][0]["date"] == "2026-10-28"
