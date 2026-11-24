import datetime as dt
from types import SimpleNamespace as NS

import pytest
from google.genai import errors, types

from ask_service import AskError, month_ranges, run_ask
from models import AskRequest, ExpenseItem


def req():
    return AskRequest(
        question="how much on food?",
        today=dt.date(2026, 11, 15),
        expenses=[ExpenseItem(title="Zomato", amount=300, category="Food", date=dt.date(2026, 11, 2))],
    )


def chunk(*parts):
    return NS(candidates=[NS(content=NS(parts=list(parts)))])


def text(t):
    return NS(text=t, thought=False, function_call=None)


def call(name, **args):
    return NS(text=None, thought=False, function_call=NS(name=name, args=args, id="c1"))


class FakeModels:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.seen = []

    def generate_content_stream(self, model, contents, config):
        self.seen.append(list(contents))
        item = self.rounds.pop(0)
        if isinstance(item, Exception):
            raise item
        return iter(item)


def fake(rounds):
    return SimpleNamespace_client(FakeModels(rounds))


def SimpleNamespace_client(models):
    return NS(models=models)


def test_month_ranges_january_rolls_back_year():
    this, last = month_ranges(dt.date(2026, 1, 10))
    assert this == (dt.date(2026, 1, 1), dt.date(2026, 1, 10))
    assert last == (dt.date(2025, 12, 1), dt.date(2025, 12, 31))


def test_tool_loop_runs_code_and_streams_answer():
    c = fake([[chunk(call("total_spent", category="Food"))], [chunk(text("You spent ")), chunk(text("₹300."))]])
    out = "".join(run_ask(c, "m", req()))
    assert out == "You spent ₹300."
    # second round must contain the real computed result
    resp = c.models.seen[1][-1].parts[0].function_response
    assert resp.response == {"result": {"total_rupees": 300.0, "count": 1}}


def test_unknown_tool_and_bad_args_do_not_crash():
    c = fake([[chunk(call("nope"))], [chunk(call("total_spent", bogus=1))], [chunk(text("ok"))]])
    assert "".join(run_ask(c, "m", req())) == "ok"


def test_gives_up_after_max_rounds():
    rounds = [[chunk(call("total_spent"))] for _ in range(4)]
    out = "".join(run_ask(fake(rounds), "m", req()))
    assert "rephrasing" in out


def test_quota_error_maps_to_429():
    err = errors.APIError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    with pytest.raises(AskError) as e:
        list(run_ask(fake([err]), "m", req()))
    assert e.value.status_code == 429
