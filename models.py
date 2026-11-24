import datetime as dt

from pydantic import BaseModel


class ParseRequest(BaseModel):
    text: str


class ParsedExpense(BaseModel):
    title: str
    amount: float
    category: str


class ScannedBill(BaseModel):
    merchant: str
    amount: float
    date: str  # YYYY-MM-DD, or empty string if not visible
    category: str


class ExpenseItem(BaseModel):
    title: str
    amount: float
    category: str
    date: dt.date  # the app sends YYYY-MM-DD; anything else is rejected with 422


class AskRequest(BaseModel):
    question: str
    today: dt.date
    expenses: list[ExpenseItem]


class AskResponse(BaseModel):
    answer: str
