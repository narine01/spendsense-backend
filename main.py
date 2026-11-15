import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException

load_dotenv()
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(api_version="v1beta"),
)
MODEL = "gemini-2.5-flash"
MAX_FILE_BYTES = 5_000_000
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf"}
app = FastAPI()


class ParseRequest(BaseModel):
    text: str


class ParsedExpense(BaseModel):
    title: str
    amount: float
    category: str


@app.post("/parse-expense", response_model=ParsedExpense)
def parse_expense(req: ParseRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=f"Extract the expense from this text: {req.text}",
            config={
                "response_mime_type": "application/json",
                "response_schema": ParsedExpense,
            },
        )
    except Exception:
        raise HTTPException(status_code=502, detail="AI service failed")
    return response.parsed

class ScannedBill(BaseModel):
    merchant: str
    amount: float
    date: str  # YYYY-MM-DD, or empty if not visible
    category: str


@app.post("/scan-bill", response_model=ScannedBill)
def scan_bill(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    data = file.file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_bytes(data=data, mime_type=file.content_type),
                "Read this bill. Return the merchant name, the total amount in rupees, "
                "the date as YYYY-MM-DD (empty string if not visible), and one category "
                "from: Food, Transport, Groceries, Bills, Entertainment, Shopping, Health, Other.",
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": ScannedBill,
            },
        )
    except Exception:
        raise HTTPException(status_code=502, detail="AI service failed")
    return response.parsed