import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException

load_dotenv()
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(api_version="v1beta"),
)
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
            model="gemini-2.5-flash",
            contents=f"Extract the expense from this text: {req.text}",
            config={
                "response_mime_type": "application/json",
                "response_schema": ParsedExpense,
            },
        )
    except Exception:
        raise HTTPException(status_code=502, detail="AI service failed")
    return response.parsed