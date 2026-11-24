import json
import logging
import os
import secrets
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import errors, types

from ask_service import FALLBACK_ANSWER, AskError, run_ask
from models import (
    AskRequest,
    AskResponse,
    ParsedExpense,
    ParseRequest,
    ScannedBill,
)

load_dotenv()
log = logging.getLogger("uvicorn.error")

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(api_version="v1beta"),
)
APP_TOKEN = os.environ["APP_TOKEN"]  # must be set locally (.env) and on Render

# Gemini 2.5 models are "legacy" now (Google recommends newer models for new projects).
# Set GEMINI_MODEL on Render to switch without a code change, and check your limits at
# https://aistudio.google.com/rate-limit
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
SCAN_MODEL = os.environ.get("GEMINI_SCAN_MODEL", MODEL)

MAX_FILE_BYTES = 10_000_000
MAX_EXPENSES = 300
MAX_QUESTION_CHARS = 500
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf"}

RATE_LIMIT = 20  # requests
RATE_WINDOW = 60  # seconds
_hits: dict[str, deque] = defaultdict(deque)


def protect(request: Request, x_app_token: str = Header(default="")):
    # 1) token check (constant-time compare)
    if not secrets.compare_digest(x_app_token, APP_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 2) simple in-memory rate limit per client IP.
    # Render puts the real client IP in X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    window = _hits[ip]
    while window and now - window[0] > RATE_WINDOW:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests, slow down")
    window.append(now)


def call_gemini(**kwargs):
    """One non-streaming Gemini call. Retries briefly on 500/503, never on 429
    (a quota error will not clear in a few seconds), and logs the real cause."""
    last = None
    for attempt in range(3):
        try:
            return client.models.generate_content(**kwargs)
        except errors.APIError as e:
            last = e
            log.warning("Gemini API error %s: %s", e.code, e.message)
            if e.code in (500, 503) and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            break
        except Exception:
            log.exception("Unexpected error calling Gemini")
            raise HTTPException(status_code=502, detail="AI service failed")

    if last is not None and last.code == 429:
        raise HTTPException(status_code=429, detail="AI is busy right now, try again in a minute")
    raise HTTPException(status_code=502, detail="AI service failed")


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}  # open on purpose: no AI call, no data


@app.post("/parse-expense", response_model=ParsedExpense, dependencies=[Depends(protect)])
def parse_expense(req: ParseRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    response = call_gemini(
        model=MODEL,
        contents=f"Extract the expense from this text: {req.text}",
        config={
            "response_mime_type": "application/json",
            "response_schema": ParsedExpense,
        },
    )
    return response.parsed


@app.post("/scan-bill", response_model=ScannedBill, dependencies=[Depends(protect)])
def scan_bill(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    data = file.file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    response = call_gemini(
        model=SCAN_MODEL,
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
    return response.parsed


def _check_question(req: AskRequest) -> None:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty")
    if len(req.question) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=413, detail="Question too long")
    req.expenses = req.expenses[:MAX_EXPENSES]


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(protect)])
def ask(req: AskRequest):
    """Non-streaming version: waits for the whole answer."""
    _check_question(req)
    try:
        text = "".join(run_ask(client, MODEL, req)).strip()
    except AskError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception:
        log.exception("Unexpected error in /ask")
        raise HTTPException(status_code=502, detail="AI service failed")
    return AskResponse(answer=text or FALLBACK_ANSWER)


def _line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


@app.post("/ask/stream", dependencies=[Depends(protect)])
def ask_stream(req: AskRequest):
    """Streams the answer as newline-delimited JSON:
    {"delta": "..."} pieces, then {"done": true}, or {"error": "..."} if it fails midway."""
    _check_question(req)
    pieces = run_ask(client, MODEL, req)

    # Pull the first piece BEFORE starting the response, so a quota or API failure
    # still becomes a proper HTTP status (429 / 502) instead of a 200 with an error line.
    try:
        first = next(pieces, "")
    except AskError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception:
        log.exception("Unexpected error starting /ask/stream")
        raise HTTPException(status_code=502, detail="AI service failed")

    def body():
        try:
            if first:
                yield _line({"delta": first})
            for piece in pieces:
                yield _line({"delta": piece})
            yield _line({"done": True})
        except AskError as e:
            yield _line({"error": e.detail})
        except Exception:
            log.exception("Error while streaming /ask/stream")
            yield _line({"error": "AI service failed"})

    return StreamingResponse(
        body(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
