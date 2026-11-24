import os
import sys
from pathlib import Path

# Must be set before main.py is imported (it reads them at import time).
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("APP_TOKEN", "test-token")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
