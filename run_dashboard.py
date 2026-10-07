"""
run_dashboard.py
Launches the modern Enterprise SQL Assistant Web Dashboard & FastAPI backend.
"""

import sys
import webbrowser
import uvicorn
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    print("=" * 65)
    print("  STARTING ENTERPRISE SQL ASSISTANT MODERN DASHBOARD")
    print("  Server:    http://127.0.0.1:8000")
    print("  API Docs:  http://127.0.0.1:8000/docs")
    print("  Engine:    FastAPI + In-Memory SQLite Sandbox + QLoRA/DPO")
    print("=" * 65)

    # Launch browser after slight delay
    webbrowser.open("http://127.0.0.1:8000")

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
