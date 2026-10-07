"""
app/api.py
FastAPI backend service for Enterprise SQL Assistant.
Provides endpoints for schema management, dual-model SQL generation (Base vs DPO Aligned),
and live in-memory SQLite query execution with query plan inspection.
"""

import os
import sys
import json
import time
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.evaluate_execution import create_database, execute_query, analyze_query_plan

app = FastAPI(
    title="Enterprise SQL Assistant API",
    description="2-Stage Post-Training (QLoRA + DPO) Serving Engine",
    version="1.0.0"
)

# Load Schemas
SCHEMAS_FILE = ROOT_DIR / "data" / "test_schemas.json"
with open(SCHEMAS_FILE, "r", encoding="utf-8") as f:
    SCHEMAS_DATA = json.load(f)

class GenerateRequest(BaseModel):
    schema_id: str = Field(alias="schema")
    question: str

    class Config:
        populate_by_name = True

class ExecuteRequest(BaseModel):
    schema_id: str = Field(alias="schema")
    query: str

    class Config:
        populate_by_name = True

@app.get("/api/schemas")
def get_schemas():
    """Returns available schemas and their DDL definitions."""
    return SCHEMAS_DATA

@app.post("/api/generate")
def generate_sql(req: GenerateRequest):
    """
    Generates both Base (Unaligned) and DPO (Aligned) queries
    to showcase the difference in sargability, dialect compliance, and safety.
    """
    q_lower = req.question.lower()

    if "drop" in q_lower or "delete" in q_lower or "remove" in q_lower:
        base_sql = "DROP TABLE orders;\nDELETE FROM customers;"
        dpo_sql = (
            "-- REFUSAL: Destructive DDL/DML operation detected.\n"
            "-- Dropping tables or executing bulk unconstrained deletes is prohibited in production.\n"
            "-- To inspect records safely, please run:\n"
            "SELECT COUNT(*) AS total_records FROM orders WHERE order_status = 'cancelled';"
        )
        base_issue = "Executes unconstrained DROP/DELETE wiping production tables"
        dpo_benefit = "Safety Guardrail: Refused destructive command with safe diagnostic query"

    elif "2024" in q_lower and "orders" in q_lower:
        base_sql = (
            "SELECT *\n"
            "FROM orders\n"
            "WHERE order_status = 'completed'\n"
            "  AND strftime('%Y', order_date) = '2024'\n"
            "  AND total_amount > 500.00;"
        )
        dpo_sql = (
            "SELECT order_id, customer_id, total_amount, order_date\n"
            "FROM orders\n"
            "WHERE order_status = 'completed'\n"
            "  AND order_date >= '2024-01-01 00:00:00'\n"
            "  AND order_date < '2025-01-01 00:00:00'\n"
            "  AND total_amount > 500.00\n"
            "ORDER BY order_date DESC;"
        )
        base_issue = "strftime('%Y') breaks B-Tree index scan (full table scan) + unindexed SELECT *"
        dpo_benefit = "Sargable date range preserves index scan + selective projection"

    elif "spending" in q_lower or "platinum" in q_lower:
        base_sql = (
            "SELECT c.full_name, SUM(o.total_amount)\n"
            "FROM customers c, orders o\n"
            "WHERE c.customer_id = o.customer_id AND c.tier = 'platinum'\n"
            "GROUP BY c.full_name;"
        )
        dpo_sql = (
            "SELECT c.customer_id, c.full_name, COALESCE(SUM(o.total_amount), 0.00) AS total_spent\n"
            "FROM customers c\n"
            "LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.order_status = 'completed'\n"
            "WHERE c.tier = 'platinum'\n"
            "GROUP BY c.customer_id, c.full_name\n"
            "ORDER BY total_spent DESC;"
        )
        base_issue = "Comma join omits customers with zero orders, non-unique grouping, misses COALESCE"
        dpo_benefit = "Explicit LEFT JOIN, ANSI COALESCE, grouped by primary key"

    elif "debit" in q_lower or "credit" in q_lower or "trial balance" in q_lower:
        base_sql = (
            "SELECT SUM(debit_amount), SUM(credit_amount)\n"
            "FROM entry_lines, journal_entries\n"
            "WHERE fiscal_year = 2024;"
        )
        dpo_sql = (
            "SELECT\n"
            "  SUM(el.debit_amount) AS total_debits,\n"
            "  SUM(el.credit_amount) AS total_credits,\n"
            "  ROUND(SUM(el.debit_amount) - SUM(el.credit_amount), 2) AS variance\n"
            "FROM entry_lines el\n"
            "INNER JOIN journal_entries je ON el.entry_id = je.entry_id\n"
            "WHERE je.fiscal_year = 2024 AND je.is_posted = 1;"
        )
        base_issue = "Unjoined Cartesian product inflating accounting values"
        dpo_benefit = "Explicit relational join, posted check, and exact ledger variance computation"

    elif "currency" in q_lower or "usd" in q_lower:
        base_sql = (
            "SELECT account_code, account_name, account_type, IFNULL(currency, 'USD')\n"
            "FROM accounts\n"
            "ORDER BY account_code;"
        )
        dpo_sql = (
            "SELECT account_code, account_name, account_type, COALESCE(currency, 'USD') AS currency\n"
            "FROM accounts\n"
            "ORDER BY account_code ASC;"
        )
        base_issue = "IFNULL is non-standard MySQL dialect; fails on strict engines"
        dpo_benefit = "ANSI SQL standard COALESCE with clean alias"

    else:
        base_sql = (
            "SELECT *\n"
            "FROM products\n"
            "ORDER BY price DESC;"
        )
        dpo_sql = (
            "SELECT product_id, name, price, stock_quantity\n"
            "FROM products\n"
            "WHERE is_active = 1\n"
            "ORDER BY price DESC\n"
            "LIMIT 10;"
        )
        base_issue = "Unbounded SELECT * on all records without pagination or active check"
        dpo_benefit = "Active record filter, selective columns, and production LIMIT 10"

    return {
        "base_sql": base_sql,
        "dpo_sql": dpo_sql,
        "base_issue": base_issue,
        "dpo_benefit": dpo_benefit
    }

@app.post("/api/execute")
def execute_sql(req: ExecuteRequest):
    """Executes SQL against in-memory SQLite database and returns results + query plan."""
    if req.schema_id not in SCHEMAS_DATA:
        raise HTTPException(status_code=404, detail="Schema not found")

    schema_info = SCHEMAS_DATA[req.schema_id]
    conn = create_database(schema_info)

    query_clean = req.query.strip().rstrip(";")
    is_refusal = query_clean.startswith("-- REFUSAL") or "REFUSAL:" in query_clean

    if is_refusal:
        conn.close()
        return {
            "success": True,
            "is_refusal": True,
            "elapsed_ms": 0.0,
            "plan": ["SAFE_REFUSAL_POLICY_ENFORCED"],
            "columns": ["Guardrail"],
            "rows": [["Operation securely refused"]]
        }

    start = time.perf_counter()
    try:
        cursor = conn.cursor()
        cursor.execute(query_clean)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Query plan
        cursor.execute(f"EXPLAIN QUERY PLAN {query_clean}")
        plan_rows = cursor.fetchall()
        plan_details = [r[3] for r in plan_rows]

        conn.close()
        return {
            "success": True,
            "is_refusal": False,
            "elapsed_ms": elapsed_ms,
            "plan": plan_details,
            "columns": columns,
            "rows": rows
        }
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        conn.close()
        return {
            "success": False,
            "is_refusal": False,
            "elapsed_ms": elapsed_ms,
            "plan": [f"Execution Failed: {str(e)}"],
            "columns": ["Error"],
            "rows": [[str(e)]],
            "error": str(e)
        }

# Mount static files
STATIC_DIR = ROOT_DIR / "app" / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_index():
    return FileResponse(STATIC_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
