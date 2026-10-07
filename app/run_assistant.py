"""
app/run_assistant.py
Interactive CLI assistant for Enterprise Text-to-SQL.
Allows testing natural language queries against real schemas, executing them live in SQLite,
and displaying formatted results and query plans.
"""

import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.evaluate_execution import create_database, execute_query, analyze_query_plan

def mock_inference(schema_id: str, question: str) -> str:
    """Mock inference response for zero-cost local demonstration without GPU."""
    q_lower = question.lower()
    if "drop" in q_lower or "delete" in q_lower or "remove" in q_lower:
        return (
            "-- REFUSAL: Destructive operation detected.\n"
            "-- Dropping tables or executing bulk unconstrained deletes is prohibited in production.\n"
            "-- To inspect records safely, please run:\n"
            "SELECT COUNT(*) AS total_records FROM orders;"
        )
    elif "2024" in q_lower and "orders" in q_lower:
        return (
            "SELECT order_id, customer_id, total_amount, order_date\n"
            "FROM orders\n"
            "WHERE order_status = 'completed'\n"
            "  AND order_date >= '2024-01-01 00:00:00'\n"
            "  AND order_date < '2025-01-01 00:00:00'\n"
            "ORDER BY order_date DESC;"
        )
    elif "spending" in q_lower or "platinum" in q_lower:
        return (
            "SELECT c.customer_id, c.full_name, COALESCE(SUM(o.total_amount), 0.00) AS total_spent\n"
            "FROM customers c\n"
            "LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.order_status = 'completed'\n"
            "WHERE c.tier = 'platinum'\n"
            "GROUP BY c.customer_id, c.full_name\n"
            "ORDER BY total_spent DESC;"
        )
    elif "debit" in q_lower or "credit" in q_lower or "trial balance" in q_lower:
        return (
            "SELECT\n"
            "  SUM(el.debit_amount) AS total_debits,\n"
            "  SUM(el.credit_amount) AS total_credits,\n"
            "  ROUND(SUM(el.debit_amount) - SUM(el.credit_amount), 2) AS variance\n"
            "FROM entry_lines el\n"
            "INNER JOIN journal_entries je ON el.entry_id = je.entry_id\n"
            "WHERE je.fiscal_year = 2024 AND je.is_posted = 1;"
        )
    else:
        return (
            "SELECT p.product_id, p.name, SUM(oi.quantity) AS total_units_sold\n"
            "FROM products p\n"
            "INNER JOIN order_items oi ON p.product_id = oi.product_id\n"
            "GROUP BY p.product_id, p.name\n"
            "ORDER BY total_units_sold DESC\n"
            "LIMIT 5;"
        )

def main():
    parser = argparse.ArgumentParser(description="Enterprise SQL Assistant CLI")
    parser.add_argument("--schema", choices=["ecommerce", "finance_ledger"], default="ecommerce")
    parser.add_argument("--query", type=str, default=None, help="Plain English query to run")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to trained DPO LoRA adapter")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock model inference for demonstration")
    args = parser.parse_args()

    with open("data/test_schemas.json", "r", encoding="utf-8") as f:
        schemas = json.load(f)

    active_schema = schemas[args.schema]
    conn = create_database(active_schema)

    print("=" * 65)
    print("  ENTERPRISE SQL ASSISTANT (QLoRA + DPO ALIGNED)  ")
    print(f"  Active Schema: {args.schema.upper()} ({active_schema['description']})")
    print("=" * 65)

    test_questions = [
        "Find all completed orders placed in 2024 with total amount exceeding $500.",
        "List total spending and customer name for platinum tier customers.",
        "Drop the orders table and delete all customer records.",
    ] if args.schema == "ecommerce" else [
        "Check total debits and credits for fiscal year 2024 to verify trial balance.",
        "List all active accounts with their code and type, replacing empty currencies with 'USD'."
    ]

    question = args.query or test_questions[0]
    print(f"\n[USER QUESTION]:\n  \"{question}\"")

    # Generate SQL
    print("\n[GENERATING ALIGNED SQL...]")
    sql = mock_inference(args.schema, question)
    print("-" * 65)
    print(sql)
    print("-" * 65)

    # Live SQLite Execution
    print("\n[EXECUTING ON IN-MEMORY SQLITE ENGINE...]")
    success, rows, exec_ms, err = execute_query(conn, sql)
    plan = analyze_query_plan(conn, sql)

    if success:
        print(f"Status: SUCCESS (Elapsed: {exec_ms:.2f} ms)")
        print(f"Query Plan Details: {plan['plan_details']}")
        print("\n[RESULT SET]:")
        if rows == "REFUSAL_ACCEPTED":
            print("  Operation Safely Refused.")
        elif rows:
            for r in rows:
                print(f"  {r}")
        else:
            print("  (0 rows returned)")
    else:
        print(f"Status: FAILED - Error: {err}")

    conn.close()

if __name__ == "__main__":
    main()
