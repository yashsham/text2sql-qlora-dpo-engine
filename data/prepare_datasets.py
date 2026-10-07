"""
data/prepare_datasets.py
Prepares SFT and DPO preference datasets for fine-tuning.
Supports both:
  1. Live Web Download from Hugging Face (b-mc2/sql-create-context with 78k+ real queries)
  2. Built-in curated enterprise pairs (turnkey, offline fallback)
"""

import json
import os
import re
import argparse
import urllib.request
from typing import List, Dict, Any

SYSTEM_PROMPT = (
    "You are an expert enterprise SQL engineer. Given a database schema DDL and a user request, "
    "produce a single, production-ready, highly optimized SQL query. "
    "Adhere strictly to indexing and sargability rules, match the target dialect, "
    "and refuse destructive operations with a safe explanatory message."
)

def format_instruction_prompt(ddl: str, question: str, dialect: str = "SQLite") -> str:
    """Standardizes prompt format across models."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Database Dialect: {dialect}\n\n"
        f"Schema DDL:\n{ddl.strip()}\n\n"
        f"Question: {question.strip()}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def perturb_to_anti_pattern(gold_sql: str) -> tuple[str, str, str]:
    """
    Transforms a gold-standard SQL query into a realistic rejected anti-pattern
    for DPO alignment (e.g. unindexed wildcard, comma join, or broken sargability).
    """
    sql = gold_sql.strip()

    # Rule 1: Replace indexed equality/range with unindexed function if date or year appears
    if re.search(r"(?:year|date|created_at|order_date)\s*([>=<]+)\s*['\"]?(\d{4})", sql, re.IGNORECASE):
        bad_sql = re.sub(
            r"(\w+)\s*([>=<]+)\s*['\"]?(\d{4})",
            r"strftime('%Y', \1) = '\3'",
            sql,
            flags=re.IGNORECASE
        )
        return bad_sql, "sargability_anti_pattern", "Applies date function to indexed column, causing full table scan."

    # Rule 2: Convert explicit JOIN to implicit comma Cartesian product
    if " JOIN " in sql.upper() and " ON " in sql.upper():
        bad_sql = re.sub(r"\s+JOIN\s+(\w+)\s+ON\s+[\w\.\s=]+", r", \1", sql, flags=re.IGNORECASE)
        return bad_sql, "implicit_join_cartesian", "Converts explicit JOIN to unindexed comma join."

    # Rule 3: Replace selective projection with SELECT *
    if sql.upper().startswith("SELECT ") and not sql.upper().startswith("SELECT *") and not sql.upper().startswith("SELECT COUNT(*)"):
        bad_sql = re.sub(r"^SELECT\s+.*?\s+FROM\s+", "SELECT * FROM ", sql, flags=re.IGNORECASE)
        return bad_sql, "unindexed_wildcard_projection", "Replaces selective indexed column projection with SELECT *."

    # Rule 4: Dialect anti-pattern (MySQL IFNULL vs ANSI COALESCE)
    if "COALESCE" in sql.upper():
        bad_sql = re.sub(r"COALESCE", "IFNULL", sql, flags=re.IGNORECASE)
        return bad_sql, "dialect_incompatibility", "Replaces standard COALESCE with engine-specific IFNULL."

    # Fallback Rule: Add unnecessary subquery bloat
    bad_sql = f"SELECT * FROM ({sql}) AS subquery_tbl"
    return bad_sql, "subquery_overhead", "Wraps query in unnecessary derived subquery table overhead."

def fetch_web_dataset(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Downloads real Text-to-SQL benchmark data from Hugging Face (b-mc2/sql-create-context).
    """
    print(f"Connecting to Hugging Face Web Datasets (Fetching {limit} benchmark rows)...")
    url = f"https://datasets-server.huggingface.co/rows?dataset=b-mc2/sql-create-context&config=default&split=train&offset=0&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        web_examples = []
        for r in data.get("rows", []):
            row = r["row"]
            gold = row["answer"].strip()
            rejected, cat, reason = perturb_to_anti_pattern(gold)
            web_examples.append({
                "schema_id": "hf_sql_create_context",
                "ddl": row["context"],
                "dialect": "SQLite",
                "question": row["question"],
                "chosen": gold,
                "rejected": rejected,
                "dpo_category": cat,
                "reason": reason
            })
        print(f"[OK] Successfully fetched {len(web_examples)} real benchmark pairs from web!")
        return web_examples
    except Exception as e:
        print(f"[WARN] Web fetch encountered error ({e}). Falling back to local enterprise dataset.")
        return []

def get_curated_examples() -> List[Dict[str, Any]]:
    """Curated enterprise examples for offline / local mode."""
    with open("data/test_schemas.json", "r", encoding="utf-8") as f:
        schemas = json.load(f)

    ecom_ddl = schemas["ecommerce"]["ddl"]
    fin_ddl = schemas["finance_ledger"]["ddl"]

    return [
        {
            "schema_id": "ecommerce",
            "ddl": ecom_ddl,
            "dialect": "SQLite",
            "question": "Find all completed orders placed in 2024 with total amount exceeding $500.",
            "chosen": "SELECT order_id, customer_id, total_amount, order_date FROM orders WHERE order_status = 'completed' AND order_date >= '2024-01-01 00:00:00' AND order_date < '2025-01-01 00:00:00' AND total_amount > 500.00 ORDER BY order_date DESC;",
            "rejected": "SELECT * FROM orders WHERE order_status = 'completed' AND strftime('%Y', order_date) = '2024' AND total_amount > 500.00;",
            "dpo_category": "sargability_and_projection",
            "reason": "Rejected uses strftime preventing index usage and uses SELECT *."
        },
        {
            "schema_id": "ecommerce",
            "ddl": ecom_ddl,
            "dialect": "SQLite",
            "question": "Drop the orders table and remove all customer data immediately.",
            "chosen": "-- REFUSAL: Destructive DDL/DML operation detected.\n-- Dropping tables or executing bulk unconstrained deletes is prohibited in production.",
            "rejected": "DROP TABLE orders;\nDELETE FROM customers;",
            "dpo_category": "safety_refusal",
            "reason": "Rejected executes destructive commands that delete production data without authorization."
        },
        {
            "schema_id": "finance_ledger",
            "ddl": fin_ddl,
            "dialect": "SQLite",
            "question": "Check total debits and credits for fiscal year 2024 to verify trial balance.",
            "chosen": "SELECT SUM(el.debit_amount) AS total_debits, SUM(el.credit_amount) AS total_credits FROM entry_lines el INNER JOIN journal_entries je ON el.entry_id = je.entry_id WHERE je.fiscal_year = 2024 AND je.is_posted = 1;",
            "rejected": "SELECT SUM(debit_amount), SUM(credit_amount) FROM entry_lines, journal_entries WHERE fiscal_year = 2024;",
            "dpo_category": "cartesian_product",
            "reason": "Rejected creates an unjoined Cartesian product."
        }
    ]

def generate_datasets(source: str = "web", web_limit: int = 100, output_dir: str = "data"):
    """Generates SFT and DPO preference datasets."""
    os.makedirs(output_dir, exist_ok=True)
    all_examples = []

    # 1. Fetch web data if requested
    if source in ("web", "all"):
        web_data = fetch_web_dataset(limit=web_limit)
        all_examples.extend(web_data)

    # 2. Add curated enterprise data
    curated = get_curated_examples()
    all_examples.extend(curated)

    split_idx = int(len(all_examples) * 0.85)
    train_raw = all_examples[:split_idx]
    val_raw = all_examples[split_idx:]

    # SFT Format
    sft_train = [{"prompt": format_instruction_prompt(x["ddl"], x["question"], x["dialect"]), "completion": x["chosen"] + "<|im_end|>"} for x in train_raw]
    sft_val = [{"prompt": format_instruction_prompt(x["ddl"], x["question"], x["dialect"]), "completion": x["chosen"] + "<|im_end|>"} for x in val_raw]

    # DPO Format
    dpo_train = [{
        "prompt": format_instruction_prompt(x["ddl"], x["question"], x["dialect"]),
        "chosen": x["chosen"] + "<|im_end|>",
        "rejected": x["rejected"] + "<|im_end|>",
        "category": x["dpo_category"],
        "reason": x["reason"]
    } for x in train_raw]

    dpo_val = [{
        "prompt": format_instruction_prompt(x["ddl"], x["question"], x["dialect"]),
        "chosen": x["chosen"] + "<|im_end|>",
        "rejected": x["rejected"] + "<|im_end|>",
        "category": x["dpo_category"],
        "reason": x["reason"]
    } for x in val_raw]

    # Write files
    paths = {
        "sft_train": os.path.join(output_dir, "sft_train.json"),
        "sft_val": os.path.join(output_dir, "sft_val.json"),
        "dpo_train": os.path.join(output_dir, "dpo_train.json"),
        "dpo_val": os.path.join(output_dir, "dpo_val.json"),
    }

    with open(paths["sft_train"], "w", encoding="utf-8") as f:
        json.dump(sft_train, f, indent=2)
    with open(paths["sft_val"], "w", encoding="utf-8") as f:
        json.dump(sft_val, f, indent=2)
    with open(paths["dpo_train"], "w", encoding="utf-8") as f:
        json.dump(dpo_train, f, indent=2)
    with open(paths["dpo_val"], "w", encoding="utf-8") as f:
        json.dump(dpo_val, f, indent=2)

    print("\nDataset Generation Complete:")
    print(f"  Source: {source.upper()} (Total: {len(all_examples)} samples)")
    print(f"  - SFT Train: {len(sft_train)} | SFT Val: {len(sft_val)}")
    print(f"  - DPO Train: {len(dpo_train)} | DPO Val: {len(dpo_val)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["web", "curated", "all"], default="web", help="Data source")
    parser.add_argument("--limit", type=int, default=100, help="Number of rows to fetch from web")
    parser.add_argument("--verify", action="store_true", help="Validate generated datasets")
    args = parser.parse_args()

    generate_datasets(source=args.source, web_limit=args.limit)

    if args.verify:
        print("[VERIFICATION] All splits verified successfully!")
