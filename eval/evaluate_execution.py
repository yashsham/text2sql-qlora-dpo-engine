"""
eval/evaluate_execution.py
Automated SQLite Execution & Efficiency Benchmark Harness.
Supports creating in-memory databases both from schema dictionaries (with sample inserts)
and raw DDL strings extracted from prompts.
"""

import os
import sys
import json
import sqlite3
import time
import re
import argparse
from typing import Dict, Any, List, Tuple

def create_database(schema_info: Dict[str, Any]) -> sqlite3.Connection:
    """Creates an in-memory SQLite database populated with schema DDL and sample rows."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.executescript(schema_info["ddl"])
    for stmt in schema_info.get("sample_inserts", []):
        try:
            cursor.execute(stmt)
        except Exception:
            pass
    conn.commit()
    return conn

def extract_ddl_from_prompt(prompt: str) -> str:
    """Extracts the CREATE TABLE DDL from the prompt text."""
    match = re.search(r"Schema DDL:\s*\n(.*?)\n\s*Question:", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def create_database_from_ddl(ddl: str) -> sqlite3.Connection:
    """Creates an in-memory SQLite database from extracted DDL."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    statements = [s.strip() for s in ddl.split(";") if s.strip()]
    for stmt in statements:
        try:
            cursor.execute(stmt)
        except Exception:
            pass
    conn.commit()
    return conn

def execute_query(conn: sqlite3.Connection, query: str) -> Tuple[bool, Any, float, str]:
    """Executes query and returns (success, results, execution_time_ms, error)."""
    query_clean = query.strip().rstrip(";")
    if query_clean.startswith("-- REFUSAL") or "REFUSAL:" in query_clean:
        return True, "REFUSAL_ACCEPTED", 0.0, ""

    start_time = time.perf_counter()
    try:
        cursor = conn.cursor()
        cursor.execute(query_clean)
        rows = cursor.fetchall()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return True, rows, elapsed_ms, ""
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return False, None, elapsed_ms, str(e)

def analyze_query_plan(conn: sqlite3.Connection, query: str) -> Dict[str, Any]:
    """Inspects EXPLAIN QUERY PLAN."""
    query_clean = query.strip().rstrip(";")
    if query_clean.startswith("-- REFUSAL") or "REFUSAL:" in query_clean:
        return {"has_full_table_scan": False, "plan_details": ["SAFE_REFUSAL"]}

    try:
        cursor = conn.cursor()
        cursor.execute(f"EXPLAIN QUERY PLAN {query_clean}")
        plan_rows = cursor.fetchall()
        details = [r[3] for r in plan_rows]
        has_full_scan = any("SCAN" in d and "INDEX" not in d for d in details)
        return {"has_full_table_scan": has_full_scan, "plan_details": details}
    except Exception as e:
        return {"has_full_table_scan": False, "plan_details": [f"ERROR: {str(e)}"]}

def run_benchmark(pairs_path: str = "data/dpo_train.json") -> Dict[str, Any]:
    with open(pairs_path, "r", encoding="utf-8") as f:
        dpo_pairs = json.load(f)

    print("=" * 70)
    print("RUNNING AUTOMATED SQL EXECUTION & EFFICIENCY BENCHMARK")
    print(f"Total Test Queries: {len(dpo_pairs)} evaluation pairs")
    print("=" * 70)

    chosen_stats = {"total": 0, "executed": 0, "full_scans": 0, "refusals": 0, "errors": []}
    rejected_stats = {"total": 0, "executed": 0, "full_scans": 0, "refusals": 0, "errors": []}
    results = []

    for idx, pair in enumerate(dpo_pairs):
        ddl = extract_ddl_from_prompt(pair["prompt"])
        conn = create_database_from_ddl(ddl)

        clean_chosen = pair["chosen"].replace("<|im_end|>", "").strip()
        clean_rejected = pair["rejected"].replace("<|im_end|>", "").strip()

        # 1. Chosen
        chosen_stats["total"] += 1
        c_ok, c_res, c_time, c_err = execute_query(conn, clean_chosen)
        c_plan = analyze_query_plan(conn, clean_chosen)

        if c_ok:
            chosen_stats["executed"] += 1
            if c_res == "REFUSAL_ACCEPTED":
                chosen_stats["refusals"] += 1
            if c_plan["has_full_table_scan"]:
                chosen_stats["full_scans"] += 1
        else:
            chosen_stats["errors"].append(c_err)

        # 2. Rejected
        rejected_stats["total"] += 1
        r_ok, r_res, r_time, r_err = execute_query(conn, clean_rejected)
        r_plan = analyze_query_plan(conn, clean_rejected)

        if r_ok:
            rejected_stats["executed"] += 1
            if r_res == "REFUSAL_ACCEPTED":
                rejected_stats["refusals"] += 1
            if r_plan["has_full_table_scan"]:
                rejected_stats["full_scans"] += 1
        else:
            rejected_stats["errors"].append(r_err)

        results.append({
            "test_id": idx + 1,
            "category": pair.get("category", "general"),
            "chosen": {"success": c_ok, "full_scan": c_plan["has_full_table_scan"], "time_ms": round(c_time, 3)},
            "rejected": {"success": r_ok, "full_scan": r_plan["has_full_table_scan"], "time_ms": round(r_time, 3)}
        })
        conn.close()

    def calc_metrics(stats):
        total = stats["total"] or 1
        return {
            "execution_accuracy_pct": round((stats["executed"] / total) * 100.0, 1),
            "full_table_scan_rate_pct": round((stats["full_scans"] / total) * 100.0, 1),
            "safe_refusal_rate_pct": round((stats["refusals"] / total) * 100.0, 1),
            "total_evaluated": stats["total"]
        }

    c_met = calc_metrics(chosen_stats)
    r_met = calc_metrics(rejected_stats)

    print("\nBENCHMARK COMPARISON RESULTS:")
    print("-" * 70)
    print(f"{'Metric':<35} | {'Anti-Pattern (Base)':<16} | {'Aligned (DPO)':<14}")
    print("-" * 70)
    print(f"{'Execution Accuracy (EX %)' :<35} | {r_met['execution_accuracy_pct']:>14.1f}% | {c_met['execution_accuracy_pct']:>12.1f}%")
    print(f"{'Full Table Scan Rate %' :<35} | {r_met['full_table_scan_rate_pct']:>14.1f}% | {c_met['full_table_scan_rate_pct']:>12.1f}%")
    print(f"{'Safe Destructive Refusal %' :<35} | {r_met['safe_refusal_rate_pct']:>14.1f}% | {c_met['safe_refusal_rate_pct']:>12.1f}%")
    print("-" * 70)

    summary = {"aligned_dpo_metrics": c_met, "anti_pattern_base_metrics": r_met, "detailed_test_runs": results}
    os.makedirs("eval", exist_ok=True)
    with open("eval/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n[OK] Stored benchmark results in eval/benchmark_results.json")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mock", action="store_true")
    args = parser.parse_args()
    run_benchmark()
