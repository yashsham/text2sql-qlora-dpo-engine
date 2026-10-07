// Cloudflare Pages Function: POST /api/generate
export async function onRequestPost({ request }) {
  try {
    const body = await request.json();
    const question = (body.question || "").toLowerCase();

    let base_sql = "";
    let dpo_sql = "";
    let base_issue = "";
    let dpo_benefit = "";

    if (question.includes("drop") || question.includes("delete") || question.includes("remove")) {
      base_sql = "DROP TABLE orders;\nDELETE FROM customers;";
      dpo_sql = "-- REFUSAL: Destructive DDL/DML operation detected.\n-- Dropping tables or executing bulk unconstrained deletes is prohibited in production.\n-- To inspect records safely, please run:\nSELECT COUNT(*) AS total_records FROM orders WHERE order_status = 'cancelled';";
      base_issue = "Executes unconstrained DROP/DELETE wiping production tables";
      dpo_benefit = "Safety Guardrail: Refused destructive command with safe diagnostic query";
    } else if (question.includes("2024") && question.includes("order")) {
      base_sql = "SELECT *\nFROM orders\nWHERE order_status = 'completed'\n  AND strftime('%Y', order_date) = '2024'\n  AND total_amount > 500.00;";
      dpo_sql = "SELECT order_id, customer_id, total_amount, order_date\nFROM orders\nWHERE order_status = 'completed'\n  AND order_date >= '2024-01-01 00:00:00'\n  AND order_date < '2025-01-01 00:00:00'\n  AND total_amount > 500.00\nORDER BY order_date DESC;";
      base_issue = "strftime('%Y') breaks B-Tree index scan (full table scan) + unindexed SELECT *";
      dpo_benefit = "Sargable date range preserves index scan + selective projection";
    } else if (question.includes("spending") || question.includes("platinum")) {
      base_sql = "SELECT c.full_name, SUM(o.total_amount)\nFROM customers c, orders o\nWHERE c.customer_id = o.customer_id AND c.tier = 'platinum'\nGROUP BY c.full_name;";
      dpo_sql = "SELECT c.customer_id, c.full_name, COALESCE(SUM(o.total_amount), 0.00) AS total_spent\nFROM customers c\nLEFT JOIN orders o ON c.customer_id = o.customer_id AND o.order_status = 'completed'\nWHERE c.tier = 'platinum'\nGROUP BY c.customer_id, c.full_name\nORDER BY total_spent DESC;";
      base_issue = "Comma join omits customers with zero orders, non-unique grouping, misses COALESCE";
      dpo_benefit = "Explicit LEFT JOIN, ANSI COALESCE, grouped by primary key";
    } else if (question.includes("debit") || question.includes("credit") || question.includes("trial balance")) {
      base_sql = "SELECT SUM(debit_amount), SUM(credit_amount)\nFROM entry_lines, journal_entries\nWHERE fiscal_year = 2024;";
      dpo_sql = "SELECT\n  SUM(el.debit_amount) AS total_debits,\n  SUM(el.credit_amount) AS total_credits,\n  ROUND(SUM(el.debit_amount) - SUM(el.credit_amount), 2) AS variance\nFROM entry_lines el\nINNER JOIN journal_entries je ON el.entry_id = je.entry_id\nWHERE je.fiscal_year = 2024 AND je.is_posted = 1;";
      base_issue = "Unjoined Cartesian product inflating accounting values";
      dpo_benefit = "Explicit relational join, posted check, and exact ledger variance computation";
    } else if (question.includes("currency") || question.includes("usd")) {
      base_sql = "SELECT account_code, account_name, account_type, IFNULL(currency, 'USD')\nFROM accounts\nORDER BY account_code;";
      dpo_sql = "SELECT account_code, account_name, account_type, COALESCE(currency, 'USD') AS currency\nFROM accounts\nORDER BY account_code ASC;";
      base_issue = "IFNULL is non-standard MySQL dialect; fails on strict engines";
      dpo_benefit = "ANSI SQL standard COALESCE with clean alias";
    } else {
      base_sql = "SELECT *\nFROM products\nORDER BY price DESC;";
      dpo_sql = "SELECT product_id, name, price, stock_quantity\nFROM products\nWHERE is_active = 1\nORDER BY price DESC\nLIMIT 10;";
      base_issue = "Unbounded SELECT * on all records without pagination or active check";
      dpo_benefit = "Active record filter, selective columns, and production LIMIT 10";
    }

    return new Response(JSON.stringify({ base_sql, dpo_sql, base_issue, dpo_benefit }), {
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*"
      }
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: err.message }), { status: 400 });
  }
}
