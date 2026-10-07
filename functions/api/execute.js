// Cloudflare Pages Function: POST /api/execute
export async function onRequestPost({ request }) {
  try {
    const body = await request.json();
    const query = (body.query || "").trim();
    const isRefusal = query.startsWith("-- REFUSAL") || query.includes("REFUSAL:");

    if (isRefusal) {
      return new Response(JSON.stringify({
        success: true,
        is_refusal: true,
        elapsed_ms: 0.1,
        plan: ["SAFE_REFUSAL_POLICY_ENFORCED"],
        columns: ["Guardrail"],
        rows: [["Operation securely refused"]]
      }), {
        headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" }
      });
    }

    // Edge simulated database records
    let columns = ["Status"];
    let rows = [["OK"]];
    let plan = ["SEARCH TABLE USING INDEX"];

    if (query.includes("orders")) {
      columns = ["order_id", "customer_id", "total_amount", "order_date"];
      rows = [
        [1002, 1, 499.99, "2024-05-12 12:30:00"],
        [1003, 2, 1299.00, "2024-04-01 09:00:00"],
        [1001, 1, 1798.99, "2024-03-10 10:00:00"]
      ];
      if (query.includes("strftime")) {
        plan = ["SCAN orders (FULL TABLE SCAN)", "USE TEMP B-TREE FOR ORDER BY"];
      } else {
        plan = ["SEARCH orders USING INDEX idx_orders_customer_date (order_date>=? AND order_date<?)"];
      }
    } else if (query.includes("customers")) {
      columns = ["customer_id", "full_name", "total_spent"];
      rows = [
        [1, "Alice Smith", 2298.98],
        [4, "Diana Prince", 499.99]
      ];
      plan = ["SEARCH customers USING INDEX idx_customers_tier (tier=?)", "LEFT JOIN orders ON customer_id"];
    } else if (query.includes("debit_amount") || query.includes("journal_entries")) {
      columns = ["total_debits", "total_credits", "variance"];
      rows = [[147500.00, 147500.00, 0.00]];
      plan = ["SEARCH journal_entries USING INDEX idx_journal_date (fiscal_year=?)", "JOIN entry_lines ON entry_id"];
    } else if (query.includes("accounts")) {
      columns = ["account_code", "account_name", "account_type", "currency"];
      rows = [
        ["1010", "Operating Cash", "asset", "USD"],
        ["1200", "Accounts Receivable", "asset", "USD"],
        ["2000", "Accounts Payable", "liability", "USD"],
        ["4000", "Subscription Revenue", "revenue", "USD"]
      ];
      plan = ["SCAN accounts"];
    } else {
      columns = ["product_id", "name", "price", "stock_quantity"];
      rows = [
        [102, "Data Analytics Suite", 1299.00, 25],
        [101, "Cloud Server Pro", 499.99, 50],
        [103, "Security Audit Addon", 250.00, 100]
      ];
      plan = ["SEARCH products USING INDEX idx_products_category"];
    }

    return new Response(JSON.stringify({
      success: true,
      is_refusal: false,
      elapsed_ms: Math.random() * 0.4 + 0.1,
      plan: plan,
      columns: columns,
      rows: rows
    }), {
      headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" }
    });

  } catch (err) {
    return new Response(JSON.stringify({ success: false, error: err.message }), { status: 400 });
  }
}
