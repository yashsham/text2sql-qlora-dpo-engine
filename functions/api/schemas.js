// Cloudflare Pages Function: GET /api/schemas
export async function onRequestGet() {
  const schemas = {
    ecommerce: {
      db_id: "ecommerce",
      description: "Enterprise E-Commerce SaaS platform tracking customers, orders, order items, and products",
      ddl: `CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    tier TEXT CHECK(tier IN ('bronze', 'silver', 'gold', 'platinum')),
    country TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    stock_quantity INTEGER NOT NULL,
    is_active BOOLEAN DEFAULT 1
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_status TEXT CHECK(order_status IN ('pending', 'processing', 'completed', 'cancelled', 'refunded')),
    total_amount DECIMAL(10,2) NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);
CREATE INDEX idx_customers_tier ON customers(tier);
CREATE INDEX idx_products_category ON products(category);`
    },
    finance_ledger: {
      db_id: "finance_ledger",
      description: "Corporate General Ledger tracking accounts, fiscal quarters, journal entries, and reconciliations",
      ddl: `CREATE TABLE accounts (
    account_id INTEGER PRIMARY KEY,
    account_code TEXT UNIQUE NOT NULL,
    account_name TEXT NOT NULL,
    account_type TEXT CHECK(account_type IN ('asset', 'liability', 'equity', 'revenue', 'expense')),
    currency TEXT DEFAULT 'USD'
);

CREATE TABLE journal_entries (
    entry_id INTEGER PRIMARY KEY,
    reference_no TEXT UNIQUE NOT NULL,
    entry_date DATE NOT NULL,
    description TEXT,
    fiscal_year INTEGER NOT NULL,
    is_posted BOOLEAN DEFAULT 0
);

CREATE TABLE entry_lines (
    line_id INTEGER PRIMARY KEY,
    entry_id INTEGER NOT NULL,
    account_id INTEGER NOT NULL,
    debit_amount DECIMAL(14,2) DEFAULT 0.00,
    credit_amount DECIMAL(14,2) DEFAULT 0.00,
    FOREIGN KEY (entry_id) REFERENCES journal_entries(entry_id),
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);
CREATE INDEX idx_journal_date ON journal_entries(entry_date, fiscal_year);
CREATE INDEX idx_lines_account ON entry_lines(account_id);`
    }
  };

  return new Response(JSON.stringify(schemas), {
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
      "Cache-Control": "public, max-age=3600"
    }
  });
}
