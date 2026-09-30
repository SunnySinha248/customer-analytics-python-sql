-- Reference schema (created by src/generate_data.py via pandas)
CREATE TABLE customers (
    customer_id       INTEGER,
    signup_date       TEXT,
    region            TEXT,
    age               REAL,
    preferred_channel TEXT
);
CREATE TABLE products (
    product_id INTEGER,
    category   TEXT,
    unit_price REAL,
    unit_cost  REAL
);
CREATE TABLE orders (
    order_id     INTEGER,
    customer_id  INTEGER,
    order_date   TEXT,
    channel      TEXT,
    discount_pct REAL
);
CREATE TABLE order_items (
    order_id   INTEGER,
    product_id INTEGER,
    quantity   INTEGER
);
