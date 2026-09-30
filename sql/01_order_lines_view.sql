-- 01: Order-line fact view with joins and revenue/profit calculations.
-- De-duplicates orders (raw data contains repeated order rows).
DROP VIEW IF EXISTS v_order_lines;
CREATE VIEW v_order_lines AS
SELECT
    o.order_id,
    o.customer_id,
    o.order_date,
    o.channel,
    c.region,
    c.age,
    p.category,
    oi.quantity,
    p.unit_price * oi.quantity * (1 - o.discount_pct / 100.0)            AS revenue,
    (p.unit_price * (1 - o.discount_pct / 100.0) - p.unit_cost) * oi.quantity AS profit
FROM (SELECT DISTINCT * FROM orders) o
JOIN order_items oi ON oi.order_id   = o.order_id
JOIN products    p  ON p.product_id  = oi.product_id
JOIN customers   c  ON c.customer_id = o.customer_id
WHERE o.order_date BETWEEN '2023-01-01' AND '2024-12-31';
