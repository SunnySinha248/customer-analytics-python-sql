-- 03: Customer-level feature table (RFM + behaviour) used for segmentation and modelling.
-- Reference date = day after last order in data.
DROP VIEW IF EXISTS v_customer_features;
CREATE VIEW v_customer_features AS
WITH ref AS (SELECT DATE(MAX(order_date), '+1 day') AS ref_date FROM v_order_lines),
agg AS (
    SELECT
        customer_id,
        MAX(order_date)              AS last_order,
        MIN(order_date)              AS first_order,
        COUNT(DISTINCT order_id)     AS frequency,
        SUM(revenue)                 AS monetary,
        SUM(profit)                  AS profit,
        SUM(quantity)                AS units,
        COUNT(DISTINCT category)     AS categories_bought
    FROM v_order_lines
    GROUP BY customer_id
)
SELECT
    a.customer_id,
    c.region,
    c.age,
    c.preferred_channel,
    CAST(julianday(r.ref_date) - julianday(a.last_order) AS INTEGER)  AS recency_days,
    CAST(julianday(a.last_order) - julianday(a.first_order) AS INTEGER) AS tenure_days,
    a.frequency,
    ROUND(a.monetary, 2)  AS monetary,
    ROUND(a.profit, 2)    AS profit,
    a.units,
    a.categories_bought
FROM agg a
JOIN customers c ON c.customer_id = a.customer_id
CROSS JOIN ref r;
