-- 02: KPI aggregations (GROUP BY) used in the EDA notebook/report.

-- Overall KPIs
SELECT
    COUNT(DISTINCT order_id)                         AS orders,
    COUNT(DISTINCT customer_id)                      AS active_customers,
    ROUND(SUM(revenue), 2)                           AS revenue,
    ROUND(SUM(profit), 2)                            AS profit,
    ROUND(SUM(profit) / SUM(revenue) * 100, 2)       AS margin_pct,
    ROUND(SUM(revenue) / COUNT(DISTINCT order_id), 2) AS avg_order_value
FROM v_order_lines;

-- KPIs by category
SELECT
    category,
    ROUND(SUM(revenue), 2)                     AS revenue,
    ROUND(SUM(profit), 2)                      AS profit,
    ROUND(SUM(profit) / SUM(revenue) * 100, 2) AS margin_pct,
    SUM(quantity)                              AS units
FROM v_order_lines
GROUP BY category
ORDER BY revenue DESC;

-- KPIs by region and channel
SELECT
    region,
    channel,
    COUNT(DISTINCT order_id)                   AS orders,
    ROUND(SUM(revenue), 2)                     AS revenue,
    ROUND(SUM(profit) / SUM(revenue) * 100, 2) AS margin_pct
FROM v_order_lines
WHERE region IS NOT NULL
GROUP BY region, channel
ORDER BY region, revenue DESC;

-- Monthly revenue (feeds the forecasting model)
SELECT
    strftime('%Y-%m', order_date) AS month,
    ROUND(SUM(revenue), 2)        AS revenue,
    COUNT(DISTINCT order_id)      AS orders
FROM v_order_lines
GROUP BY month
ORDER BY month;
