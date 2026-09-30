"""Generate a synthetic retail database (SQLite) for the project.

The data is synthetic so the repo is fully reproducible and contains no
confidential information. To use your own data, load it into the same
table layout (see sql/schema.sql) and re-run the pipeline.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "retail.db"

REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["Online", "Store", "Marketplace"]
CATEGORIES = {
    "Electronics": (80, 600),
    "Apparel": (15, 120),
    "Home": (20, 250),
    "Grocery": (2, 40),
    "Beauty": (8, 90),
}


def build_customers(rng, n=2500):
    signup = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 730, n), unit="D"
    )
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "signup_date": signup.strftime("%Y-%m-%d"),
            "region": rng.choice(REGIONS, n, p=[0.3, 0.2, 0.25, 0.25]),
            "age": np.clip(rng.normal(36, 11, n), 18, 75).round().astype(int),
            "preferred_channel": rng.choice(CHANNELS, n, p=[0.5, 0.35, 0.15]),
        }
    )
    # latent engagement drives order frequency and spend
    df["_engagement"] = rng.gamma(2.0, 1.0, n)
    return df


def build_products(rng, n=150):
    cats = rng.choice(list(CATEGORIES), n)
    price = [round(rng.uniform(*CATEGORIES[c]), 2) for c in cats]
    return pd.DataFrame(
        {
            "product_id": np.arange(1, n + 1),
            "category": cats,
            "unit_price": price,
            "unit_cost": [round(p * rng.uniform(0.45, 0.8), 2) for p in price],
        }
    )


def build_orders(rng, customers, products):
    end = pd.Timestamp("2024-12-31")
    orders, items = [], []
    oid = 1
    for _, c in customers.iterrows():
        start = pd.Timestamp(c.signup_date)
        days = (end - start).days
        if days <= 0:
            continue
        n_orders = rng.poisson(max(0.3, c._engagement * days / 150))
        for _ in range(n_orders):
            odate = start + pd.Timedelta(days=int(rng.integers(0, days + 1)))
            # mild seasonality: Q4 uplift via extra orders
            orders.append(
                (oid, c.customer_id, odate.strftime("%Y-%m-%d"),
                 rng.choice(CHANNELS, p=[0.5, 0.35, 0.15]),
                 round(float(rng.choice([0, 0, 0, 5, 10, 15])), 2))
            )
            for _ in range(int(rng.integers(1, 5))):
                p = products.iloc[int(rng.integers(0, len(products)))]
                items.append((oid, int(p.product_id), int(rng.integers(1, 4))))
            oid += 1
            if odate.month in (10, 11, 12) and rng.random() < 0.35:
                orders.append(
                    (oid, c.customer_id, odate.strftime("%Y-%m-%d"),
                     "Online", round(float(rng.choice([10, 15, 20])), 2))
                )
                p = products.iloc[int(rng.integers(0, len(products)))]
                items.append((oid, int(p.product_id), int(rng.integers(1, 3))))
                oid += 1
    o = pd.DataFrame(orders, columns=["order_id", "customer_id", "order_date", "channel", "discount_pct"])
    i = pd.DataFrame(items, columns=["order_id", "product_id", "quantity"])
    return o, i


def inject_quality_issues(rng, customers, orders):
    """Add realistic dirt so the cleaning step has something to do."""
    customers = customers.copy()
    orders = orders.copy()
    customers.loc[rng.choice(customers.index, 40, replace=False), "age"] = np.nan
    customers.loc[rng.choice(customers.index, 15, replace=False), "age"] = 150
    customers.loc[rng.choice(customers.index, 30, replace=False), "region"] = None
    dupes = orders.sample(25, random_state=SEED)
    orders = pd.concat([orders, dupes], ignore_index=True)
    return customers, orders


def main():
    rng = np.random.default_rng(SEED)
    customers = build_customers(rng)
    products = build_products(rng)
    orders, items = build_orders(rng, customers, products)
    customers, orders = inject_quality_issues(rng, customers, orders)

    DB_PATH.parent.mkdir(exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    customers.drop(columns="_engagement").to_sql("customers", con, index=False)
    products.to_sql("products", con, index=False)
    orders.to_sql("orders", con, index=False)
    items.to_sql("order_items", con, index=False)
    con.close()
    print(f"Wrote {DB_PATH}: {len(customers)} customers, {len(orders)} orders, {len(items)} order lines")


if __name__ == "__main__":
    main()
