"""End-to-end pipeline: SQL extraction -> validation/cleaning -> EDA & stats ->
segmentation -> decision tree -> forecasting.

Run:  python src/analysis.py
Outputs go to outputs/figures and outputs/tables.
"""
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (classification_report, mean_absolute_percentage_error,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "retail.db"
SQL = ROOT / "sql"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
SEED = 42
sns.set_theme(style="whitegrid")


# ----------------------------------------------------------------------------
# 1. SQL extraction
# ----------------------------------------------------------------------------
def run_sql_file(con, name):
    con.executescript((SQL / name).read_text())


def split_queries(text):
    return [q.strip() for q in text.split(";") if "SELECT" in q.upper()]


def extract(con):
    run_sql_file(con, "01_order_lines_view.sql")
    run_sql_file(con, "03_customer_features.sql")
    kpi_queries = split_queries((SQL / "02_kpis.sql").read_text())
    names = ["kpi_overall", "kpi_by_category", "kpi_by_region_channel", "monthly_revenue"]
    kpis = {n: pd.read_sql_query(q, con) for n, q in zip(names, kpi_queries)}
    for n, df in kpis.items():
        df.to_csv(TAB / f"{n}.csv", index=False)
    lines = pd.read_sql_query("SELECT * FROM v_order_lines", con, parse_dates=["order_date"])
    feats = pd.read_sql_query("SELECT * FROM v_customer_features", con)
    return kpis, lines, feats


# ----------------------------------------------------------------------------
# 2. Data validation & cleaning
# ----------------------------------------------------------------------------
def validate_and_clean(con):
    """Run governance checks on RAW tables, log results, return cleaned customers."""
    cust = pd.read_sql_query("SELECT * FROM customers", con)
    orders = pd.read_sql_query("SELECT * FROM orders", con)
    items = pd.read_sql_query("SELECT * FROM order_items", con)
    prods = pd.read_sql_query("SELECT * FROM products", con)

    checks = {
        "duplicate_order_rows": int(orders.duplicated().sum()),
        "duplicate_customer_ids": int(cust["customer_id"].duplicated().sum()),
        "null_age": int(cust["age"].isna().sum()),
        "null_region": int(cust["region"].isna().sum()),
        "age_out_of_range_(<18_or_>90)": int(((cust["age"] < 18) | (cust["age"] > 90)).sum()),
        "orphan_order_items": int((~items["order_id"].isin(orders["order_id"])).sum()),
        "orphan_orders_no_customer": int((~orders["customer_id"].isin(cust["customer_id"])).sum()),
        "non_positive_quantity": int((items["quantity"] <= 0).sum()),
        "price_below_cost": int((prods["unit_price"] < prods["unit_cost"]).sum()),
    }
    pd.Series(checks, name="issues_found").to_csv(TAB / "data_quality_checks.csv")

    clean = cust.copy()
    clean.loc[(clean["age"] < 18) | (clean["age"] > 90), "age"] = np.nan
    clean["age"] = clean["age"].fillna(clean["age"].median())
    clean["region"] = clean["region"].fillna("Unknown")
    return checks, clean


# ----------------------------------------------------------------------------
# 3. EDA + statistical analysis
# ----------------------------------------------------------------------------
def eda(kpis, lines):
    out = {}
    monthly = kpis["monthly_revenue"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(monthly["month"]), monthly["revenue"], marker="o")
    ax.set(title="Monthly revenue", xlabel="", ylabel="Revenue")
    fig.tight_layout(); fig.savefig(FIG / "01_monthly_revenue.png", dpi=150); plt.close(fig)

    cat = kpis["kpi_by_category"]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=cat, x="category", y="revenue", ax=ax, color="#4c72b0")
    ax.set(title="Revenue by category", xlabel="")
    fig.tight_layout(); fig.savefig(FIG / "02_revenue_by_category.png", dpi=150); plt.close(fig)

    # order-level table for tests
    orders = (lines.groupby(["order_id", "channel", "region"], dropna=False)
                   .agg(order_value=("revenue", "sum")).reset_index())
    orders.to_csv(TAB / "order_level.csv", index=False)

    # Welch t-test: Online vs Store order value
    a = orders.loc[orders.channel == "Online", "order_value"]
    b = orders.loc[orders.channel == "Store", "order_value"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    out["ttest_online_vs_store_order_value"] = {
        "online_mean": round(a.mean(), 2), "store_mean": round(b.mean(), 2),
        "t": round(t, 3), "p": float(p)}

    # One-way ANOVA across regions (+ non-parametric check, order values are skewed)
    groups = [g["order_value"].values for _, g in orders.dropna(subset=["region"]).groupby("region")]
    f, p = stats.f_oneway(*groups)
    h, p_kw = stats.kruskal(*groups)
    out["anova_order_value_by_region"] = {"F": round(f, 3), "p": float(p),
                                          "kruskal_H": round(h, 3), "kruskal_p": float(p_kw)}

    # Correlation of customer metrics
    return out, orders


def customer_eda(feats):
    cols = ["recency_days", "tenure_days", "frequency", "monetary", "profit", "units", "age"]
    corr = feats[cols].corr()
    corr.to_csv(TAB / "customer_correlations.csv")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Customer metric correlations")
    fig.tight_layout(); fig.savefig(FIG / "03_correlations.png", dpi=150); plt.close(fig)
    feats[cols].describe().round(2).to_csv(TAB / "customer_describe.csv")


# ----------------------------------------------------------------------------
# 4. Customer segmentation (RFM + KMeans)
# ----------------------------------------------------------------------------
def segmentation(feats):
    X = feats[["recency_days", "frequency", "monetary"]].copy()
    X["monetary"] = np.log1p(X["monetary"])
    X["frequency"] = np.log1p(X["frequency"])
    Xs = StandardScaler().fit_transform(X)

    scores = {}
    for k in range(2, 8):
        labels = KMeans(k, n_init=10, random_state=SEED).fit_predict(Xs)
        scores[k] = silhouette_score(Xs, labels)
    best_k = 4  # 4 segments chosen for business actionability; silhouette scores are saved for reference
    km = KMeans(best_k, n_init=10, random_state=SEED).fit(Xs)
    feats = feats.assign(segment=km.labels_)

    prof = (feats.groupby("segment")
                 .agg(customers=("customer_id", "count"),
                      avg_recency_days=("recency_days", "mean"),
                      avg_frequency=("frequency", "mean"),
                      avg_monetary=("monetary", "mean"),
                      total_revenue=("monetary", "sum"))
                 .round(1))
    prof["revenue_share_pct"] = (prof["total_revenue"] / prof["total_revenue"].sum() * 100).round(1)
    # readable names: rank on value, then split the middle two by recency
    order = prof["avg_monetary"].sort_values(ascending=False).index.tolist()
    mid = sorted(order[1:-1], key=lambda s: prof.loc[s, "avg_recency_days"])
    names = {order[0]: "Champions", order[-1]: "Low value",
             mid[0]: "Loyal / Growing", mid[-1]: "At risk"}
    prof["label"] = [names[s] for s in prof.index]
    prof.to_csv(TAB / "segment_profile.csv")
    pd.Series(scores, name="silhouette").round(3).to_csv(TAB / "silhouette_by_k.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    lab = feats["segment"].map(prof["label"])
    sns.scatterplot(x=feats["frequency"], y=feats["monetary"], hue=lab, s=18, ax=ax)
    ax.set(title=f"Customer segments (k={best_k})", xscale="log", yscale="log")
    fig.tight_layout(); fig.savefig(FIG / "04_segments.png", dpi=150); plt.close(fig)
    return feats, prof, scores, best_k


# ----------------------------------------------------------------------------
# 5. Decision tree: who will buy again? (time-based split, no leakage)
# ----------------------------------------------------------------------------
def decision_tree(lines, cust_clean):
    cutoff = pd.Timestamp("2024-07-01")
    past = lines[lines.order_date < cutoff]
    future = lines[lines.order_date >= cutoff]

    g = past.groupby("customer_id")
    X = pd.DataFrame({
        "recency_days": (cutoff - g["order_date"].max()).dt.days,
        "frequency": g["order_id"].nunique(),
        "monetary": g["revenue"].sum(),
        "avg_margin_pct": g["profit"].sum() / g["revenue"].sum() * 100,
        "categories_bought": g["category"].nunique(),
    }).join(cust_clean.set_index("customer_id")[["age"]])
    y = X.index.isin(future["customer_id"].unique()).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=40, random_state=SEED,
                                  class_weight="balanced").fit(Xtr, ytr)
    proba = tree.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)
    rep = classification_report(yte, tree.predict(Xte), output_dict=True)
    imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.round(3).to_csv(TAB / "tree_feature_importance.csv", header=["importance"])
    (TAB / "tree_rules.txt").write_text(export_text(tree, feature_names=list(X.columns)))

    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(tree, feature_names=X.columns, class_names=["No repeat", "Repeat"],
              filled=True, rounded=True, fontsize=8, ax=ax)
    fig.tight_layout(); fig.savefig(FIG / "05_decision_tree.png", dpi=150); plt.close(fig)
    return {"auc": round(auc, 3), "base_rate_repeat": round(float(y.mean()), 3),
            "accuracy": round(rep["accuracy"], 3),
            "top_drivers": imp.head(3).round(3).to_dict()}


# ----------------------------------------------------------------------------
# 6. Forecasting monthly revenue (trend + seasonality regression)
# ----------------------------------------------------------------------------
def design(months):
    t = np.arange(len(months))
    M = pd.get_dummies(pd.to_datetime(months).month, prefix="m", drop_first=True).astype(float)
    return np.column_stack([t, M.values]), M.columns


def forecast(kpis, horizon=6, holdout=6):
    m = kpis["monthly_revenue"].copy()
    m = m[m["month"] >= "2023-03"].reset_index(drop=True)  # drop ramp-up months
    months = m["month"].tolist()
    X, _ = design(months)
    y = m["revenue"].values

    # holdout backtest vs seasonal-naive baseline
    Xtr, Xte, ytr, yte = X[:-holdout], X[-holdout:], y[:-holdout], y[-holdout:]
    lr = LinearRegression().fit(Xtr, ytr)
    mape_model = mean_absolute_percentage_error(yte, lr.predict(Xte))
    naive = np.repeat(ytr[-1], holdout)  # baseline: carry last observed month forward
    mape_naive = mean_absolute_percentage_error(yte, naive)

    # refit on all data, project forward
    lr = LinearRegression().fit(X, y)
    future_months = pd.period_range(months[-1], periods=horizon + 1, freq="M")[1:].astype(str).tolist()
    Xf, _ = design(months + future_months)
    # align dummy columns
    pred = lr.predict(np.column_stack([Xf[:, 0], Xf[:, 1:]])[-horizon:]) if Xf.shape[1] == X.shape[1] else None
    if pred is None:
        Xfull = pd.get_dummies(pd.to_datetime(months + future_months).month, prefix="m").astype(float)
        Xfull = Xfull.reindex(columns=[f"m_{i}" for i in range(2, 13)], fill_value=0.0)
        Xf = np.column_stack([np.arange(len(months) + horizon), Xfull.values])
        pred = lr.predict(Xf[-horizon:])
    fc = pd.DataFrame({"month": future_months, "forecast_revenue": pred.round(2)})
    fc.to_csv(TAB / "revenue_forecast.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pd.to_datetime(months), y, label="Actual", marker="o")
    ax.plot(pd.to_datetime(future_months), pred, label="Forecast", marker="o", linestyle="--")
    ax.set(title="Monthly revenue forecast (trend + seasonality)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "06_forecast.png", dpi=150); plt.close(fig)
    return {"holdout_months": holdout, "mape_model_pct": round(mape_model * 100, 1),
            "mape_last_value_baseline_pct": round(mape_naive * 100, 1)}


# ----------------------------------------------------------------------------
def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    checks, cust_clean = validate_and_clean(con)
    kpis, lines, feats = extract(con)
    tests, _ = eda(kpis, lines)
    customer_eda(feats)
    feats, prof, scores, k = segmentation(feats)
    tree = decision_tree(lines, cust_clean)
    fc = forecast(kpis)
    con.close()

    summary = {"data_quality_issues": checks, "kpis": kpis["kpi_overall"].iloc[0].to_dict(),
               "statistical_tests": tests, "segmentation_k": k,
               "segment_profile": prof.reset_index().to_dict("records"),
               "decision_tree": tree, "forecast_backtest": fc}
    (ROOT / "outputs" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
