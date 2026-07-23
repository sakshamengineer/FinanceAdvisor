"""
budget.py
----------
Generates budget recommendations using two techniques:
  1. K-MEANS CLUSTERING (unsupervised ML) — groups spending categories into
     tiers (e.g. "fixed/essential", "high-variable", "low-discretionary")
     based on their monthly spend patterns (mean + volatility).
  2. RULE-BASED THRESHOLDS — flags categories where the most recent month's
     spend significantly exceeds the historical average, and proposes a
     simple budget cap per category (avg + small buffer).

Why cluster on (mean, std) rather than raw values: two categories can have
similar averages but very different consistency (Rent is fixed every month,
Shopping swings wildly) — clustering on volatility alongside mean spend
groups categories by BEHAVIOR, which is more useful for budgeting advice
than grouping by size alone.

Run with:
    python budget.py
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CATEGORIZED_PATH = "data/transactions_categorized.csv"
OUTPUT_PATH = "data/budget_recommendations.csv"
OVERSPEND_THRESHOLD = 1.3   # flag if latest month > 1.3x historical average


def load_monthly_category_spend():
    df = pd.read_csv(CATEGORIZED_PATH, parse_dates=["Date"])
    debit = df[df["Type"] == "Debit"]
    pivot = debit.pivot_table(
        index="Category", columns="YearMonth", values="Amount", aggfunc="sum", fill_value=0
    )
    return pivot


def cluster_categories(pivot: pd.DataFrame, n_clusters: int = 3):
    """Cluster categories by (mean spend, spend volatility) into tiers."""
    features = pd.DataFrame({
        "mean_spend": pivot.mean(axis=1),
        "std_spend": pivot.std(axis=1),
    })

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    k = min(n_clusters, len(features))  # guard against fewer categories than clusters
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    features["cluster"] = kmeans.fit_predict(scaled)

    # Label clusters meaningfully by their average mean_spend
    # (highest mean -> "Major/Fixed", lowest -> "Minor/Discretionary")
    cluster_order = features.groupby("cluster")["mean_spend"].mean().sort_values(ascending=False).index
    tier_names = ["Major Expense", "Moderate Expense", "Minor/Discretionary"]
    tier_map = {cluster_id: tier_names[i] if i < len(tier_names) else f"Tier {i}"
                for i, cluster_id in enumerate(cluster_order)}
    features["tier"] = features["cluster"].map(tier_map)

    return features.drop(columns=["cluster"])


def generate_recommendations(pivot: pd.DataFrame, cluster_info: pd.DataFrame):
    months = sorted(pivot.columns)
    latest_month = months[-1]
    history_months = months[:-1] if len(months) > 1 else months

    recommendations = []
    for category in pivot.index:
        historical_avg = pivot.loc[category, history_months].mean()
        latest_spend = pivot.loc[category, latest_month]
        std = pivot.loc[category, history_months].std()

        is_overspending = (
            historical_avg > 0 and latest_spend > historical_avg * OVERSPEND_THRESHOLD
        )

        # Simple recommended budget: historical average + a small buffer,
        # scaled down slightly for discretionary categories to nudge saving.
        tier = cluster_info.loc[category, "tier"]
        buffer_multiplier = {"Major Expense": 1.05, "Moderate Expense": 1.0, "Minor/Discretionary": 0.9}
        recommended_budget = historical_avg * buffer_multiplier.get(tier, 1.0)

        recommendations.append({
            "Category": category,
            "Tier": tier,
            "HistoricalAvgMonthly": round(historical_avg, 2),
            "LatestMonthSpend": round(latest_spend, 2),
            "RecommendedBudget": round(recommended_budget, 2),
            "OverspendingAlert": is_overspending,
            "PctChangeVsAvg": round(
                ((latest_spend - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0, 1
            ),
        })

    return pd.DataFrame(recommendations).sort_values("HistoricalAvgMonthly", ascending=False)


if __name__ == "__main__":
    print("Loading monthly category spend...")
    pivot = load_monthly_category_spend()
    print(f"Categories: {list(pivot.index)}")
    print(f"Months: {list(pivot.columns)}\n")

    print("Clustering categories by spend behavior (K-Means)...")
    cluster_info = cluster_categories(pivot)
    print(cluster_info.to_string())

    print("\nGenerating budget recommendations...")
    recs = generate_recommendations(pivot, cluster_info)
    recs.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved recommendations to {OUTPUT_PATH}\n")
    print(recs.to_string(index=False))

    overspending = recs[recs["OverspendingAlert"]]
    if len(overspending) > 0:
        print("\n⚠️  OVERSPENDING ALERTS:")
        for _, row in overspending.iterrows():
            print(f"  {row['Category']}: Rs.{row['LatestMonthSpend']:,.0f} vs avg Rs.{row['HistoricalAvgMonthly']:,.0f} "
                  f"({row['PctChangeVsAvg']:+.1f}%)")
    else:
        print("\nNo categories currently over the overspending threshold.")
