"""
eda.py
-------
Exploratory Data Analysis on the cleaned transaction data.
Generates summary statistics + saves charts as PNGs to eda_plots/.

Run with:
    python eda.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CLEAN_PATH = "data/transactions_clean.csv"
PLOTS_DIR = "eda_plots"

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


def load_data():
    df = pd.read_csv(CLEAN_PATH, parse_dates=["Date"])
    return df


def print_summary(df):
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total transactions: {len(df)}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Total spend (Debit): Rs. {df[df['Type']=='Debit']['Amount'].sum():,.2f}")
    print(f"Total income (Credit): Rs. {df[df['Type']=='Credit']['Amount'].sum():,.2f}")
    print(f"Missing categories remaining: {df['Category'].isna().sum()}")
    print()
    print("Spend by category:")
    print(df[df["Type"] == "Debit"].groupby("Category")["Amount"].sum().sort_values(ascending=False))
    print()
    print("Amount statistics (Debit only):")
    print(df[df["Type"] == "Debit"]["Amount"].describe())


def plot_monthly_trend(df):
    monthly = (
        df[df["Type"] == "Debit"]
        .groupby("YearMonth")["Amount"]
        .sum()
        .reset_index()
    )
    plt.figure()
    plt.plot(monthly["YearMonth"], monthly["Amount"], marker="o", color="#8E2DE2", linewidth=2)
    plt.xticks(rotation=45)
    plt.title("Monthly Spending Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Spend (Rs.)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/monthly_trend.png", dpi=120)
    plt.close()


def plot_category_breakdown(df):
    cat_spend = (
        df[df["Type"] == "Debit"]
        .groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )
    plt.figure()
    sns.barplot(x=cat_spend.values, y=cat_spend.index, hue=cat_spend.index, palette="viridis", legend=False)
    plt.title("Total Spend by Category")
    plt.xlabel("Total Spend (Rs.)")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/category_breakdown.png", dpi=120)
    plt.close()


def plot_category_pie(df):
    cat_spend = (
        df[df["Type"] == "Debit"]
        .groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )
    plt.figure(figsize=(8, 8))
    plt.pie(cat_spend.values, labels=cat_spend.index, autopct="%1.1f%%", startangle=90,
            colors=sns.color_palette("viridis", len(cat_spend)))
    plt.title("Spending Distribution by Category")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/category_pie.png", dpi=120)
    plt.close()


def plot_amount_boxplot(df):
    """Boxplot to visually spot the outlier transactions we injected."""
    plt.figure()
    sns.boxplot(x=df[df["Type"] == "Debit"]["Amount"], color="#8E2DE2")
    plt.title("Transaction Amount Distribution (Debit) — Outlier Check")
    plt.xlabel("Amount (Rs.)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/amount_boxplot.png", dpi=120)
    plt.close()


def plot_heatmap(df):
    """Category spend across months — good for spotting seasonal patterns."""
    pivot = (
        df[df["Type"] == "Debit"]
        .pivot_table(index="Category", columns="YearMonth", values="Amount", aggfunc="sum", fill_value=0)
    )
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="magma", annot=False, fmt=".0f")
    plt.title("Category Spend Heatmap (Month x Category)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/category_month_heatmap.png", dpi=120)
    plt.close()


def plot_payment_mode(df):
    mode_counts = df["PaymentMode"].value_counts()
    plt.figure()
    sns.barplot(x=mode_counts.index, y=mode_counts.values, hue=mode_counts.index,
                palette="crest", legend=False)
    plt.title("Transaction Count by Payment Mode")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/payment_mode.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = load_data()
    print_summary(df)

    print("\nGenerating plots...")
    plot_monthly_trend(df)
    plot_category_breakdown(df)
    plot_category_pie(df)
    plot_amount_boxplot(df)
    plot_heatmap(df)
    plot_payment_mode(df)
    print(f"Saved 6 plots to {PLOTS_DIR}/")
