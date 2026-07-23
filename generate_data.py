"""
generate_data.py
-----------------
Generates a synthetic personal finance transaction dataset that mimics a
real UPI/bank statement export — INCLUDING realistic messiness so the
cleaning/EDA section of the project has genuine work to show, not just
a already-clean CSV.

Deliberate quirks baked in (documented here so you can cite them in your
report's "Data Cleaning" section):
  1. Inconsistent date formats (3 different formats mixed across rows)
  2. Missing values in Category (~12%) and a few in Amount (~2%)
  3. Duplicate transactions (~20 exact dupes, simulating double-entries)
  4. Inconsistent merchant name casing/spacing ("SWIGGY", "swiggy  ", "Swiggy*Order")
  5. A handful of outlier transactions (e.g. one-time big purchases)
  6. Currency symbol inconsistency (some amounts have "₹" prefix, some don't)
  7. A few negative amounts entered by mistake for debit transactions
  8. Whitespace / mixed-case in the Category column

Run with:
    python generate_data.py
Output: data/transactions.csv
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

N_MONTHS = 10
START_DATE = datetime(2025, 9, 1)
TRANSACTIONS_PER_MONTH = (60, 90)  # random range

# Merchant pools per category, with realistic Indian brand names.
# Multiple casing/spacing variants included ON PURPOSE for the messiness.
MERCHANTS = {
    "Food": ["Swiggy", "SWIGGY", "swiggy  ", "Zomato", "Zomato*Order", "Dominos", "McDonald's", "Chai Point"],
    "Groceries": ["BigBasket", "DMart", "Blinkit", "Zepto", "Reliance Fresh", "Local Kirana Store"],
    "Transport": ["Uber", "UBER", "Ola Cabs", "ola", "Rapido", "IRCTC", "Metro Card Recharge"],
    "Shopping": ["Amazon", "AMAZON.IN", "Flipkart", "Myntra", "Ajio", "Local Market"],
    "Bills": ["Jio Recharge", "Airtel Postpaid", "Electricity Board", "Water Bill", "Broadband - ACT"],
    "Entertainment": ["Netflix", "Spotify", "BookMyShow", "PVR Cinemas", "Steam Games"],
    "Rent": ["Rent Payment - Landlord"],
    "Health": ["Apollo Pharmacy", "PharmEasy", "Practo Consultation", "Local Clinic"],
    "Education": ["Udemy", "Coursera", "College Fee Payment", "Book Store"],
    "Investment": ["Zerodha", "Groww SIP", "PPF Deposit"],
    "Income": ["Salary Credit - Employer", "Freelance Payment", "Family Transfer"],
}

# Typical amount ranges per category (min, max) in INR
AMOUNT_RANGES = {
    "Food": (80, 600),
    "Groceries": (200, 2500),
    "Transport": (30, 500),
    "Shopping": (300, 5000),
    "Bills": (200, 2000),
    "Entertainment": (99, 800),
    "Rent": (8000, 8000),
    "Health": (100, 3000),
    "Education": (300, 5000),
    "Investment": (1000, 10000),
    "Income": (15000, 35000),
}

DATE_FORMATS = ["%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%d"]


def random_date_in_month(year, month):
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(days=1)
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def generate_transactions():
    rows = []
    current = START_DATE

    for _m in range(N_MONTHS):
        year, month = current.year, current.month
        n_txns = random.randint(*TRANSACTIONS_PER_MONTH)

        for _ in range(n_txns):
            category = random.choices(
                population=list(MERCHANTS.keys()),
                weights=[14, 10, 12, 10, 8, 6, 1, 5, 3, 3, 1],  # Income/Rent rarer per-txn
                k=1,
            )[0]

            merchant = random.choice(MERCHANTS[category])
            date = random_date_in_month(year, month)
            lo, hi = AMOUNT_RANGES[category]
            amount = round(random.uniform(lo, hi), 2) if lo != hi else float(lo)

            txn_type = "Credit" if category == "Income" else "Debit"
            payment_mode = random.choice(["UPI", "UPI", "UPI", "Debit Card", "Credit Card", "NetBanking"])

            rows.append({
                "Date": date,
                "Description": merchant,
                "Amount": amount,
                "Type": txn_type,
                "Category": category,
                "PaymentMode": payment_mode,
            })

        # Guaranteed monthly fixed transactions (salary + rent) for realism
        rows.append({
            "Date": datetime(year, month, random.choice([1, 2])),
            "Description": "Salary Credit - Employer",
            "Amount": round(random.uniform(22000, 32000), 2),
            "Type": "Credit",
            "Category": "Income",
            "PaymentMode": "NetBanking",
        })
        rows.append({
            "Date": datetime(year, month, random.choice([3, 4, 5])),
            "Description": "Rent Payment - Landlord",
            "Amount": 8000.0,
            "Type": "Debit",
            "Category": "Rent",
            "PaymentMode": "UPI",
        })

        # advance month
        if month == 12:
            current = datetime(year + 1, 1, 1)
        else:
            current = datetime(year, month + 1, 1)

    df = pd.DataFrame(rows)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def inject_messiness(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # 1. Mixed date formats (stored as strings, not datetime, to force parsing work)
    def format_date(d):
        fmt = random.choice(DATE_FORMATS)
        return d.strftime(fmt)

    df["Date"] = df["Date"].apply(format_date)

    # 2. Missing categories (~12%)
    missing_cat_idx = np.random.choice(n, size=int(n * 0.12), replace=False)
    df.loc[missing_cat_idx, "Category"] = np.nan

    # 3. A few missing amounts (~2%)
    missing_amt_idx = np.random.choice(n, size=int(n * 0.02), replace=False)
    df.loc[missing_amt_idx, "Amount"] = np.nan

    # 4. Duplicate rows (~20 exact duplicates appended)
    dupes = df.sample(n=20, random_state=1)
    df = pd.concat([df, dupes], ignore_index=True)

    # 5. Outlier transactions (a handful of unusually large one-time purchases)
    outlier_idx = np.random.choice(df.index, size=6, replace=False)
    df.loc[outlier_idx, "Amount"] = df.loc[outlier_idx, "Amount"] * random.uniform(15, 30)
    df.loc[outlier_idx, "Description"] = "Electronics Store - Laptop Purchase"
    df.loc[outlier_idx, "Category"] = "Shopping"

    # 6. Currency symbol inconsistency on ~15% of rows
    currency_idx = np.random.choice(df.index, size=int(len(df) * 0.15), replace=False)
    df["Amount"] = df["Amount"].astype(object)
    for idx in currency_idx:
        val = df.loc[idx, "Amount"]
        if pd.notna(val):
            df.loc[idx, "Amount"] = f"₹{val}"

    # 7. A few negative amounts entered by mistake for Debit transactions
    neg_idx = np.random.choice(
        df[df["Type"] == "Debit"].index, size=8, replace=False
    )
    for idx in neg_idx:
        val = df.loc[idx, "Amount"]
        if pd.notna(val) and isinstance(val, (int, float)):
            df.loc[idx, "Amount"] = -abs(val)

    # 8. Whitespace / case inconsistency in Category
    case_idx = np.random.choice(df.index, size=int(len(df) * 0.1), replace=False)
    for idx in case_idx:
        cat = df.loc[idx, "Category"]
        if pd.notna(cat):
            variant = random.choice([cat.upper(), cat.lower(), f"  {cat}  ", f"{cat} "])
            df.loc[idx, "Category"] = variant

    # Shuffle rows so duplicates aren't conveniently adjacent, then reset index
    df = df.sample(frac=1, random_state=2).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Generating base transaction data...")
    df = generate_transactions()
    print(f"Base transactions: {len(df)}")

    print("Injecting realistic messiness (missing values, dupes, format issues)...")
    df = inject_messiness(df)
    print(f"Final row count (with duplicates): {len(df)}")

    df.to_csv("data/transactions.csv", index=False)
    print("Saved to data/transactions.csv")
    print("\nPreview:")
    print(df.head(10).to_string())
    print(f"\nMissing values:\n{df.isna().sum()}")
