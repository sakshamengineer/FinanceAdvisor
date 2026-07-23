"""
clean_data.py
--------------
Cleans the raw synthetic transaction data (data/transactions.csv) and
produces data/transactions_clean.csv.

This directly addresses each messiness issue baked into generate_data.py:
  1. Mixed date formats        -> parsed into a single datetime dtype
  2. Currency symbol (₹) noise -> stripped, converted to float
  3. Negative amounts on Debit -> sign corrected (abs value)
  4. Exact duplicate rows      -> dropped
  5. Category casing/whitespace -> normalised (strip + title case)
  6. Missing Amount            -> dropped (can't reliably impute a transaction value)
  7. Missing Category          -> left as NaN here on purpose; filled later
     by categorize.py using a text classifier on the Description column
     (this is intentional — imputing category from raw text is itself
     a modelling step, not just cleaning, so it's kept separate)

Run with:
    python clean_data.py
"""

import re

import pandas as pd

RAW_PATH = "data/transactions.csv"
CLEAN_PATH = "data/transactions_clean.csv"


def parse_mixed_dates(date_series: pd.Series) -> pd.Series:
    """
    The raw data mixes three date formats across rows:
    %d-%m-%Y, %m/%d/%Y, %Y-%m-%d

    IMPORTANT: pandas' format='mixed' + dayfirst inference is NOT reliable
    here — it guesses per-row and silently mis-parses ambiguous strings
    like "03/19/2026" vs "07-06-2026", scrambling months/years. Since our
    three formats are structurally distinguishable (separator + digit
    count of the first group), we parse deterministically instead:
      - contains '/'                          -> %m/%d/%Y
      - contains '-' and starts with 4 digits  -> %Y-%m-%d
      - contains '-' and starts with 2 digits  -> %d-%m-%Y
    """
    def parse_one(val):
        if pd.isna(val):
            return pd.NaT
        s = str(val).strip()
        try:
            if "/" in s:
                return pd.to_datetime(s, format="%m/%d/%Y")
            first_group = s.split("-")[0]
            if len(first_group) == 4:
                return pd.to_datetime(s, format="%Y-%m-%d")
            return pd.to_datetime(s, format="%d-%m-%Y")
        except (ValueError, TypeError):
            return pd.NaT

    return date_series.apply(parse_one)


def clean_amount(amount_series: pd.Series) -> pd.Series:
    """Strip currency symbols, coerce to float, and fix sign errors later."""
    def to_float(val):
        if pd.isna(val):
            return None
        if isinstance(val, (int, float)):
            return float(val)
        # strip ₹ symbol and any stray whitespace
        cleaned = re.sub(r"[₹,\s]", "", str(val))
        try:
            return float(cleaned)
        except ValueError:
            return None
    return amount_series.apply(to_float)


def clean_category(category_series: pd.Series) -> pd.Series:
    """Normalise whitespace and casing: '  FOOD ' / 'food' -> 'Food'."""
    def normalise(val):
        if pd.isna(val):
            return None
        return str(val).strip().title()
    return category_series.apply(normalise)


def clean():
    print(f"Loading raw data from {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)
    initial_rows = len(df)
    print(f"Initial rows: {initial_rows}")

    # --- 1. Parse dates ---
    df["Date"] = parse_mixed_dates(df["Date"])
    unparseable_dates = df["Date"].isna().sum()
    if unparseable_dates:
        print(f"  Dropping {unparseable_dates} rows with unparseable dates")
        df = df.dropna(subset=["Date"])

    # --- 2. Clean amount (strip currency symbols, coerce to numeric) ---
    df["Amount"] = clean_amount(df["Amount"])

    # --- 3. Drop rows with missing amount (can't impute reliably) ---
    missing_amount = df["Amount"].isna().sum()
    print(f"  Dropping {missing_amount} rows with missing Amount")
    df = df.dropna(subset=["Amount"])

    # --- 4. Fix sign errors: Debit transactions should always be positive
    #        magnitude (sign mistakes were injected on purpose) ---
    debit_mask = df["Type"] == "Debit"
    negative_debits = (df.loc[debit_mask, "Amount"] < 0).sum()
    print(f"  Correcting sign on {negative_debits} negative Debit amounts")
    df.loc[debit_mask, "Amount"] = df.loc[debit_mask, "Amount"].abs()

    # --- 5. Normalise category text ---
    df["Category"] = clean_category(df["Category"])

    # --- 6. Drop exact duplicate rows ---
    dupes = df.duplicated().sum()
    print(f"  Dropping {dupes} exact duplicate rows")
    df = df.drop_duplicates()

    # --- 7. Normalise description whitespace (kept as its own column
    #        for the categorization model to use as-is; only trim spaces) ---
    df["Description"] = df["Description"].str.strip()

    # --- 8. Sort chronologically and reset index ---
    df = df.sort_values("Date").reset_index(drop=True)

    # --- Add derived columns useful for later EDA/modelling ---
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

    final_rows = len(df)
    print(f"\nFinal rows: {final_rows} (removed {initial_rows - final_rows} total)")
    print(f"Remaining missing Category (to be filled by categorize.py): {df['Category'].isna().sum()}")

    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned data to {CLEAN_PATH}")

    return df


if __name__ == "__main__":
    df = clean()
    print("\nPreview of cleaned data:")
    print(df.head(10).to_string())
    print("\nDtypes:")
    print(df.dtypes)
