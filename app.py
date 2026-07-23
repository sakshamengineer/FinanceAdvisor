import os
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import io

st.set_page_config(page_title="Personal Finance Analyzer", page_icon="💰", layout="wide")

st.markdown("""
<style>
    .hero {
        background: linear-gradient(135deg, #0F9D58 0%, #00C9A7 100%);
        padding: 26px 32px;
        border-radius: 16px;
        margin-bottom: 20px;
    }
    .hero h1 { color: white; margin: 0; font-size: 28px; font-weight: 800; }
    .hero p { color: rgba(255,255,255,0.9); margin: 6px 0 0 0; }
    .alert-card {
        background: rgba(255, 90, 90, 0.12);
        border: 1px solid rgba(255, 90, 90, 0.4);
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 8px;
    }
    .mode-banner {
        background: rgba(0, 201, 167, 0.12);
        border: 1px solid rgba(0, 201, 167, 0.4);
        border-radius: 10px;
        padding: 8px 16px;
        margin-bottom: 12px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>💰 Personal Finance Analyzer</h1>
    <p>Spending categorization · trend analysis · expense forecasting · budget recommendations</p>
</div>
""", unsafe_allow_html=True)

DATA_DIR = "data"
PLOTS_DIR = "eda_plots"

# ---------------------------------------------------------------------------
# Keyword fallback categorizer — used only when an uploaded file has no
# labeled Category rows at all to train a classifier from.
# ---------------------------------------------------------------------------
KEYWORD_MAP = {
    "Food": ["swiggy", "zomato", "restaurant", "cafe", "dominos", "mcdonald", "kfc", "pizza", "food", "chai"],
    "Groceries": ["bigbasket", "dmart", "grocery", "mart", "blinkit", "zepto", "reliance fresh", "kirana"],
    "Transport": ["uber", "ola", "rapido", "irctc", "metro", "fuel", "petrol", "diesel", "cab"],
    "Shopping": ["amazon", "flipkart", "myntra", "ajio", "mall", "shopping"],
    "Bills": ["recharge", "electricity", "water bill", "broadband", "jio", "airtel", "postpaid", "bill"],
    "Entertainment": ["netflix", "spotify", "bookmyshow", "pvr", "cinema", "movie", "steam"],
    "Rent": ["rent"],
    "Health": ["pharmacy", "hospital", "clinic", "medical", "doctor", "apollo", "pharmeasy"],
    "Education": ["udemy", "coursera", "course", "tuition", "college", "school"],
    "Investment": ["zerodha", "groww", "mutual fund", "sip", "stock", "invest", "ppf"],
    "Income": ["salary", "refund", "cashback", "interest credit"],
}


def keyword_categorize(description: str) -> str:
    desc = str(description).lower()
    for category, keywords in KEYWORD_MAP.items():
        if any(kw in desc for kw in keywords):
            return category
    return "Uncategorized"


def clean_amount_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = re.sub(r"[₹$,\s]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return None


@st.cache_data(show_spinner="Cleaning and categorizing your data...")
def process_uploaded_csv(raw_bytes: bytes):
    """Clean + categorize an uploaded CSV. Expects columns roughly matching:
    Date, Description, Amount, Type, Category (optional), PaymentMode (optional)."""
    df = pd.read_csv(io.BytesIO(raw_bytes))

    # Be lenient about column naming — try to match common variants
    col_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc in ("date", "transaction date", "txn date"):
            col_map[col] = "Date"
        elif lc in ("description", "narration", "merchant", "particulars"):
            col_map[col] = "Description"
        elif lc in ("amount", "amt", "transaction amount"):
            col_map[col] = "Amount"
        elif lc in ("type", "txn type", "debit/credit"):
            col_map[col] = "Type"
        elif lc in ("category",):
            col_map[col] = "Category"
        elif lc in ("paymentmode", "payment mode", "mode"):
            col_map[col] = "PaymentMode"
    df = df.rename(columns=col_map)

    required = ["Date", "Description", "Amount"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return None, f"Missing required column(s): {missing_cols}. Found columns: {list(df.columns)}"

    # Clean amount
    df["Amount"] = df["Amount"].apply(clean_amount_value)
    df = df.dropna(subset=["Amount"])

    # Parse date leniently — real-world exports are usually one consistent
    # format (unlike our intentionally-messy synthetic set), so a single
    # flexible parse pass is fine here.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"])

    # Infer Type if missing, from amount sign or a Credit/Debit hint
    if "Type" not in df.columns:
        df["Type"] = df["Amount"].apply(lambda x: "Credit" if x > 0 else "Debit")
    df["Amount"] = df["Amount"].abs()

    if "PaymentMode" not in df.columns:
        df["PaymentMode"] = "Unknown"

    if "Category" not in df.columns:
        df["Category"] = None
    else:
        df["Category"] = df["Category"].apply(lambda v: str(v).strip().title() if pd.notna(v) else None)

    df["Description"] = df["Description"].astype(str).str.strip()
    df = df.drop_duplicates().sort_values("Date").reset_index(drop=True)
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)

    # Categorize missing rows
    labeled = df[df["Category"].notna()]
    unlabeled = df[df["Category"].isna()]
    method = None

    if len(unlabeled) > 0:
        if labeled["Category"].nunique() >= 2 and len(labeled) >= 10:
            # Enough labeled data to train a real classifier
            pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
                ("clf", LinearSVC(random_state=42)),
            ])
            desc_clean = labeled["Description"].str.lower().str.replace("*", " ", regex=False)
            pipeline.fit(desc_clean, labeled["Category"])
            unlabeled_desc = unlabeled["Description"].str.lower().str.replace("*", " ", regex=False)
            df.loc[df["Category"].isna(), "Category"] = pipeline.predict(unlabeled_desc)
            method = "TF-IDF + Linear SVM (trained on your labeled rows)"
        else:
            # Not enough labeled examples — fall back to keyword matching
            df.loc[df["Category"].isna(), "Category"] = unlabeled["Description"].apply(keyword_categorize)
            method = "Keyword matching (not enough labeled rows to train a classifier)"

    return df, method


def compute_budget_recommendations(df: pd.DataFrame, overspend_threshold: float = 1.3) -> pd.DataFrame:
    """Lightweight inline version of budget.py's logic, for live use on uploaded data."""
    debit = df[df["Type"] == "Debit"]
    pivot = debit.pivot_table(index="Category", columns="YearMonth", values="Amount", aggfunc="sum", fill_value=0)

    if pivot.shape[1] == 0 or pivot.shape[0] == 0:
        return pd.DataFrame()

    features = pd.DataFrame({"mean_spend": pivot.mean(axis=1), "std_spend": pivot.std(axis=1)}).fillna(0)
    k = min(3, len(features))
    if k >= 1:
        scaled = StandardScaler().fit_transform(features)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        features["cluster"] = kmeans.fit_predict(scaled)
        cluster_order = features.groupby("cluster")["mean_spend"].mean().sort_values(ascending=False).index
        tier_names = ["Major Expense", "Moderate Expense", "Minor/Discretionary"]
        tier_map = {cid: tier_names[i] if i < len(tier_names) else f"Tier {i}" for i, cid in enumerate(cluster_order)}
        features["tier"] = features["cluster"].map(tier_map)
    else:
        features["tier"] = "Moderate Expense"

    months = sorted(pivot.columns)
    latest_month = months[-1]
    history_months = months[:-1] if len(months) > 1 else months

    rows = []
    for category in pivot.index:
        historical_avg = pivot.loc[category, history_months].mean()
        latest_spend = pivot.loc[category, latest_month]
        is_overspending = historical_avg > 0 and latest_spend > historical_avg * overspend_threshold
        tier = features.loc[category, "tier"]
        buffer_mult = {"Major Expense": 1.05, "Moderate Expense": 1.0, "Minor/Discretionary": 0.9}
        recommended = historical_avg * buffer_mult.get(tier, 1.0)
        rows.append({
            "Category": category,
            "Tier": tier,
            "HistoricalAvgMonthly": round(historical_avg, 2),
            "LatestMonthSpend": round(latest_spend, 2),
            "RecommendedBudget": round(recommended, 2),
            "OverspendingAlert": is_overspending,
            "PctChangeVsAvg": round(((latest_spend - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0, 1),
        })
    return pd.DataFrame(rows).sort_values("HistoricalAvgMonthly", ascending=False)


# ---------------------------------------------------------------------------
# Sidebar: data source selector (default dataset vs. live upload)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Data Source")
    uploaded_file = st.file_uploader("Upload your own transactions CSV", type=["csv"])
    st.caption("Expected columns: Date, Description, Amount, Type (optional), Category (optional), PaymentMode (optional)")

using_upload = uploaded_file is not None
categorization_method = None

if using_upload:
    df, result = process_uploaded_csv(uploaded_file.getvalue())
    if df is None:
        st.error(result)
        st.stop()
    categorization_method = result
    budget_df = compute_budget_recommendations(df)
    st.markdown(f"""<div class="mode-banner">🔬 <b>Live mode:</b> showing results for your uploaded file
    ({len(df)} transactions). Forecast tab still shows the default dataset's pre-trained models.</div>""",
                unsafe_allow_html=True)
else:
    REQUIRED_FILES = {
        "categorized": f"{DATA_DIR}/transactions_categorized.csv",
        "budget": f"{DATA_DIR}/budget_recommendations.csv",
    }
    missing = [name for name, path in REQUIRED_FILES.items() if not os.path.exists(path)]
    if missing:
        st.error(f"Missing required file(s): {missing}")
        st.info(
            "Run the pipeline first, in order:\n\n"
            "```\npython generate_data.py\npython clean_data.py\npython categorize.py\n"
            "python forecast.py\npython budget.py\n```"
        )
        st.stop()
    df = pd.read_csv(REQUIRED_FILES["categorized"], parse_dates=["Date"])
    budget_df = pd.read_csv(REQUIRED_FILES["budget"])

# ---------------------------------------------------------------------------
# Top-level metrics
# ---------------------------------------------------------------------------
total_spend = df[df["Type"] == "Debit"]["Amount"].sum()
total_income = df[df["Type"] == "Credit"]["Amount"].sum()
n_months = df["YearMonth"].nunique()
avg_monthly_spend = total_spend / n_months if n_months else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Spend", f"₹{total_spend:,.0f}")
col2.metric("Total Income", f"₹{total_income:,.0f}")
col3.metric("Avg Monthly Spend", f"₹{avg_monthly_spend:,.0f}")
col4.metric("Categories Predicted", int(df["CategoryPredicted"].sum()) if "CategoryPredicted" in df.columns else 0)

st.divider()

# ---------------------------------------------------------------------------
# Tabs for each section of the pipeline
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview & Trends", "🏷️ Categorization", "📈 Forecast", "💡 Budget"])

with tab1:
    st.subheader("Monthly Spending Trend")
    monthly = df[df["Type"] == "Debit"].groupby("YearMonth")["Amount"].sum().reset_index()
    fig = px.line(monthly, x="YearMonth", y="Amount", markers=True,
                  color_discrete_sequence=["#0F9D58"])
    fig.update_layout(yaxis_title="Total Spend (₹)", xaxis_title="Month")
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Spend by Category")
        cat_spend = df[df["Type"] == "Debit"].groupby("Category")["Amount"].sum().sort_values(ascending=False)
        fig2 = px.bar(x=cat_spend.values, y=cat_spend.index, orientation="h",
                      color_discrete_sequence=["#00C9A7"])
        fig2.update_layout(xaxis_title="Total Spend (₹)", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.subheader("Category Share")
        fig3 = px.pie(values=cat_spend.values, names=cat_spend.index, hole=0.4)
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Raw Transactions")
    st.dataframe(df.sort_values("Date", ascending=False), use_container_width=True, height=300)

with tab2:
    st.subheader("Category Assignment")
    if using_upload:
        st.write(f"**Method used:** {categorization_method}")
    else:
        st.write(
            "Categories were assigned via a TF-IDF + Linear SVM text classifier trained on "
            "transaction descriptions. Rows below were originally missing a category and "
            "filled in automatically by the model."
        )
    if "CategoryPredicted" in df.columns:
        predicted_rows = df[df["CategoryPredicted"] == True]
        st.write(f"**{len(predicted_rows)} categories were predicted by the model** (originally missing)")
        st.dataframe(predicted_rows[["Date", "Description", "Category", "Amount"]], use_container_width=True)

    if not using_upload:
        cm_path = f"{PLOTS_DIR}/categorization_confusion_matrix.png"
        if os.path.exists(cm_path):
            st.subheader("Model Evaluation — Confusion Matrix")
            try:
                st.image(cm_path, use_container_width=True)
            except TypeError:
                st.image(cm_path)

with tab3:
    st.subheader("ARIMA vs LSTM — Expense Forecasting")
    if using_upload:
        st.info(
            "Forecasting always shows the default dataset's pre-trained models — "
            "retraining ARIMA/LSTM live on every upload would be slow and needs extra "
            "dependencies (torch, statsmodels) this app deliberately keeps out of the "
            "live/deployed version. Switch back to the default dataset (remove the "
            "uploaded file) to see forecasting, or run `forecast.py` locally on your "
            "own data to regenerate it."
        )
    st.write(
        "Two models were compared for predicting daily spending: ARIMA (classical "
        "statistical model) and LSTM (recurrent neural network). See the report/README "
        "for why one may outperform the other given the amount of data available."
    )
    forecast_path = f"{PLOTS_DIR}/forecast_comparison.png"
    if os.path.exists(forecast_path):
        try:
            st.image(forecast_path, use_container_width=True)
        except TypeError:
            st.image(forecast_path)
    else:
        st.warning("Run `python forecast.py` to generate the forecast comparison plot.")

    next_month_path = f"{DATA_DIR}/next_month_prediction.csv"
    if os.path.exists(next_month_path):
        pred = pd.read_csv(next_month_path, index_col=0)
        predicted_value = pred.iloc[0, 0]
        st.metric("Predicted Next 30-Day Spend", f"₹{predicted_value:,.0f}")

with tab4:
    st.subheader("Budget Recommendations")
    st.write(
        "Categories are grouped into spend tiers using K-Means clustering (on average "
        "spend + volatility), then compared against historical averages to flag overspending."
    )

    if budget_df.empty:
        st.warning("Not enough data to generate budget recommendations (need at least one full month of transactions).")
    else:
        overspending = budget_df[budget_df["OverspendingAlert"] == True]
        if len(overspending) > 0:
            st.markdown("**⚠️ Overspending Alerts:**")
            for _, row in overspending.iterrows():
                st.markdown(f"""
                <div class="alert-card">
                    <b>{row['Category']}</b>: ₹{row['LatestMonthSpend']:,.0f} this month vs
                    ₹{row['HistoricalAvgMonthly']:,.0f} average ({row['PctChangeVsAvg']:+.1f}%)
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No categories are currently over the overspending threshold.")

        st.divider()
        st.dataframe(budget_df, use_container_width=True)

        fig4 = px.bar(budget_df.sort_values("RecommendedBudget", ascending=True),
                      x="RecommendedBudget", y="Category", orientation="h",
                      color="Tier", title="Recommended Monthly Budget by Category")
        st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.caption("Pipeline: Generate → Clean → Categorize (SVM) → Forecast (ARIMA + LSTM) → Budget (K-Means + rules)")
