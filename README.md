# Personal Finance Analyzer

A pipeline that ingests transaction data, cleans it, categorizes spending
via a text classifier, forecasts future expenses (ARIMA vs LSTM), and
generates budget recommendations via K-Means clustering — all viewable in
a Streamlit dashboard.



```
Finance_Project/
├── data/                            =
│   └── (CSV files land here automatically when scripts run)
├── eda_plots/                        
│   └── (PNG charts land here automatically when scripts run)
├── generate_data.py
├── clean_data.py
├── categorize.py
├── eda.py
├── forecast.py
├── budget.py
├── app.py
└── requirements.txt
```



## Run order 

```bash
# 1. One-time setup
pip install -r requirements.txt

# 2. Generate the synthetic dataset (raw + intentionally messy)
python generate_data.py
# -> creates data/transactions.csv

# 3. Clean it (fixes dates, currency symbols, duplicates, sign errors)
python clean_data.py
# -> creates data/transactions_clean.csv

# 4. Fill missing categories using a trained text classifier
python categorize.py
# -> creates data/transactions_categorized.csv
# -> creates eda_plots/categorization_confusion_matrix.png

# 5. Run EDA (summary stats + 6 charts)
python eda.py
# -> creates eda_plots/*.png (trend, category breakdown, pie, boxplot, heatmap, payment mode)

# 6. Run forecasting comparison (ARIMA vs LSTM) — this one takes longest
python forecast.py
# -> creates eda_plots/forecast_comparison.png
# -> creates data/next_month_prediction.csv

# 7. Generate budget recommendations (K-Means + rule-based)
python budget.py
# -> creates data/budget_recommendations.csv

# 8. Launch the dashboard
streamlit run app.py
```

Steps 2–7 only need to be run once 
After that, you only need `streamlit run app.py` to view the dashboard.

## What each file actually does

| File | Role | Model/Technique |
|---|---|---|
| `generate_data.py` | Creates synthetic messy transaction data | — |
| `clean_data.py` | Fixes dates, currency symbols, duplicates, sign errors | Rule-based cleaning |
| `categorize.py` | Fills missing spending categories from description text | TF-IDF + Linear SVM |
| `eda.py` | Summary stats + visualizations | matplotlib/seaborn |
| `forecast.py` | Predicts future daily spending, compares two approaches | ARIMA (statsmodels) vs LSTM (PyTorch) |
| `budget.py` | Groups categories into spend tiers, flags overspending | K-Means clustering + rules |
| `app.py` | Interactive dashboard tying everything together | Streamlit + Plotly |

This mapping — classification, statistical forecasting, deep learning
forecasting, and unsupervised clustering — is what covers "multiple model
integration" across the project, not just one algorithm reused four times.

## Deployment (GitHub + Streamlit Cloud)

Same process as your other project:
1. Push the whole `Finance_Project` folder to a GitHub repo — **including**
   the generated `data/` and `eda_plots/` contents (commit the CSVs and
   PNGs too, so the dashboard has something to read immediately on deploy —)

2. share.streamlit.io → connect the repo → point at `app.py`



## Known limitation

- The categorization classifier hits ~100% accuracy on this synthetic data
  because each merchant maps to exactly one category with no ambiguity —
  real-world data would be messier (e.g. "Amazon" spanning multiple
  categories) and accuracy would likely be lower.
- With ~10 months of data, ARIMA often outperforms LSTM — this is expected
  and worth discussing as a finding, not something to "fix" by tuning the
  LSTM further.
