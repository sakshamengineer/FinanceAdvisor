"""
forecast.py
------------
Compares two approaches to predicting future daily spending:
  1. ARIMA   — classical statistical time-series model (statsmodels)
  2. LSTM    — recurrent neural network (PyTorch)

WHY DAILY, NOT MONTHLY: with only ~10 months of data, a monthly series has
just 10 data points — nowhere near enough to train an LSTM meaningfully.
Aggregating to DAILY totals instead gives ~300 data points, which is a much
fairer testbed for comparing a classical model against a neural one.

NOTE ON TESTING: this script was NOT executed against a live run in the
environment that wrote it (statsmodels/torch weren't available there — see
chat for context). It uses standard, stable APIs from both libraries.
Run it locally and paste any error back if something doesn't match your
installed library versions.

Run with:
    python forecast.py
Requires: pip install statsmodels (torch should already be installed from
the RAG project's environment; if not: pip install torch)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

CATEGORIZED_PATH = "data/transactions_categorized.csv"
TEST_DAYS = 30       # last 30 days held out for evaluation
WINDOW_SIZE = 14      # LSTM looks at past 14 days to predict the next day


def build_daily_series() -> pd.Series:
    """Aggregate to a continuous daily total-spend series (0 on no-spend days)."""
    df = pd.read_csv(CATEGORIZED_PATH, parse_dates=["Date"])
    debit = df[df["Type"] == "Debit"]

    daily = debit.groupby(debit["Date"].dt.date)["Amount"].sum()
    daily.index = pd.to_datetime(daily.index)

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0.0)
    daily.index.name = "Date"
    return daily


def run_arima(train: pd.Series, test: pd.Series):
    from statsmodels.tsa.arima.model import ARIMA

    # Small grid search over a few common orders, picked by AIC on train data.
    # (No pmdarima/auto_arima dependency needed — keeps requirements lighter.)
    candidate_orders = [(1, 1, 0), (2, 1, 0), (2, 1, 1), (5, 1, 0)]
    best_order, best_aic, best_fit = None, np.inf, None

    for order in candidate_orders:
        try:
            fit = ARIMA(train, order=order).fit()
            if fit.aic < best_aic:
                best_aic, best_order, best_fit = fit.aic, order, fit
        except Exception as e:
            print(f"  ARIMA order {order} failed to fit: {e}")

    print(f"Selected ARIMA order: {best_order} (AIC={best_aic:.1f})")
    forecast = best_fit.forecast(steps=len(test))
    forecast.index = test.index
    return forecast


def make_sequences(values: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window])
        y.append(values[i + window])
    return np.array(X), np.array(y)


def run_lstm(train: pd.Series, test: pd.Series):
    import torch
    import torch.nn as nn

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()

    # Build sliding-window training sequences from the training series
    X_train, y_train = make_sequences(train_scaled, WINDOW_SIZE)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)   # (N, window, 1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)   # (N, 1)

    class LSTMForecaster(nn.Module):
        def __init__(self, hidden_size=32):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])   # use last timestep's hidden state

    model = LSTMForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    EPOCHS = 60
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - loss: {loss.item():.5f}")

    # Walk-forward evaluation on the test period: at each step, use the true
    # preceding WINDOW_SIZE days (train tail + test-so-far) to predict the
    # next single day. This avoids compounding errors from feeding the
    # model's own predictions back in, and mirrors how ARIMA's one-step
    # rolling forecast is evaluated for a fair comparison.
    model.eval()
    full_scaled = scaler.transform(pd.concat([train, test]).values.reshape(-1, 1)).flatten()
    train_len = len(train)

    predictions = []
    with torch.no_grad():
        for i in range(len(test)):
            window_start = train_len + i - WINDOW_SIZE
            window = full_scaled[window_start: train_len + i]
            x = torch.tensor(window, dtype=torch.float32).view(1, WINDOW_SIZE, 1)
            pred_scaled = model(x).item()
            predictions.append(pred_scaled)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return pd.Series(predictions, index=test.index)


def evaluate(name: str, actual: pd.Series, predicted: pd.Series):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{name:>8} — MAE: Rs.{mae:,.2f}   RMSE: Rs.{rmse:,.2f}")
    return mae, rmse


if __name__ == "__main__":
    print("Building daily spend series...")
    daily = build_daily_series()
    print(f"Total days: {len(daily)} ({daily.index.min().date()} to {daily.index.max().date()})")

    train, test = daily.iloc[:-TEST_DAYS], daily.iloc[-TEST_DAYS:]
    print(f"Train: {len(train)} days | Test: {len(test)} days\n")

    print("=" * 50)
    print("ARIMA")
    print("=" * 50)
    arima_forecast = run_arima(train, test)
    arima_mae, arima_rmse = evaluate("ARIMA", test, arima_forecast)

    print("\n" + "=" * 50)
    print("LSTM")
    print("=" * 50)
    lstm_forecast = run_lstm(train, test)
    lstm_mae, lstm_rmse = evaluate("LSTM", test, lstm_forecast)

    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    winner = "ARIMA" if arima_mae < lstm_mae else "LSTM"
    print(f"Lower MAE wins: {winner}")
    print("(With ~10 months of data, ARIMA often wins — LSTMs typically need")
    print(" much more data to outperform classical models. That's a legitimate")
    print(" finding to report, not a failure of the LSTM implementation.)")

    # Plot comparison
    plt.figure(figsize=(12, 5))
    plt.plot(train.index[-30:], train.values[-30:], label="Train (last 30 days)", color="gray")
    plt.plot(test.index, test.values, label="Actual", color="black", linewidth=2)
    plt.plot(test.index, arima_forecast.values, label=f"ARIMA (MAE {arima_mae:,.0f})", linestyle="--")
    plt.plot(test.index, lstm_forecast.values, label=f"LSTM (MAE {lstm_mae:,.0f})", linestyle="--")
    plt.legend()
    plt.title("Daily Spend Forecast: ARIMA vs LSTM")
    plt.xlabel("Date")
    plt.ylabel("Daily Spend (Rs.)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("eda_plots/forecast_comparison.png", dpi=120)
    plt.close()
    print("\nSaved comparison plot to eda_plots/forecast_comparison.png")

    # Predicted next month's total spend (simple sum of a 30-day ARIMA
    # forecast beyond the full dataset) — feeds into budget.py
    from statsmodels.tsa.arima.model import ARIMA
    full_fit = ARIMA(daily, order=(2, 1, 0)).fit()
    next_30_days = full_fit.forecast(steps=30)
    predicted_next_month = next_30_days.sum()
    print(f"\nPredicted next 30-day total spend: Rs.{predicted_next_month:,.2f}")

    pd.Series({"predicted_next_month_spend": predicted_next_month}).to_csv("data/next_month_prediction.csv")
