import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import numpy as np
import datetime
from pathlib import Path


# ===================== CONFIG =====================
EXCEL_PATH = Path("Data/NC-Energy-Cnsmp.xlsx")
SHEET_NAME = "15 minute Data"
RESAMPLE_RULE = "12H"
TIMEZONE = None                 # e.g., "America/Phoenix" if you want to localize
FORECAST_PERIODS = 14
# ==================================================


def _norm(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower().replace(" ", "").replace("_", "")


def load_excel_36_resampled(excel_path: Path, sheet_name=SHEET_NAME, rule=RESAMPLE_RULE, tz=None) -> pd.DataFrame:
    """
    Loads the '15 minute Data' sheet, finds the header row, selects the '36' block
    (leftmost Date/kWh pair), and resamples to 12-hour Average/Maximum/Minimum.

    Returns DataFrame with columns: Time, Average, Maximum, Minimum
    """
    # 1) Read raw without assuming header to detect true header row
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, dtype=str)

    header_idx = None
    # Find the first row that contains something like "date"/"timestamp" and "kwh"/"energy"
    for i in range(min(120, len(raw))):
        row_norm = [_norm(x) for x in raw.iloc[i].tolist()]
        if ("date" in row_norm or "datetime" in row_norm or "timestamp" in row_norm) and \
           ("kwh" in row_norm or "energy" in row_norm or "usage" in row_norm):
            header_idx = i
            break

    # Fallback: if a row contains literal '36', assume next row is headers
    if header_idx is None:
        for i in range(min(120, len(raw))):
            if any(str(x).strip() == "36" for x in raw.iloc[i].tolist()):
                header_idx = i + 1 if i + 1 < len(raw) else None
                break

    if header_idx is None:
        raise ValueError("Could not locate a header row in the sheet. Please verify the workbook structure.")

    # 2) Re-read with the detected header
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=header_idx)

    # 3) Select the bldg 36 block: first Date/kWh pair
    candidate_date_cols = [c for c in df.columns if str(c).startswith("Date")]
    candidate_kwh_cols  = [c for c in df.columns if str(c).startswith("kWh")]

    if not candidate_date_cols or not candidate_kwh_cols:
        raise ValueError("Expected 'Date' and 'kWh' columns were not found after header detection.")

    date_col = candidate_date_cols[0]
    val_col  = candidate_kwh_cols[0]

    sub = df[[date_col, val_col]].copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub[val_col]  = pd.to_numeric(sub[val_col], errors="coerce")
    sub = sub.dropna(subset=[date_col, val_col]).sort_values(date_col)

    # Index + optional timezone handling
    sub = sub.set_index(date_col)
    if tz:
        if sub.index.tz is None:
            sub.index = sub.index.tz_localize(tz)
        else:
            sub.index = sub.index.tz_convert(tz)

    # 4) Resample to 12-hour buckets (Average/Maximum/Minimum)
    agg = sub[val_col].resample(rule).agg(["mean", "max", "min"]).dropna()
    out = agg.rename(columns={"mean": "Average", "max": "Maximum", "min": "Minimum"}).reset_index()
    out = out.rename(columns={out.columns[0]: "Time"})
    out = out[["Time", "Average", "Maximum", "Minimum"]]

    # Final clean/sort
    out = out.dropna(subset=["Time", "Average", "Maximum", "Minimum"]).sort_values("Time").reset_index(drop=True)
    return out


def build_features_adaptive(df: pd.DataFrame, max_lag_periods: int = 2):
    work = df.copy()
    # Base time features
    work["hour"]        = pd.to_datetime(work["Time"]).dt.hour
    work["day_of_week"] = pd.to_datetime(work["Time"]).dt.dayofweek
    work["month"]       = pd.to_datetime(work["Time"]).dt.month
    work["day_of_year"] = pd.to_datetime(work["Time"]).dt.dayofyear

    best_w, best_cols, best_lag = None, None, None
    for lag in range(max_lag_periods, -1, -1):
        w = work.copy()
        feature_cols = ["hour", "day_of_week", "month", "day_of_year"]

        if lag >= 1:
            w["prev_day_avg"] = w["Average"].shift(lag)
            w["prev_day_max"] = w["Maximum"].shift(lag)
            w["prev_day_min"] = w["Minimum"].shift(lag)
            feature_cols += ["prev_day_avg", "prev_day_max", "prev_day_min"]

        # Drop rows with NaNs in required columns
        needed_cols = ["Average", "Maximum", "Minimum"]
        if lag >= 1:
            needed_cols += ["prev_day_avg", "prev_day_max", "prev_day_min"]

        w = w.dropna(subset=needed_cols)
        if len(w) >= 10:
            print(f"[INFO] Using lag={lag}. Rows after feature build: {len(w)}")
            return w.reset_index(drop=True), feature_cols, lag

        # Keep the last attempt for fallback
        best_w, best_cols, best_lag = w, feature_cols, lag

    # Fallback if all attempts are small
    print(f"[WARN] Very small dataset. Proceeding with lag={best_lag}. Rows: {len(best_w)}")
    return best_w.reset_index(drop=True), best_cols, best_lag


# ===================== Load & Prepare Data =====================
df = load_excel_36_resampled(EXCEL_PATH, sheet_name=SHEET_NAME, rule=RESAMPLE_RULE, tz=TIMEZONE)
print(f"[INFO] Loaded rows: {len(df)} from {pd.to_datetime(df['Time']).min()} to {pd.to_datetime(df['Time']).max()}")

# Build features (adaptive lags to avoid empty-train issues)
df_model, feature_columns, used_lag = build_features_adaptive(df, max_lag_periods=2)
print(f"[INFO] Feature columns: {feature_columns}")
print(f"[INFO] Rows ready for modeling: {len(df_model)}")

# ===================== Train/Test Split =====================
X = df_model[feature_columns]
y = df_model["Maximum"]

n = len(df_model)
if n == 0:
    raise ValueError("No rows available after cleaning/feature building. "
                     "Resampling may have produced too few periods or values were all NaN.")

# Chronological split; ensure at least 1 training row
split_point = max(1, int(n * 0.8)) if n > 1 else 1
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

# ===================== Train Model =====================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Optional quick evaluation
if len(X_test) > 0:
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"[INFO] Holdout RMSE: {rmse:.3f} (same units as 'Maximum')")

# ===================== Future Prediction Window =====================
# Build 14 future 12-hour timestamps starting right after the last row
last_time = pd.to_datetime(df["Time"].iloc[-1])
future_dates = [last_time + datetime.timedelta(hours=12 * (i + 1)) for i in range(FORECAST_PERIODS)]
future_df = pd.DataFrame({"Time": future_dates})

# Time features
future_df["hour"]        = future_df["Time"].dt.hour
future_df["day_of_week"] = future_df["Time"].dt.dayofweek
future_df["month"]       = future_df["Time"].dt.month
future_df["day_of_year"] = future_df["Time"].dt.dayofyear

# Only add prev_* columns if they were used
if "prev_day_avg" in feature_columns:
    last_avg = df["Average"].iloc[-1]
    last_max = df["Maximum"].iloc[-1]
    last_min = df["Minimum"].iloc[-1]
    future_df["prev_day_avg"] = last_avg
    future_df["prev_day_max"] = last_max
    future_df["prev_day_min"] = last_min

future_features = future_df[feature_columns]
future_predictions = model.predict(future_features)

# ===================== Reporting =====================
max_prediction_idx = int(np.argmax(future_predictions))
max_prediction_value = float(np.max(future_predictions))

predicted_date = future_df.iloc[max_prediction_idx]["Time"]
day_name = predicted_date.strftime("%A")
time_period = "AM" if predicted_date.hour < 12 else "PM"
total_periods = len(future_df)

units_label = " (same units as your 'Maximum' column)"
print("\n--- Prediction Summary ---\n")
print(f"Highest predicted peak: {max_prediction_value:.2f}{units_label}")
print(f"Predicted for: {day_name} {time_period} ({predicted_date})")
print(f"This is data point {max_prediction_idx + 1} in the next {total_periods} periods")
print("\n--------------------------")

# User I/O for further data
print(f"\nWould you like to see the full {total_periods}-period prediction? (y/n) : ", end="")
user_input = input().strip().lower()
if user_input == "y":
    for idx, (time, pred) in enumerate(zip(future_df["Time"], future_predictions)):
        day_name = pd.to_datetime(time).strftime("%A")
        time_period = "AM" if pd.to_datetime(time).hour < 12 else "PM"
        print(f"Period {idx + 1}: {day_name} {time_period} ({time}) - Predicted Peak: {pred:.2f}{units_label}")
else:
    print("Exiting without showing full prediction.")
