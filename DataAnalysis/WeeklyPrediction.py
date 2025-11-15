# Weekly energy prediction using the Willow Watt ONNX model.

from __future__ import annotations

import pathlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd


# --- Configuration --------------------------------------------------------- #

# Starting value (in W) (should be first of week)
STARTING_VALUE_WATTS: float = 3_447_235.80

# Starting datetime (first of week - Nov 3, 2025, 12:00 AM)
START_DATETIME = pd.Timestamp("2025-11-03 00:00:00")

# 10 min data -> 6 intervals every hour -> 144 every day -> 1008 every week
MAX_DATA_POINTS: int = 1_008

# Relative path to the ONNX model from this script (use 10min model which has time features)
MODEL_PATH = (pathlib.Path(__file__).resolve().parent / "../Models/willow_energy_10min.onnx").resolve()
DATA_PATH = (pathlib.Path(__file__).resolve().parent / "../Data/Nov 03--09 2025 (FORECAST TESTING).csv").resolve()


# --- Model Session --------------------------------------------------------- #

def _create_session(model_path: pathlib.Path) -> tuple[ort.InferenceSession, str]:
    """
    Create a single ONNX Runtime session and return it with the primary input name.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {model_path.resolve()}")

    session = ort.InferenceSession(
        model_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )

    if not session.get_inputs():
        raise RuntimeError("The ONNX model has no inputs defined.")

    input_name = session.get_inputs()[0].name
    return session, input_name


SESSION, SESSION_INPUT_NAME = _create_session(MODEL_PATH)


# --- Prediction Helpers ---------------------------------------------------- #

def predict_next_value(history_window: List[float], current_time: pd.Timestamp) -> float:
    """
    Run the ONNX model to predict the next value using:
    - Last 6 values (lags)
    - Hour of day (normalized)
    - Day of week (normalized)
    """
    # Convert history to kW and ensure we have at least 6 values (pad with last value if needed)
    history_kw = [v / 1_000.0 for v in history_window[-6:]]
    while len(history_kw) < 6:
        history_kw.insert(0, history_kw[0] if history_kw else STARTING_VALUE_WATTS / 1_000.0)
    
    # Extract time features
    hour_normalized = current_time.hour / 23.0
    day_normalized = current_time.dayofweek / 6.0
    
    # Build feature vector: [lag_1, lag_2, lag_3, lag_4, lag_5, lag_6, hour, day]
    features = np.array([[
        history_kw[-1],  # lag_1 (most recent)
        history_kw[-2] if len(history_kw) >= 2 else history_kw[-1],  # lag_2
        history_kw[-3] if len(history_kw) >= 3 else history_kw[-1],  # lag_3
        history_kw[-4] if len(history_kw) >= 4 else history_kw[-1],  # lag_4
        history_kw[-5] if len(history_kw) >= 5 else history_kw[-1],  # lag_5
        history_kw[-6] if len(history_kw) >= 6 else history_kw[-1],  # lag_6
        hour_normalized,
        day_normalized
    ]], dtype=np.float32)
    
    outputs = SESSION.run(None, {SESSION_INPUT_NAME: features})
    if not outputs:
        raise RuntimeError("Model inference returned no outputs.")
    
    predicted_kw = float(outputs[0].squeeze())
    return predicted_kw * 1_000.0


def store_value(history: List[float], data_point: float) -> None:
    """
    Store the generated data point inside the provided list.
    """
    history.append(data_point)


def find_next_value(history_window: List[float], current_time: pd.Timestamp) -> float:
    """
    Wrapper that calls the model to obtain the next predicted value.
    """
    return predict_next_value(history_window, current_time)


def load_forecast_testing_series() -> pd.Series:
    """
    Load the historical forecast testing data and return the 'Average' column as a series.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Forecast testing CSV not found at {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH, skipinitialspace=True)

    if "Average" not in df.columns:
        raise ValueError("Expected column 'Average' not found in forecast testing CSV.")

    series = pd.to_numeric(df["Average"], errors="coerce").dropna().reset_index(drop=True)
    return series


def plot_predictions(predictions: List[float], historical_series: pd.Series) -> None:
    """
    Plot generated predictions and historical forecast testing values.
    """
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    plt.plot(predictions, label="Predicted Weekly Energy Usage", color="tab:blue")
    plt.ylabel("Watts")
    plt.title("Willow Watt Weekly Predictions")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    plt.subplot(2, 1, 2)
    plt.plot(historical_series.values, label="Forecast Testing (Average)", color="tab:orange")
    plt.ylabel("Watts")
    plt.xlabel("10-minute Interval")
    plt.title("Forecast Testing Data: Nov 03–09 2025")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# --- Main Generation Loop -------------------------------------------------- #

def main() -> List[float]:
    """
    Generate a week's worth of predictions starting from STARTING_VALUE_WATTS.
    """
    data_points: List[float] = []
    current_time = START_DATETIME

    # Initialize with starting value
    store_value(data_points, STARTING_VALUE_WATTS)

    for step in range(1, MAX_DATA_POINTS):
        # Predict next value using history and current time
        next_value = find_next_value(data_points, current_time)
        store_value(data_points, next_value)
        
        # Advance time by 10 minutes
        current_time = current_time + pd.Timedelta(minutes=10)

    return data_points


if __name__ == "__main__":
    weekly_predictions = main()
    historical_average = load_forecast_testing_series()

    print(f"Generated {len(weekly_predictions)} weekly predictions.")
    print(f"Loaded {len(historical_average)} rows from forecast testing CSV.")

    plot_predictions(weekly_predictions, historical_average)
