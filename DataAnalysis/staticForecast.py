import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

INPUT_DATA_PATH = PROJECT_ROOT / "Data/StaticForecast/Jun 9--15 2025 INPUT.csv"
CONTROL_DATA_PATH = PROJECT_ROOT / "Data/StaticForecast/Jun 16--22 2025 OUTPUT.csv"
MODEL_PATH = PROJECT_ROOT / "Models/willow_energy_10min_multistep.joblib"
FIGURE_PATH = PROJECT_ROOT / "Logs"

HISTORY_WINDOW = 1008
HORIZON = 1008

def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and process CSV data."""
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df["Time"] = pd.to_datetime(df["Time"], format="mixed")
    df["Average"] = pd.to_numeric(df["Average"], errors="coerce")
    df = df[["Time", "Average"]].dropna()
    return df

def prepare_features(history_kw: np.ndarray, start_datetime: pd.Timestamp) -> np.ndarray:
    """Prepare input features for the multi-step model."""
    recent_24h = history_kw[-144:] if len(history_kw) >= 144 else history_kw
    recent_48h = history_kw[-288:] if len(history_kw) >= 288 else history_kw
    recent_168h = history_kw[-168*6:] if len(history_kw) >= 168*6 else history_kw
    
    features = []
    
    if len(history_kw) >= 144:
        features.extend(recent_24h.tolist())
    else:
        features.extend([0.0] * (144 - len(history_kw)) + history_kw.tolist())
    
    features.extend([
        np.mean(recent_24h),
        np.std(recent_24h) if len(recent_24h) > 1 else 0.0,
        np.min(recent_24h),
        np.max(recent_24h),
        np.mean(recent_48h) if len(recent_48h) > 0 else np.mean(recent_24h),
        np.std(recent_48h) if len(recent_48h) > 1 else 0.0,
        np.mean(recent_168h) if len(recent_168h) > 0 else np.mean(recent_24h),
    ])
    
    features.extend([
        start_datetime.hour / 23.0,
        start_datetime.dayofweek / 6.0,
        start_datetime.month / 12.0,
    ])
    
    return np.array([features], dtype=np.float32)

def main():
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(MODEL_PATH)
    
    input_df = load_data(INPUT_DATA_PATH)
    control_df = load_data(CONTROL_DATA_PATH)
    
    input_series_kw = input_df["Average"].values / 1000.0
    start_datetime = input_df["Time"].iloc[-1] + pd.Timedelta(minutes=10)
    
    if len(input_series_kw) < HISTORY_WINDOW:
        mean_val = np.mean(input_series_kw)
        padding = [mean_val] * (HISTORY_WINDOW - len(input_series_kw))
        history = np.concatenate([padding, input_series_kw])
    else:
        history = input_series_kw[-HISTORY_WINDOW:].copy()
    
    X_input = prepare_features(history, start_datetime)
    predictions_kw = model.predict(X_input)[0]
    predictions_watts = predictions_kw * 1000.0
    
    input_times = pd.to_datetime(input_df["Time"].values)
    prediction_times = pd.date_range(start=start_datetime, periods=HORIZON, freq="10min")
    control_times = pd.to_datetime(control_df["Time"].values)
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(input_times, input_df["Average"].values, label="Input Data (Jun 9--15, 2025)", color="blue", linewidth=1.5)
    plt.plot(prediction_times, predictions_watts, label="Model Predictions (Next Week)", color="orange", linewidth=1.5)
    plt.ylabel("Watts")
    plt.title("Model Forecast: Input Data vs Predictions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(input_times, input_df["Average"].values, label="Input Data (Jun 9--15, 2025)", color="blue", linewidth=1.5)
    plt.plot(control_times, control_df["Average"].values, label="Actual Data (Jun 16--22, 2025)", color="green", linewidth=1.5)
    plt.ylabel("Watts")
    plt.xlabel("Time")
    plt.title("Control Comparison: Input Data vs Actual Next Week")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = FIGURE_PATH / "static_forecast_comparison.png"
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    main()