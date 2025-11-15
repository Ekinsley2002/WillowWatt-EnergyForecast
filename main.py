"""
Willow Watt Energy Forecasting - Unified Training and Testing Script


Model Input Parameters (8 features):
    - lag_1: Most recent energy value (previous 10-min)
    - lag_2: Second most recent energy value (20-min ago)
    - lag_3: Third most recent energy value (30-min ago)
    - lag_4: Fourth most recent energy value (40-min ago)
    - lag_5: Fifth most recent energy value (50-min ago)
    - lag_6: Sixth most recent energy value (60-min ago = 1 hour of history)
    - hour_of_day: Current hour (normalized 0-1)
    - day_of_week: Current day of week (normalized 0-1)

Model Output:
    - Next 10-minute energy usage prediction (in kW, converted to Watts)

Usage:
    python main.py train   # Train a new model
    python main.py test    # Test existing model with forecasting
"""

import glob
import os
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# =========================================================================== #
# CONSTANTS - File Directory Paths
# =========================================================================== #

TRAINING_DATA_DIR = Path("Data/WillowData - Weekly")
TESTING_DATA_PATH = Path("Data/Oct 6--12 2025.csv")
MODEL_PATH = Path("Models/willow_energy_10min.onnx")
FIGURE_PATH = Path("Logs")

# =========================================================================== #
# CONSTANTS - Configuration
# =========================================================================== #

# Time granularity (minutes between data points)
GRANULARITY = 10  # 10 minutes

# Forecast horizon (number of data points to predict)
HORIZON = 1008  # 1008 points = 1 week of 10-minute data

# =========================================================================== #
# TRAINING FUNCTIONALITY
# =========================================================================== #

def train_model():
    """Train a RandomForest model with 8 inputs and export to ONNX."""
    print("=" * 70)
    print("TRAINING MODE")
    print("=" * 70)

    # Ensure output directories exist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------- #
    # Load and prepare training data
    # ----------------------------------------------------------------------- #
    print(f"\nLoading training data from: {TRAINING_DATA_DIR}")
    if not TRAINING_DATA_DIR.exists():
        raise FileNotFoundError(f"Training data directory not found: {TRAINING_DATA_DIR}")

    all_files = glob.glob(os.path.join(TRAINING_DATA_DIR, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {TRAINING_DATA_DIR}")

    print(f"Found {len(all_files)} CSV files")
    dataframes = []

    for file in all_files:
        print(f"  Loading: {os.path.basename(file)}")
        df = pd.read_csv(file)
        df["Time"] = pd.to_datetime(df["Time"], format="mixed")
        df["Energy_kW"] = df["Average"] / 1000
        df = df[["Time", "Energy_kW"]].set_index("Time")
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=False)
    combined_df = combined_df.sort_index()

    print(f"\nTotal records: {len(combined_df)}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

    # Resample to 10-minute intervals
    combined_df_10min = combined_df.resample("10min").mean().ffill()
    print(f"After resampling to {GRANULARITY}-minute intervals: {len(combined_df_10min)} records")

    # Plot data overview
    plt.figure(figsize=(15, 5))
    plt.plot(combined_df_10min.index, combined_df_10min["Energy_kW"])
    plt.title("Willow Energy Usage - 10 Minute Resolution")
    plt.ylabel("Kilowatts (kW)")
    plt.xlabel("Datetime")
    plt.grid(True)
    plt.tight_layout()
    overview_plot_path = FIGURE_PATH / "data_overview_plot_10min.png"
    plt.savefig(overview_plot_path)
    print(f"Saved: {overview_plot_path}")

    # ----------------------------------------------------------------------- #
    # Create features: 6 lags + hour of day + day of week
    # ----------------------------------------------------------------------- #
    print("\nCreating features...")
    df_model = combined_df_10min.copy()

    # Create 6 lag features (last hour of 10-min data)
    for i in range(1, 7):
        df_model[f"lag_{i}"] = df_model["Energy_kW"].shift(i)

    # Add time-based features
    df_model["hour_of_day"] = df_model.index.hour / 23.0  # Normalize 0-1
    df_model["day_of_week"] = df_model.index.dayofweek / 6.0  # Normalize 0-1

    df_model.dropna(inplace=True)

    print(f"Final data after creating features: {len(df_model)} records")
    feature_cols = [f"lag_{i}" for i in range(1, 7)] + ["hour_of_day", "day_of_week"]
    print(f"Features: {feature_cols}")

    # ----------------------------------------------------------------------- #
    # Train model
    # ----------------------------------------------------------------------- #
    X = df_model[feature_cols]
    y = df_model["Energy_kW"]

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    print("\nTraining RandomForest model...")
    model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nModel MSE: {mse:.4f}")
    print(f"Model RMSE: {np.sqrt(mse):.4f} kW")

    # ----------------------------------------------------------------------- #
    # Export to ONNX
    # ----------------------------------------------------------------------- #
    print(f"\nConverting model to ONNX...")
    initial_type = [("float_input", FloatTensorType([None, 8]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Model saved to: {MODEL_PATH}")

    # Verify ONNX model
    session = ort.InferenceSession(MODEL_PATH.as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    print(f"ONNX input name: {input_name}")
    print(f"ONNX input shape: {session.get_inputs()[0].shape}")
    print(f"ONNX output name: {label_name}")

    test_input = X_test.to_numpy().astype(np.float32)
    onnx_pred = session.run([label_name], {input_name: test_input})[0]

    onnx_mse = mean_squared_error(y_test, onnx_pred)
    print(f"ONNX Model MSE: {onnx_mse:.4f}")
    print(f"ONNX Model RMSE: {np.sqrt(onnx_mse):.4f} kW")

    # ----------------------------------------------------------------------- #
    # Plot training results
    # ----------------------------------------------------------------------- #
    y_test_series = y_test.copy()
    y_test_series.index = X_test.index

    plt.figure(figsize=(15, 5))
    plt.plot(y_test_series.index, y_test_series, label="Actual", color="blue")
    plt.plot(y_test_series.index, y_pred, label="Predicted (Sklearn)", color="orange")
    plt.plot(
        y_test_series.index,
        onnx_pred,
        label="Predicted (ONNX)",
        color="green",
        linestyle="dashed",
    )
    plt.title("Actual vs Predicted Energy Usage (8 Input Model)")
    plt.xlabel("Datetime")
    plt.ylabel("Kilowatts (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    comparison_plot_path = FIGURE_PATH / "forecast_comparison_plot_8input.png"
    plt.savefig(comparison_plot_path)
    print(f"\nSaved: {comparison_plot_path}")

    plt.figure(figsize=(15, 5))
    plt.plot(y_test_series.index, y_test_series, label="Actual", color="blue")
    plt.plot(y_test_series.index, y_pred, label="Predicted", color="orange")
    plt.title("Actual vs Predicted Energy Usage (Sklearn Only - 8 Input Model)")
    plt.xlabel("Datetime")
    plt.ylabel("Kilowatts (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    sklearn_only_plot_path = FIGURE_PATH / "forecast_comparison_plot_sklearn_only_8input.png"
    plt.savefig(sklearn_only_plot_path)
    print(f"Saved: {sklearn_only_plot_path}")

    print("\n" + "=" * 70)
    print("Model training completed successfully!")
    print(f"Output model: {MODEL_PATH}")
    print("=" * 70)


# =========================================================================== #
# TESTING FUNCTIONALITY
# =========================================================================== #

def test_model():
    """Test the ONNX model by generating autoregressive forecasts."""
    print("=" * 70)
    print("TESTING MODE")
    print("=" * 70)

    # Ensure output directories exist
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------- #
    # Load ONNX model
    # ----------------------------------------------------------------------- #
    print(f"\nLoading ONNX model from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}. Run 'train' mode first.")

    session = ort.InferenceSession(MODEL_PATH.as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    print(f"Model loaded successfully")
    print(f"Input name: {input_name}")
    print(f"Input shape: {session.get_inputs()[0].shape}")

    # ----------------------------------------------------------------------- #
    # Load testing data
    # ----------------------------------------------------------------------- #
    print(f"\nLoading testing data from: {TESTING_DATA_PATH}")
    if not TESTING_DATA_PATH.exists():
        raise FileNotFoundError(f"Testing data not found: {TESTING_DATA_PATH}")

    df_test = pd.read_csv(TESTING_DATA_PATH, skipinitialspace=True)
    if "Average" not in df_test.columns:
        raise ValueError("Expected column 'Average' not found in testing CSV.")

    # Convert Average to numeric and find first valid value
    df_test["Average"] = pd.to_numeric(df_test["Average"], errors="coerce")
    first_valid_idx = df_test["Average"].first_valid_index()
    
    if first_valid_idx is None:
        raise ValueError("No valid Average values found in testing CSV.")
    
    # Get starting value and datetime from first valid data point
    starting_value_watts = float(df_test.loc[first_valid_idx, "Average"])
    
    # Parse the Time column from the first valid row
    if "Time" not in df_test.columns:
        raise ValueError("Expected column 'Time' not found in testing CSV.")
    
    first_time_str = df_test.loc[first_valid_idx, "Time"]
    # Handle mixed date formats
    try:
        start_datetime = pd.to_datetime(first_time_str, format="mixed")
    except:
        start_datetime = pd.to_datetime(first_time_str)
    
    # Create historical series for plotting (all non-null values)
    historical_series = df_test["Average"].dropna().reset_index(drop=True)
    print(f"Loaded {len(historical_series)} historical data points")
    
    print(f"Starting value (from testing data): {starting_value_watts:,.0f} W")
    print(f"Starting datetime (from testing data): {start_datetime}")
    print(f"  - Raw time string: {first_time_str}")
    print(f"  - Hour of day: {start_datetime.hour}")
    print(f"  - Day of week: {start_datetime.dayofweek} ({start_datetime.strftime('%A')})")

    # ----------------------------------------------------------------------- #
    # Generate forecasts
    # ----------------------------------------------------------------------- #
    print(f"\nGenerating {HORIZON} forecasts (autoregressive)...")

    predictions: List[float] = []
    current_time = start_datetime

    # Initialize with starting value
    predictions.append(starting_value_watts)

    def predict_next_value(history_window: List[float], current_time: pd.Timestamp) -> float:
        """Predict next value using last 6 values + time features."""
        # Convert history to kW
        history_kw = [v / 1_000.0 for v in history_window[-6:]]
        while len(history_kw) < 6:
            history_kw.insert(0, history_kw[0] if history_kw else starting_value_watts / 1_000.0)

        # Extract time features
        hour_normalized = current_time.hour / 23.0
        day_normalized = current_time.dayofweek / 6.0
        
        # Debug first few predictions
        if len(history_window) <= 3:
            print(f"  Step {len(history_window)}: hour={current_time.hour}, day={current_time.dayofweek} ({current_time.strftime('%A')}), hour_norm={hour_normalized:.3f}, day_norm={day_normalized:.3f}")

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

        outputs = session.run(None, {input_name: features})
        predicted_kw = float(outputs[0].squeeze())
        return predicted_kw * 1_000.0

    # Generate predictions autoregressively
    for step in range(1, HORIZON):
        if step % 100 == 0:
            print(f"  Progress: {step}/{HORIZON} ({100*step/HORIZON:.1f}%)")

        next_value = predict_next_value(predictions, current_time)
        predictions.append(next_value)
        current_time = current_time + pd.Timedelta(minutes=GRANULARITY)

    print(f"Generated {len(predictions)} predictions")

    # ----------------------------------------------------------------------- #
    # Plot results
    # ----------------------------------------------------------------------- #
    print("\nGenerating comparison plots...")

    plt.figure(figsize=(14, 8))

    # Full week predictions
    plt.subplot(3, 1, 1)
    plt.plot(predictions, label="Predicted Weekly Energy Usage", color="tab:blue")
    plt.ylabel("Watts")
    plt.title("Willow Watt Weekly Predictions (Full Week)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    plt.subplot(3, 1, 2)
    # First 144 steps = 24 hours to see starting differences
    first_24h_steps = min(144, len(predictions))
    plt.plot(predictions[:first_24h_steps], label="Predicted (First 24h)", color="tab:blue", linewidth=2)
    if len(historical_series) >= first_24h_steps:
        plt.plot(historical_series.values[:first_24h_steps], label="Actual (First 24h)", color="tab:orange", linewidth=2)
    plt.ylabel("Watts")
    plt.title(f"First 24 Hours Comparison (to see starting differences)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    plt.subplot(3, 1, 3)
    plt.plot(historical_series.values, label="Forecast Testing (Average)", color="tab:orange")
    plt.ylabel("Watts")
    plt.xlabel("10-minute Interval")
    plt.title(f"Forecast Testing Data: {TESTING_DATA_PATH.stem}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    plt.tight_layout()
    forecast_plot_path = FIGURE_PATH / "weekly_forecast_comparison.png"
    plt.savefig(forecast_plot_path)
    print(f"Saved: {forecast_plot_path}")
    plt.show()

    print("\n" + "=" * 70)
    print("Model testing completed successfully!")
    print(f"Generated {len(predictions)} forecasts")
    print(f"Comparison plot: {forecast_plot_path}")
    print("=" * 70)


# =========================================================================== #
# MAIN ENTRY POINT
# =========================================================================== #

def main():
    """Main entry point - routes to train or test based on command line argument."""
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|test]")
        print("\n  train  - Train a new model from training data")
        print("  test   - Test existing model with autoregressive forecasting")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "train":
        train_model()
    elif mode == "test":
        test_model()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [train|test]")
        sys.exit(1)


if __name__ == "__main__":
    main()
