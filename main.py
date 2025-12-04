"""
Willow Watt Energy Forecasting - Direct Multi-Step Forecasting

This version implements DIRECT MULTI-STEP forecasting to predict all 1008 steps at once:
1. Direct multi-step: Model predicts entire week (1008 steps) in single forward pass
2. No error accumulation: Each prediction is independent, no autoregressive feedback
3. History window: Uses last 7 days (1008 points) as input context
4. Time features: Adds hour_of_day and day_of_week for each future step

Model Input: History window (1008 past values) + time features
Model Output: All 1008 future steps predicted simultaneously

THIS MODEL IS DESIGNED TO MAKE 1 week of predictions at once (1008 steps @ 10 min a step)

Usage:
    python main2.py train   # Train a new multi-step model
    python main2.py test    # Test with direct multi-step forecasting
"""

import glob
import os
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# =========================================================================== #
# CONSTANTS - File Directory Paths
# =========================================================================== #

TRAINING_DATA_DIR = Path("Data/WillowData - Weekly")
TESTING_DATA_PATH = Path("Data/Oct 6--12 2025.csv")
MODEL_PATH = Path("Models/willow_energy_10min_multistep.onnx")
FIGURE_PATH = Path("Logs")

# =========================================================================== #
# CONSTANTS - Configuration
# =========================================================================== #

# Time granularity (minutes between data points)
GRANULARITY = 10  # 10 minutes

# Forecast horizon (number of data points to predict)
HORIZON = 1008  # 1008 points = 1 week of 10-minute data

# History window size (number of past steps to use as input)
# Use 7 days = 1008 steps to predict next 7 days
HISTORY_WINDOW = 1008  # 7 days of 10-minute data


# =========================================================================== #
# TRAINING FUNCTIONALITY
# =========================================================================== #

def train_model():
    """Train a multi-output RandomForest model that predicts all 1008 steps at once."""
    print("=" * 70)
    print("TRAINING MODE (DIRECT MULTI-STEP)")
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
    overview_plot_path = FIGURE_PATH / "data_overview_plot_multistep.png"
    plt.savefig(overview_plot_path)
    print(f"Saved: {overview_plot_path}")

    # ----------------------------------------------------------------------- #
    # Create multi-step training samples
    # ----------------------------------------------------------------------- #
    print("\nCreating multi-step training samples...")
    print(f"History window: {HISTORY_WINDOW} steps (7 days)")
    print(f"Forecast horizon: {HORIZON} steps (7 days)")
    
    # Extract energy values as array
    energy_values = combined_df_10min["Energy_kW"].values
    timestamps = combined_df_10min.index
    
    # Create training samples: for each valid position, use HISTORY_WINDOW as input, HORIZON as output
    X_samples = []
    y_samples = []
    
    min_samples_needed = HISTORY_WINDOW + HORIZON
    print(f"Creating samples from {len(energy_values)} total values...")
    
    for i in range(len(energy_values) - min_samples_needed + 1):
        # Input: last HISTORY_WINDOW values (flattened) + summary statistics
        history = energy_values[i:i+HISTORY_WINDOW]
        
        # Create compact features: recent values + summary stats
        # Use last 144 values (24h) + summary stats from different windows
        recent_24h = history[-144:] if len(history) >= 144 else history
        recent_48h = history[-288:] if len(history) >= 288 else history
        recent_168h = history[-168*6:] if len(history) >= 168*6 else history  # Last 7 days at hourly
        
        # Compress history: use recent values + stats
        features = []
        
        # Last 144 values (24 hours) - key recent context
        if len(history) >= 144:
            features.extend(recent_24h.tolist())
        else:
            features.extend([0.0] * (144 - len(history)) + history.tolist())
        
        # Summary statistics
        features.extend([
            np.mean(recent_24h),
            np.std(recent_24h) if len(recent_24h) > 1 else 0.0,
            np.min(recent_24h),
            np.max(recent_24h),
            np.mean(recent_48h) if len(recent_48h) > 0 else np.mean(recent_24h),
            np.std(recent_48h) if len(recent_48h) > 1 else 0.0,
            np.mean(recent_168h) if len(recent_168h) > 0 else np.mean(recent_24h),
        ])
        
        # Starting time features (for the prediction start point)
        start_time = timestamps[i + HISTORY_WINDOW]
        features.extend([
            start_time.hour / 23.0,  # Hour of day
            start_time.dayofweek / 6.0,  # Day of week
            start_time.month / 12.0,  # Month
        ])
        
        # Output: next HORIZON values
        future_values = energy_values[i+HISTORY_WINDOW:i+HISTORY_WINDOW+HORIZON]
        
        X_samples.append(features)
        y_samples.append(future_values)
    
    X = np.array(X_samples, dtype=np.float32)
    y = np.array(y_samples, dtype=np.float32)
    
    print(f"Created {len(X_samples)} training samples")
    print(f"Input shape: {X.shape} ({X.shape[1]} features)")
    print(f"Output shape: {y.shape} ({y.shape[1]} future steps)")

    # ----------------------------------------------------------------------- #
    # Train multi-output model
    # ----------------------------------------------------------------------- #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    print("\nTraining Multi-Output RandomForest model...")
    print("(This predicts all 1008 steps at once - may take a few minutes)")
    
    # Use MultiOutputRegressor to handle 1008 outputs
    base_rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    model = MultiOutputRegressor(base_rf, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    # Calculate MSE for each step
    mse_per_step = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    overall_mse = mean_squared_error(y_test, y_pred)
    
    print(f"\nOverall MSE: {overall_mse:.4f}")
    print(f"Overall RMSE: {np.sqrt(overall_mse):.4f} kW")
    print(f"Average MSE per step: {np.mean(mse_per_step):.4f}")
    print(f"MSE at step 1: {mse_per_step[0]:.4f}, step 504: {mse_per_step[503]:.4f}, step 1008: {mse_per_step[-1]:.4f}")

    # ----------------------------------------------------------------------- #
    # Export to ONNX (Note: MultiOutputRegressor ONNX export is complex)
    # ----------------------------------------------------------------------- #
    print(f"\nNote: Multi-Output ONNX export is complex. Saving model using joblib instead...")
    import joblib
    
    # Save as joblib (ONNX doesn't easily support multi-output)
    model_path_joblib = MODEL_PATH.with_suffix('.joblib')
    joblib.dump(model, model_path_joblib)
    print(f"Model saved to: {model_path_joblib}")
    
    # For ONNX, we'd need to export each output separately or use a different approach
    # For now, we'll use joblib in testing mode
    print("Using joblib format for multi-step model (ONNX support is limited)")

    # ----------------------------------------------------------------------- #
    # Plot training results (sample a few test cases)
    # ----------------------------------------------------------------------- #
    print("\nGenerating training evaluation plots...")
    
    # Plot a sample prediction from test set
    sample_idx = len(y_test) // 2  # Middle of test set
    sample_actual = y_test[sample_idx]
    sample_pred = y_pred[sample_idx]
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(sample_actual, label="Actual (Sample Test Case)", color="blue", linewidth=2)
    plt.plot(sample_pred, label="Predicted (Sample Test Case)", color="orange", linewidth=2)
    plt.title("Direct Multi-Step Prediction - Sample Test Case (All 1008 Steps)")
    plt.xlabel("Step (10-minute intervals)")
    plt.ylabel("Kilowatts (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    # Show first 144 steps (24 hours) in detail
    plt.plot(sample_actual[:144], label="Actual (First 24h)", color="blue", linewidth=2)
    plt.plot(sample_pred[:144], label="Predicted (First 24h)", color="orange", linewidth=2)
    plt.title("Detail View: First 24 Hours")
    plt.xlabel("Step (10-minute intervals)")
    plt.ylabel("Kilowatts (kW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comparison_plot_path = FIGURE_PATH / "forecast_comparison_plot_multistep.png"
    plt.savefig(comparison_plot_path)
    print(f"Saved: {comparison_plot_path}")

    print("\n" + "=" * 70)
    print("Model training completed successfully!")
    print(f"Output model: {model_path_joblib}")
    print("=" * 70)


# =========================================================================== #
# TESTING FUNCTIONALITY
# =========================================================================== #

def test_model():
    """Test the direct multi-step model - predicts all 1008 steps at once."""
    print("=" * 70)
    print("TESTING MODE (DIRECT MULTI-STEP)")
    print("=" * 70)

    # Ensure output directories exist
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------- #
    # Load model (joblib format)
    # ----------------------------------------------------------------------- #
    import joblib
    
    model_path_joblib = MODEL_PATH.with_suffix('.joblib')
    print(f"\nLoading model from: {model_path_joblib}")
    if not model_path_joblib.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path_joblib}. Run 'train' mode first."
        )

    model = joblib.load(model_path_joblib)
    print(f"Model loaded successfully")
    print(f"Model type: {type(model)}")

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
    try:
        start_datetime = pd.to_datetime(first_time_str, format="mixed")
    except:
        start_datetime = pd.to_datetime(first_time_str)
    
    # Create historical series for plotting (all non-null values)
    historical_series = df_test["Average"].dropna().reset_index(drop=True)
    print(f"Loaded {len(historical_series)} historical data points")
    
    # Convert to kW for model input
    historical_series_kw = historical_series / 1000.0
    
    print(f"Starting datetime (from testing data): {start_datetime}")
    
    # Get history window for prediction (use available data, pad if needed)
    available_history = len(historical_series_kw)
    
    if available_history < HISTORY_WINDOW:
        print(f"Warning: Only {available_history} data points available, need {HISTORY_WINDOW}.")
        print(f"Will use available data and pad with mean value if needed.")
        
        # Use all available history
        history = historical_series_kw.values.copy()
        
        # Pad to HISTORY_WINDOW with mean value
        mean_value = np.mean(history)
        padding_needed = HISTORY_WINDOW - len(history)
        history = np.concatenate([[mean_value] * padding_needed, history])
        
        print(f"Padded history to {len(history)} values (using mean: {mean_value:.2f} kW)")
    else:
        # Use last HISTORY_WINDOW values
        history = historical_series_kw.values[-HISTORY_WINDOW:].copy()
        print(f"Using last {HISTORY_WINDOW} values as history window")
    
    # Get corresponding timestamps (for info only)
    valid_indices = df_test["Average"].notna()
    valid_times = df_test.loc[valid_indices, "Time"].values
    if len(valid_times) > 0:
        print(f"History date range: {valid_times[0]} to {valid_times[-1]}")

    # ----------------------------------------------------------------------- #
    # Build input features (same as training)
    # ----------------------------------------------------------------------- #
    print("\nBuilding input features...")
    
    # Create compact features: recent values + summary stats (same as training)
    recent_24h = history[-144:] if len(history) >= 144 else history
    recent_48h = history[-288:] if len(history) >= 288 else history
    recent_168h = history[-168*6:] if len(history) >= 168*6 else history
    
    features = []
    
    # Last 144 values (24 hours) - key recent context
    if len(history) >= 144:
        features.extend(recent_24h.tolist())
    else:
        features.extend([0.0] * (144 - len(history)) + history.tolist())
    
    # Summary statistics
    features.extend([
        np.mean(recent_24h),
        np.std(recent_24h) if len(recent_24h) > 1 else 0.0,
        np.min(recent_24h),
        np.max(recent_24h),
        np.mean(recent_48h) if len(recent_48h) > 0 else np.mean(recent_24h),
        np.std(recent_48h) if len(recent_48h) > 1 else 0.0,
        np.mean(recent_168h) if len(recent_168h) > 0 else np.mean(recent_24h),
    ])
    
    # Starting time features (for the prediction start point = current time)
    features.extend([
        start_datetime.hour / 23.0,  # Hour of day
        start_datetime.dayofweek / 6.0,  # Day of week
        start_datetime.month / 12.0,  # Month
    ])
    
    X_input = np.array([features], dtype=np.float32)
    print(f"Input features shape: {X_input.shape}")
    
    # ----------------------------------------------------------------------- #
    # Generate all 1008 predictions at once!
    # ----------------------------------------------------------------------- #
    print(f"\nGenerating all {HORIZON} forecasts in single forward pass...")
    predictions_kw = model.predict(X_input)[0]  # Get first (and only) prediction
    predictions = predictions_kw * 1000.0  # Convert back to Watts
    
    print(f"Generated {len(predictions)} predictions at once!")
    print(f"Prediction range: {predictions.min():,.0f} - {predictions.max():,.0f} W")

    # ----------------------------------------------------------------------- #
    # Plot results
    # ----------------------------------------------------------------------- #
    print("\nGenerating comparison plots...")

    plt.figure(figsize=(14, 10))

    # Full week predictions
    plt.subplot(3, 1, 1)
    plt.plot(predictions, label="Predicted Weekly Energy Usage (Direct Multi-Step)", color="tab:blue", linewidth=2)
    if len(historical_series) >= HORIZON:
        plt.plot(historical_series.values[:HORIZON], label="Actual (Full Week)", color="tab:orange", linewidth=2)
    plt.ylabel("Watts")
    plt.title("Willow Watt Weekly Predictions - Direct Multi-Step (All 1008 Steps at Once)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")

    plt.subplot(3, 1, 2)
    # First 144 steps = 24 hours to see starting accuracy
    first_24h_steps = min(144, len(predictions))
    plt.plot(predictions[:first_24h_steps], label="Predicted (First 24h)", color="tab:blue", linewidth=2)
    if len(historical_series) >= first_24h_steps:
        plt.plot(historical_series.values[:first_24h_steps], label="Actual (First 24h)", color="tab:orange", linewidth=2)
    plt.ylabel("Watts")
    plt.title(f"First 24 Hours Comparison")
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
    forecast_plot_path = FIGURE_PATH / "weekly_forecast_comparison_multistep.png"
    plt.savefig(forecast_plot_path)
    print(f"Saved: {forecast_plot_path}")
    plt.show()

    print("\n" + "=" * 70)
    print("Model testing completed successfully!")
    print(f"Generated all {len(predictions)} forecasts in single forward pass (no error accumulation!)")
    print(f"Comparison plot: {forecast_plot_path}")
    print("=" * 70)


# =========================================================================== #
# MAIN ENTRY POINT
# =========================================================================== #

def main():
    """Main entry point - routes to train or test based on command line argument."""
    if len(sys.argv) < 2:
        print("Usage: python main2.py [train|test]")
        print("\n  train  - Train a new improved model from training data")
        print("  test   - Test improved model with hybrid autoregressive forecasting")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "train":
        train_model()
    elif mode == "test":
        test_model()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main2.py [train|test]")
        sys.exit(1)


if __name__ == "__main__":
    main()

