import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


data_folder = "../Data/Weekly_15_Min"
all_files = glob.glob(os.path.join(data_folder, "*.csv"))

print(f"Found {len(all_files)} CSV files")
dataframes = []

for file in all_files:
    print(f"Loading: {os.path.basename(file)}")
    df = pd.read_csv(file)
    
    df["Time"] = pd.to_datetime(df["Time"], format="mixed")
    
    df["Energy_kW"] = df["Average"] / 1000
    
    df = df[["Time", "Energy_kW"]].set_index("Time")
    
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=False)
combined_df = combined_df.sort_index()

print(f"\nTotal records: {len(combined_df)}")
print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

# --------------------------------------------------------------------------- #
# Build a continuous 15-minute time-series.
# --------------------------------------------------------------------------- #
combined_df_15min = combined_df.resample("15min").mean().ffill()
print(f"\nAfter resampling to 15-minute intervals: {len(combined_df_15min)} records")


def train_and_export(df, lag_column_name, model_path, log_prefix):
    """Train a RandomForest on the provided dataframe and export to ONNX."""
    df_model = df.copy()
    
    # Create multiple lag features (last 6 values = 1.5 hours of 15-min data)
    for i in range(1, 7):
        df_model[f"lag_{i}"] = df_model["Energy_kW"].shift(i)
    
    # Add time-based features
    df_model["hour_of_day"] = df_model.index.hour / 23.0  # Normalize 0-1
    df_model["day_of_week"] = df_model.index.dayofweek / 6.0  # Normalize 0-1
    
    df_model.dropna(inplace=True)

    print(
        f"\nFinal {log_prefix} data after creating lag features: {len(df_model)} records"
    )

    # Use all lag features plus time features
    feature_cols = [f"lag_{i}" for i in range(1, 7)] + ["hour_of_day", "day_of_week"]
    X = df_model[feature_cols]
    y = df_model["Energy_kW"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"{log_prefix} model MSE: {mean_squared_error(y_test, y_pred):.4f}")

    # Update ONNX input type to match 8 features (6 lags + hour + day)
    initial_type = [("float_input", FloatTensorType([None, 8]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"{log_prefix} model converted to ONNX: {model_path}")

    session = rt.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    test_input = X_test.to_numpy().astype(np.float32)
    onnx_pred = session.run([label_name], {input_name: test_input})[0]

    print(f"{log_prefix} ONNX Model MSE: {mean_squared_error(y_test, onnx_pred):.4f}")

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

    plt.title(f"Actual vs Predicted Energy Usage ({log_prefix})")
    plt.xlabel("Datetime")
    plt.ylabel("Kilowatts (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../Logs/forecast_comparison_plot_{log_prefix}.png")
    print(f"Saved: ../Logs/forecast_comparison_plot_{log_prefix}.png")

    plt.figure(figsize=(15, 5))
    plt.plot(y_test_series.index, y_test_series, label="Actual", color="blue")
    plt.plot(y_test_series.index, y_pred, label="Predicted (Sklearn)", color="orange")

    plt.title(f"Actual vs Predicted Energy Usage (Sklearn Only - {log_prefix})")
    plt.xlabel("Datetime")
    plt.ylabel("Kilowatts (kW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../Logs/forecast_comparison_plot_sklearn_only_{log_prefix}.png")
    print(f"Saved: ../Logs/forecast_comparison_plot_sklearn_only_{log_prefix}.png")


# --------------------------------------------------------------------------- #
# Train and export the 15-minute model.
# --------------------------------------------------------------------------- #
plt.figure(figsize=(15, 5))
plt.plot(combined_df_15min.index, combined_df_15min["Energy_kW"])
plt.title("Willow Energy Usage - 15 Minute Resolution")
plt.ylabel("Kilowatts (kW)")
plt.xlabel("Datetime")
plt.grid(True)
plt.tight_layout()
plt.savefig("../Logs/data_overview_plot_15min.png")
print("Saved: ../Logs/data_overview_plot_15min.png")

train_and_export(
    combined_df_15min,
    lag_column_name="prev_15min",
    model_path="../Models/willow_energy_15min.onnx",
    log_prefix="15min",
)

print("\nModel training completed successfully!")
