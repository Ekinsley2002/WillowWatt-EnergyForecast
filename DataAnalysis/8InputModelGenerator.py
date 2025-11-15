"""
Train a RandomForest model with 8 inputs for 10-minute energy prediction:
- 6 lag features (last 6 values = 1 hour of 10-minute data)
- Hour of day (normalized)
- Day of week (normalized)
"""

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


# --------------------------------------------------------------------------- #
# Load and prepare data
# --------------------------------------------------------------------------- #

data_folder = "../Data/WillowData - Weekly"
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

# Resample to 10-minute intervals
combined_df_10min = combined_df.resample("10min").mean().ffill()
print(f"\nAfter resampling to 10-minute intervals: {len(combined_df_10min)} records")

# Plot data overview
plt.figure(figsize=(15, 5))
plt.plot(combined_df_10min.index, combined_df_10min["Energy_kW"])
plt.title("Willow Energy Usage - 10 Minute Resolution")
plt.ylabel("Kilowatts (kW)")
plt.xlabel("Datetime")
plt.grid(True)
plt.tight_layout()
plt.savefig("../Logs/data_overview_plot_10min.png")
print("Saved: ../Logs/data_overview_plot_10min.png")


# --------------------------------------------------------------------------- #
# Create features: 6 lags + hour of day + day of week
# --------------------------------------------------------------------------- #

df_model = combined_df_10min.copy()

# Create 6 lag features (last hour of 10-min data)
print("\nCreating lag features...")
for i in range(1, 7):
    df_model[f"lag_{i}"] = df_model["Energy_kW"].shift(i)

# Add time-based features
print("Adding time features...")
df_model["hour_of_day"] = df_model.index.hour / 23.0  # Normalize 0-1
df_model["day_of_week"] = df_model.index.dayofweek / 6.0  # Normalize 0-1

df_model.dropna(inplace=True)

print(f"\nFinal data after creating features: {len(df_model)} records")
print(f"Features: {[f'lag_{i}' for i in range(1, 7)] + ['hour_of_day', 'day_of_week']}")


# --------------------------------------------------------------------------- #
# Train model
# --------------------------------------------------------------------------- #

feature_cols = [f"lag_{i}" for i in range(1, 7)] + ["hour_of_day", "day_of_week"]
X = df_model[feature_cols]
y = df_model["Energy_kW"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
print("\nTraining RandomForest model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nModel MSE: {mse:.4f}")
print(f"Model RMSE: {np.sqrt(mse):.4f} kW")


# --------------------------------------------------------------------------- #
# Export to ONNX
# --------------------------------------------------------------------------- #

print("\nConverting model to ONNX...")
# 8 features: 6 lags + hour + day
initial_type = [("float_input", FloatTensorType([None, 8]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

model_path = "../Models/willow_energy_10min.onnx"
with open(model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Model converted to ONNX: {model_path}")

# Verify ONNX model
session = rt.InferenceSession(model_path)
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


# --------------------------------------------------------------------------- #
# Plot results
# --------------------------------------------------------------------------- #

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
plt.savefig("../Logs/forecast_comparison_plot_8input.png")
print("\nSaved: ../Logs/forecast_comparison_plot_8input.png")

plt.figure(figsize=(15, 5))
plt.plot(y_test_series.index, y_test_series, label="Actual", color="blue")
plt.plot(y_test_series.index, y_pred, label="Predicted", color="orange")

plt.title("Actual vs Predicted Energy Usage (Sklearn Only - 8 Input Model)")
plt.xlabel("Datetime")
plt.ylabel("Kilowatts (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Logs/forecast_comparison_plot_sklearn_only_8input.png")
print("Saved: ../Logs/forecast_comparison_plot_sklearn_only_8input.png")

print("\n" + "=" * 70)
print("Model training completed successfully!")
print(f"Output model: {model_path}")
print("=" * 70)

