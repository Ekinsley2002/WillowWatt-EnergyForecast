import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import os
import glob

data_folder = "../Data/WillowData - Weekly"
all_files = glob.glob(os.path.join(data_folder, "*.csv"))

print(f"Found {len(all_files)} CSV files")
dataframes = []

for file in all_files:
    print(f"Loading: {os.path.basename(file)}")
    df = pd.read_csv(file)
    
    df['Time'] = pd.to_datetime(df['Time'], format='mixed')
    
    df['Energy_kW'] = df['Average'] / 1000
    
    df = df[['Time', 'Energy_kW']].set_index('Time')
    
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=False)

combined_df = combined_df.sort_index()

print(f"\nTotal records: {len(combined_df)}")
print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

combined_df = combined_df.resample('h').mean()

combined_df = combined_df.ffill()

print(f"\nAfter resampling to hourly: {len(combined_df)} records")

combined_df['prev_hour'] = combined_df['Energy_kW'].shift(1)
combined_df.dropna(inplace=True)

print(f"\nFinal data after creating lag features: {len(combined_df)} records")

plt.figure(figsize=(15, 5))
plt.plot(combined_df.index, combined_df['Energy_kW'])
plt.title("Willow Energy Hourly Usage")
plt.ylabel("Kilowatts (kW)")
plt.xlabel("Datetime")
plt.grid()
plt.tight_layout()
plt.savefig("../Logs/data_overview_plot.png")
print("Saved: ../Logs/data_overview_plot.png")

X = combined_df[['prev_hour']]
y = combined_df['Energy_kW']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nModel MSE: {mean_squared_error(y_test, y_pred):.4f}")

initial_type = [('float_input', FloatTensorType([None, 1]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

model_path = "../Models/willow_energy_weekly.onnx"
with open(model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print(f"Model converted to ONNX: {model_path}")

sess = rt.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
test_input = X_test.to_numpy().astype(np.float32)
onnx_pred = sess.run([label_name], {input_name: test_input})[0]

print(f"ONNX Model MSE: {mean_squared_error(y_test, onnx_pred):.4f}")

y_test_series = y_test.copy()
y_test_series.index = X_test.index

plt.figure(figsize=(15, 5))
plt.plot(y_test_series.index, y_test_series, label='Actual', color='blue')
plt.plot(y_test_series.index, y_pred, label='Predicted (Sklearn)', color='orange')
plt.plot(y_test_series.index, onnx_pred, label='Predicted (ONNX)', color='green', linestyle='dashed')

plt.title('Actual vs Predicted Energy Usage (Sklearn & ONNX)')
plt.xlabel('Datetime')
plt.ylabel('Kilowatts (kW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Logs/forecast_comparison_plot.png")
print("Saved: ../Logs/forecast_comparison_plot.png")

plt.figure(figsize=(15, 5))
plt.plot(y_test_series.index, y_test_series, label='Actual', color='blue')
plt.plot(y_test_series.index, y_pred, label='Predicted', color='orange')

plt.title('Actual vs Predicted Energy Usage')
plt.xlabel('Datetime')
plt.ylabel('Kilowatts (kW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../Logs/forecast_comparison_plot_sklearn_only.png")
print("Saved: ../Logs/forecast_comparison_plot_sklearn_only.png")

print("\nModel training completed successfully!")
