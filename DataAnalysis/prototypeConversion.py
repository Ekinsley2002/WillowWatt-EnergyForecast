'''
Welcome! To begin using my program you will want to:
First install and integrate Python 3.10 as your interpreter

Next pip install the following:
pip install pandas numpy scikit-learn matplotlib onnx onnxruntime skl2onnx
'''

# Import pandas for loading, manipulating and analyzing AEP_hourly excel
# Why I use pandas:
    # Load energy dataset
    # Set timestamps as an index
    # resample and clean the time-series
import pandas as pd

# Import matplotlib for data visualization
# Why I use matplotlib:
    # Plot actual vs predicted energy usage
    # Visualize patterns in the energy data
    # Understanding how accurate my forcasting is
import matplotlib.pyplot as plt

# Import train_test_split which splits my data into training and testing sets
# Why I use train_test_split
    # I split the data up so that the model can use some of the data to train->
    # ->and other parts of the data for testing accuracy
from sklearn.model_selection import train_test_split

# Import randomForestRegressor which is a machine learning model that uses multiple->
# ->Decision trees to make predictions
# Why I use randomForestRegressor
    # To forcast the next hour of energuy usage based on the previous hour
    # This kind of machine learning model works well with numbers and is robust
from sklearn.ensemble import RandomForestRegressor

# Import mean_squared_error which is a standard math algorithm that tests for "Loss"
# Why I use mean_squared_error:
    # Measuring loss is important in AI models to know how wrong the prediction is
from sklearn.metrics import mean_squared_error

# Import convert_sklearn to convert the trained model into ONNX format
# Why I use convert_sklearn:
    # I need to export to ONNX format (recommended by client) to have a cross ->
    #->platform compatable model so that it has more multipurpose.
from skl2onnx import convert_sklearn

# Import FloatTensorType to define the input type for the ONNX conversion
# Why I use FloatTensorType:
    # ONNX needs to know the input shape (how many features) to build the graph correctly
from skl2onnx.common.data_types import FloatTensorType

# Import onnxruntime which is an runtime engine to load and run ONNX models
# Why I use onnxruntimeL
    # Us ethe converted ONNX model and run fast predictions outside of scikit-learn
import onnxruntime as rt

# Import numpy which is a core library for numerical operations in python
# Why I use Numpy:
    # To convert test data into a format (float32 arrays) that ONNX understands
    # To quickly handle math operations needed for model evaluation
import numpy as np

# Load historic building data
try:
    building_28 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-28")
    building_54 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-54")
    building_36 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-36")
    
    print("Historic building data loaded successfully!")
    
    # Clean and prepare building 28 data (main dataset)
    building_28 = building_28.dropna(subset=['Date', 'kWh']).reset_index(drop=True)
    building_28['Time'] = pd.to_datetime(building_28['Date'])
    
    # Filter out placeholder values (1.0)
    building_28 = building_28[building_28['kWh'] != 1.0]
    
    # Use building 28 as our main dataset
    df = building_28.copy()
    df = df.rename(columns={'Time': 'Datetime', 'kWh': 'Energy_kWh'})
    
    print(f"Building 28 data shape: {df.shape}")
    print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    print(f"Energy range: {df['Energy_kWh'].min():.2f} to {df['Energy_kWh'].max():.2f} kWh")
    
except Exception as e:
    print(f"Error loading historic data: {e}")
    quit()

# STEP 1. Preprocess the data by identifying and initializing column data
df.set_index('Datetime', inplace=True)
df = df.resample('h').mean().ffill()

# Step 2. Visualize time series by plotting the data frame I got in the previous step
df.plot(figsize=(15, 5), title="Building 28 Historic Energy Usage")
plt.ylabel("Energy (kWh)")
plt.xlabel("Datetime")
plt.grid()
plt.tight_layout()
plt.savefig("../DataAnalysis/Figures/historic_data_overview_plot.png")
print("Saved: historic_data_overview_plot.png")

# Create lag features for time-series forecasting
df['prev_hour'] = df['Energy_kWh'].shift(1)
df['prev_day'] = df['Energy_kWh'].shift(24)  # Previous day at same hour
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

# Add rolling averages
df['avg_3h'] = df['Energy_kWh'].rolling(window=3).mean()
df['avg_24h'] = df['Energy_kWh'].rolling(window=24).mean()

df.dropna(inplace=True)

X = df[['prev_hour', 'prev_day', 'hour', 'day_of_week', 'month', 'avg_3h', 'avg_24h']]
y = df['Energy_kWh']

# Split data into training and testing sections
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train optimized Random Forest model
model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)
print("Sklearn Model MSE:", mean_squared_error(y_test, y_pred))
print(f"Model R² Score: {model.score(X_test, y_test):.4f}")

# Generate predictions for the next 3 days (72 hours)
print("\n" + "="*50)
print("PREDICTING NEXT 3 DAYS (72 HOURS)")
print("="*50)

# Get the last known data point
last_data = df.iloc[-1].copy()
last_time = df.index[-1]

# Create future dates (next 72 hours)
future_dates = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=72, freq='H')

# Prepare features for future predictions
future_predictions = []
current_features = last_data[['prev_hour', 'prev_day', 'hour', 'day_of_week', 'month', 'avg_3h', 'avg_24h']].copy()

for i, future_date in enumerate(future_dates):
    # Update time-based features
    current_features['hour'] = future_date.hour
    current_features['day_of_week'] = future_date.dayofweek
    current_features['month'] = future_date.month
    
    # Make prediction using DataFrame to avoid warnings
    pred = model.predict(current_features)[0]
    future_predictions.append(pred)
    
    # Update lag features for next prediction
    current_features['prev_hour'] = pred  # prev_hour becomes current prediction
    current_features['prev_day'] = pred  # prev_day (simplified for this example)
    
    # Update rolling averages (simplified)
    current_features['avg_3h'] = pred  # avg_3h
    current_features['avg_24h'] = pred  # avg_24h

# Display predictions
print(f"Starting from: {last_time}")
print(f"Predicting until: {future_dates[-1]}")
print("\nDaily Summaries:")
print("-" * 50)

# Group predictions by day
for day_offset in range(3):
    day_start = day_offset * 24
    day_end = (day_offset + 1) * 24
    day_predictions = future_predictions[day_start:day_end]
    day_dates = future_dates[day_start:day_end]
    
    avg_consumption = np.mean(day_predictions)
    max_consumption = np.max(day_predictions)
    min_consumption = np.min(day_predictions)
    
    print(f"Day {day_offset + 1} ({day_dates[0].strftime('%Y-%m-%d')}):")
    print(f"  Average: {avg_consumption:.2f} kWh")
    print(f"  Maximum: {max_consumption:.2f} kWh")
    print(f"  Minimum: {min_consumption:.2f} kWh")
    print()

print("Hourly Predictions (first 24 hours):")
print("-" * 50)
for i in range(24):
    print(f"{future_dates[i].strftime('%Y-%m-%d %H:%M')}: {future_predictions[i]:.2f} kWh")

print(f"\n... and {48} more hours of predictions")
print("="*50)

# Plot actual vs predicted for model validation
y_test_series = y_test.copy()
y_test_series.index = X_test.index

plt.figure(figsize=(15, 8))
plt.plot(y_test_series.index, y_test_series, label='Actual', color='blue', alpha=0.7)
plt.plot(y_test_series.index, y_pred, label='Predicted', color='orange', alpha=0.7)

plt.title('Model Validation: Actual vs Predicted Energy Usage')
plt.xlabel('Datetime')
plt.ylabel('Energy (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../DataAnalysis/Figures/historic_forecast_validation.png")
print("Saved: historic_forecast_validation.png")

# Plot the future predictions
plt.figure(figsize=(15, 8))
# Plot last 168 hours (1 week) of actual data
recent_data = df.tail(168)
plt.plot(recent_data.index, recent_data['Energy_kWh'], label='Historical Data (Last Week)', color='blue', alpha=0.7)

# Plot future predictions
plt.plot(future_dates, future_predictions, label='Future Predictions (Next 3 Days)', color='red', linewidth=2)

plt.title('Building 28: Historical Data and 3-Day Forecast')
plt.xlabel('Datetime')
plt.ylabel('Energy (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../DataAnalysis/Figures/historic_3day_forecast.png")
print("Saved: historic_3day_forecast.png")

