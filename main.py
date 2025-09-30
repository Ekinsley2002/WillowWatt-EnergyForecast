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
    # Use the converted ONNX model and run fast predictions outside of scikit-learn
import onnxruntime as rt

# Import numpy which is a core library for numerical operations in python
# Why I use Numpy:
    # To convert test data into a format (float32 arrays) that ONNX understands
    # To quickly handle math operations needed for model evaluation
import numpy as np

# Pull data
df = pd.read_csv('Data/09-06-2024 -- 09-06-2025.csv')

df['Time'] = pd.to_datetime(df['Time'].str.strip(), format='%m/%d/%Y %I:%M:%S %p')

# Divide by 1000 to get MW Multiply by 1000 to get Watts
averageEnergy = df['Average']/1000 # This is the average energy usage in 12 hour intervals

maximumEnergy = df['Maximum']/1000 # This is the maximum energy usage in 12 hour intervals

minimumEnergy = df['Minimum']/1000 # This is the minimum energy usage in 12 hour intervals

# Add time-based features to look for in the model
df['hour'] = pd.to_datetime(df['Time']).dt.hour
df['day_of_week'] = pd.to_datetime(df['Time']).dt.dayofweek  # 0=Monday, 6=Sunday
df['month'] = pd.to_datetime(df['Time']).dt.month
df['day_of_year'] = pd.to_datetime(df['Time']).dt.dayofyear

# Create lag features (previous day's energy)
df['prev_day_avg'] = df['Average'].shift(2)  # Since you have 12-hour intervals
df['prev_day_max'] = df['Maximum'].shift(2)
df['prev_day_min'] = df['Minimum'].shift(2)

# Define features (what the model will look for)
feature_columns = ['hour', 'day_of_week', 'month', 'day_of_year', 
                   'prev_day_avg', 'prev_day_max', 'prev_day_min']
X = df[feature_columns]
y = df['Maximum']  # Predicting the maximum energy for each period

# Split your data (use last 20% for testing)
split_point = int(len(df) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

import datetime

start_date = datetime.datetime(2025, 9, 15, 5, 0, 0)
future_dates = []

for i in range(14):
    if i % 2 == 0:
        future_dates.append(start_date + datetime.timedelta(days=i//2, hours=0))
    else:
        future_dates.append(start_date + datetime.timedelta(days=i//2, hours=12))

future_df = pd.DataFrame({'Time': future_dates})

future_df['hour'] = future_df['Time'].dt.hour
future_df['day_of_week'] = future_df['Time'].dt.dayofweek
future_df['month'] = future_df['Time'].dt.month
future_df['day_of_year'] = future_df['Time'].dt.dayofyear

last_avg = df['Average'].iloc[-1]
last_max = df['Maximum'].iloc[-1] 
last_min = df['Minimum'].iloc[-1]

future_df['prev_day_avg'] = last_avg
future_df['prev_day_max'] = last_max
future_df['prev_day_min'] = last_min

future_features = future_df[feature_columns]
future_predictions = model.predict(future_features)

max_prediction_idx = future_predictions.argmax()
max_prediction_value = future_predictions.max()

predicted_date = future_df.iloc[max_prediction_idx]['Time']
day_name = predicted_date.strftime('%A')
time_period = "AM" if predicted_date.hour < 12 else "PM"

total_days_predicted = len(future_df)

print('--- Prediction Summary ---\n')
print(f'Highest predicted peak: {max_prediction_value:.2f} MW')
print(f'Predicted for: {day_name} {time_period} ({predicted_date})')
print(f'This is data point {max_prediction_idx} in the next week')
print('\n--------------------------')

# User I/O for further data
print(f'\nWould you like to see the full {total_days_predicted} day prediction? (y/n) : ', end='')
user_input = input().strip().lower()
if user_input == 'y':
    for idx, (time, pred) in enumerate(zip(future_df['Time'], future_predictions)):
        day_name = time.strftime('%A')
        time_period = "AM" if time.hour < 12 else "PM"
        print(f'Day {idx+1}: {day_name} {time_period} ({time}) - Predicted Peak: {pred:.2f} MW')
else:
    print('Exiting without showing full prediction.')