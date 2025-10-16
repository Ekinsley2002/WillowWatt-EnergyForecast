# The purpose of this file is to create a model using three different buildings 28, 36, 54

# Input for creation: 3 seperate pages (one per building) of datetime (MM/DD/YY) and energy data (kWh)

# Process:
# Data format
#   - create data frames for each building
#   - format the datetime to (MM/DD/YY)
#   - define building names in relation to datafames
#   - pre process
# Visualize Time Series
#   - create plot of the average energy use across the three buildings
#   - create lag features ofr simple time series forecasting
# Create and Test model
#   - split the data into training and testing sections
#   - train RFR model
# Get predictions
#   - use the model (should be designed to only spit out three days) to get predictions
#   - show testing statistics (log to file)
# Convert to ONNYX
#   - package data into ONNYX
#   - store package in models file

# Where to go from here
# - Luke will use this three day forecaster for alert testing
# - Ayla will use this to compare with the next three days of API
# - OP Ravi will directly use this to plot the next three day predictions

# Input of Model:
# - The starting day to predict the next three (MM/DD/YY)

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

# Data format
try:
    building_28 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-28")

    building_54 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-54")

    building_36 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-36")

    buildings = [building_28, building_54, building_36]

except:
    print("Error finding data..") 

# GLOBAL

building_names = ['BLDG_28', 'BLDG_54', 'BLDG_36']

building_28['Time'] = pd.to_datetime(building_28['Date'])

building_54['Time'] = pd.to_datetime(building_54['Date'])

building_36['Time'] = pd.to_datetime(building_36['Date'])

# proprocess
building_54 = building_54.dropna(subset=['Time', 'kWh']).reset_index(drop=True)

# Loop through each building and filter out values >= 1000 AND negative values
for i, building in enumerate(buildings):
    # Check how many rows before filtering
    original_count = len(building)
    
    # Find outliers (for logging/inspection)
    outliers_high = building[building['kWh'] >= 1000]
    outliers_negative = building[building['kWh'] < 0]
    outlier_count = len(outliers_high) + len(outliers_negative)
    
    # Filter: keep only rows where kWh is between 0 and 1000 (exclusive)
    buildings[i] = building[(building['kWh'] >= 0) & (building['kWh'] < 1000)].reset_index(drop=True)
    
    # Print summary
    print(f"{building_names[i]}: Removed {outlier_count} outliers (kept {len(buildings[i])}/{original_count} rows)")
    
    if len(outliers_high) > 0:
        print(f"  High values removed: {len(outliers_high)} (max: {outliers_high['kWh'].max()})")
    if len(outliers_negative) > 0:
        print(f"  Negative values removed: {len(outliers_negative)} (min: {outliers_negative['kWh'].min()})")

# Update individual variables after filtering
building_28 = buildings[0]
building_54 = buildings[1]
building_36 = buildings[2]

# plot this all this data for all three buildings, then save the figures.
for building, name in zip(buildings, building_names):
    plt.figure(figsize=(12, 6))
    plt.plot(building['Time'], building['kWh'])  # Specify which columns to plot
    plt.title(f'Consumption of Energy Over Time: {name}')
    plt.xlabel('Date')
    plt.ylabel('kWh energy consumption')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'../DataAnalysis/Figures/{name}_TxkWh.png')
    plt.close()


