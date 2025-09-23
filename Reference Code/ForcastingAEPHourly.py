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

# First load and explore the dataset
try:
    df = pd.read_csv("AEP_hourly.csv")
    print("Dataset loaded successfully!")
    print(df.head())
    print(df.info())

except FileNotFoundError:
    print("Error: AEP_hourly.csv not found. Make sure it's in the same folder as this script.")
    quit()  # or use exit()

# STEP 1. Preprocess the data by identifying and initializing collumn data
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)
df = df.resample('h').mean().ffill()

# Step 2. Visualize time series by plotting the data frame I got in the previous step
df.plot(figsize=(15, 5), title="AEP Hourly Energy Usage")
plt.ylabel("Megawatts (MW)")
plt.xlabel("Datetime")
plt.grid()
plt.tight_layout()
plt.savefig("data_overview_plot.png")
print("Saved: data_overview_plot.png")

# Create lag feature for simple time-series forecasting by defining previous historic data
df['prev_hour'] = df['AEP_MW'].shift(1)
df.dropna(inplace=True)

X = df[['prev_hour']]
y = df['AEP_MW']

# Split data into training and testing sections
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train optimized Random Forest model
model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print("Sklearn Model MSE:", mean_squared_error(y_test, y_pred))

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, 1]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("aep_forecast_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model converted to ONNX!")

# Run inference with ONNX
sess = rt.InferenceSession("aep_forecast_model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
test_input = X_test.to_numpy().astype(np.float32)
onnx_pred = sess.run([label_name], {input_name: test_input})[0]

print("ONNX Model MSE:", mean_squared_error(y_test, onnx_pred))

# Plot actual vs predicted (Sklearn and ONNX)
y_test_series = y_test.copy()
y_test_series.index = X_test.index

plt.figure(figsize=(15, 5))
plt.plot(y_test_series.index, y_test_series, label='Actual', color='blue')
plt.plot(y_test_series.index, y_pred, label='Predicted (Sklearn)', color='orange')
plt.plot(y_test_series.index, onnx_pred, label='Predicted (ONNX)', color='green', linestyle='dashed')

plt.title('Actual vs Predicted Energy Usage (Sklearn & ONNX)')
plt.xlabel('Datetime')
plt.ylabel('Megawatts (MW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("forecast_comparison_plot.png")
print("Saved: forecast_comparison_plot.png")

# Plot just actual vs predicted Sklearn model for more clarity
plt.figure(figsize=(15, 5))
plt.plot(y_test_series.index, y_test_series, label='Actual', color='blue')
plt.plot(y_test_series.index, y_pred, label='Predicted (Sklearn)', color='orange')

plt.title('Actual vs Predicted Energy Usage (Sklearn)')
plt.xlabel('Datetime')
plt.ylabel('Megawatts (MW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("forecast_comparison_plot_Sklearn_only.png")
print("Saved: forecast_comparison_plot_Sklearn_only.png")

# Plot just actual vs predicted ONNX model for more clarity
plt.figure(figsize=(15, 5))
plt.plot(y_test_series.index, y_test_series, label='Actual', color='blue')
plt.plot(y_test_series.index, onnx_pred, label='Predicted (ONNX)', color='green', linestyle='dashed')

plt.title('Actual vs Predicted Energy Usage (ONNX)')
plt.xlabel('Datetime')
plt.ylabel('Megawatts (MW)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("forecast_comparison_plot_ONNX_only.png")
print("Saved: forecast_comparison_plot_ONNX_only.png")

