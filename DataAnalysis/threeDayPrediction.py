# Energy Prediction Models
# Creates 5 clean models that take one parameter and output 24-hour predictions
# 
# Input Parameters:
# - CurrentLoad: Current energy load in kWh (float)
# 
# Output:
# - 24 one-hour predictions for the next 24 hours
# 
# Models Created:
# 1. Building 28 model
# 2. Building 36 model  
# 3. Building 54 model
# 4. WillowEnergyData model (09-06-2024 -- 09-06-2025.csv)
# 5. Combined model (all data sources)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def create_features(df):
    """Create simple lag feature for prediction (like reference code)"""
    df['prev_hour'] = df['Energy_kWh'].shift(1)
    df.dropna(inplace=True)
    return df

def prepare_data(source_data, source_name):
    """Prepare and clean data from any source"""
    print(f"\nProcessing {source_name}...")
    
    # Convert to datetime and set as index
    if 'Date' in source_data.columns:
        source_data['Datetime'] = pd.to_datetime(source_data['Date'])
        source_data = source_data.set_index('Datetime')
    elif 'Time' in source_data.columns:
        source_data['Datetime'] = pd.to_datetime(source_data['Time'])
        source_data = source_data.set_index('Datetime')
    
    # Rename energy column if needed
    if 'kWh' in source_data.columns:
        source_data = source_data.rename(columns={'kWh': 'Energy_kWh'})
    elif 'Average' in source_data.columns:
        source_data = source_data.rename(columns={'Average': 'Energy_kWh'})
    
    # Filter outliers based on data characteristics
    original_count = len(source_data)
    
    # Check data range to determine appropriate filtering
    max_val = source_data['Energy_kWh'].max()
    min_val = source_data['Energy_kWh'].min()
    
    if max_val > 10000:  # Likely kW data (larger values)
        source_data = source_data[(source_data['Energy_kWh'] >= 0) & (source_data['Energy_kWh'] < 100000)]
    else:  # Likely kWh data (smaller values)
        source_data = source_data[(source_data['Energy_kWh'] >= 0) & (source_data['Energy_kWh'] < 1000)]
    
    filtered_count = len(source_data)
    
    print(f"  Removed {original_count - filtered_count} outliers ({filtered_count}/{original_count} rows kept)")
    
    # Resample to hourly data
    df = source_data[['Energy_kWh']].resample('h').mean().ffill()
    
    # Create features
    df = create_features(df)
    
    print(f"  Final data shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Energy range: {df['Energy_kWh'].min():.2f} to {df['Energy_kWh'].max():.2f} kWh")
    
    return df

def train_model(df, model_name):
    """Train Random Forest model on prepared data (like reference code)"""
    print(f"\nTraining {model_name}...")
    
    # Prepare features and target (only prev_hour like reference code)
    X = df[['prev_hour']]
    y = df['Energy_kWh']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train model (same parameters as reference code)
    model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    # Test model
    y_pred = model.predict(X_test)
    r2_score = model.score(X_test, y_test)
    
    print(f"  R² Score: {r2_score:.4f}")
    print(f"  MSE: {np.mean((y_test - y_pred)**2):.4f}")
    
    return model, X.shape[1]

def save_onnx_model(model, input_features, model_name):
    """Convert and save model to ONNX format"""
    print(f"  Converting {model_name} to ONNX...")
    
    # Define input type for ONNX conversion
    initial_type = [('float_input', FloatTensorType([None, input_features]))]
    
    # Convert to ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    # Save ONNX model
    model_path = f"../Models/{model_name}.onnx"
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"  Saved: {model_path}")

def predict_24_hours(model, current_load):
    """
    Predict next 24 hours using only current load input (like reference code)
    
    Args:
        model: Trained RandomForestRegressor model
        current_load: Current energy load in kWh (float)
    
    Returns:
        List of 24 predictions for next 24 hours
    """
    # Convert Willow's kW input to kWh
    current_load_kwh = current_load / 12
    
    predictions = []
    next_hour_load = current_load_kwh
    
    for hour in range(24):
        # Predict next hour using current load (use DataFrame format to avoid warnings)
        input_df = pd.DataFrame({'prev_hour': [next_hour_load]})
        prediction = model.predict(input_df)[0]
        predictions.append(prediction)
        
        # Use prediction as input for next hour
        next_hour_load = prediction
    
    return predictions

# Load data from Excel file (Buildings 28, 36, 54)
print("Loading building data from Excel...")
building_28 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-28")
building_36 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-36")
building_54 = pd.read_excel("../Data/Historic-15MIN.xlsx", sheet_name="BLDG-54")

# Load CSV data
print("Loading CSV data...")
csv_data = pd.read_csv("../Data/09-06-2024 -- 09-06-2025.csv")

print("="*60)
print("CREATING ENERGY PREDICTION MODELS")
print("="*60)

# Create and train individual building models
models = {}

# Building 28 model
df_28 = prepare_data(building_28, "Building 28")
model_28, features_28 = train_model(df_28, "Building 28 Model")
save_onnx_model(model_28, features_28, "building_28_model")
models['building_28'] = model_28

# Building 36 model
df_36 = prepare_data(building_36, "Building 36")
model_36, features_36 = train_model(df_36, "Building 36 Model")
save_onnx_model(model_36, features_36, "building_36_model")
models['building_36'] = model_36

# Building 54 model
df_54 = prepare_data(building_54, "Building 54")
model_54, features_54 = train_model(df_54, "Building 54 Model")
save_onnx_model(model_54, features_54, "building_54_model")
models['building_54'] = model_54

# CSV data model
df_csv = prepare_data(csv_data, "CSV Data")
model_csv, features_csv = train_model(df_csv, "Willow Energy Data")
save_onnx_model(model_csv, features_csv, "willow_energy_data")
models['csv_data'] = model_csv

# Combined model (all data sources)
print(f"\nProcessing Combined Data...")
combined_data = []

# Add building data
for building, name in zip([df_28, df_36, df_54], ['BLDG_28', 'BLDG_36', 'BLDG_54']):
    building_copy = building.copy()
    building_copy['Building'] = name
    combined_data.append(building_copy)

# Add CSV data
csv_copy = df_csv.copy()
csv_copy['Building'] = 'CSV_DATA'
combined_data.append(csv_copy)

# Combine all data
df_combined = pd.concat(combined_data, ignore_index=False)
df_combined = df_combined[['Energy_kWh']].resample('h').mean().ffill()
df_combined = create_features(df_combined)

print(f"  Combined data shape: {df_combined.shape}")
print(f"  Date range: {df_combined.index.min()} to {df_combined.index.max()}")

# Train combined model
model_combined, features_combined = train_model(df_combined, "Combined Model")
save_onnx_model(model_combined, features_combined, "combined_model")
models['combined'] = model_combined

print("\n" + "="*60)
print("ALL MODELS CREATED SUCCESSFULLY!")
print("="*60)

print("\nCreated Models:")
print("1. building_28_model.onnx - Uses Building 28 data only")
print("2. building_36_model.onnx - Uses Building 36 data only") 
print("3. building_54_model.onnx - Uses Building 54 data only")
print("4. WillowEnergyData.onnx - Uses CSV data only")
print("5. combined_model.onnx - Uses all data sources")

print(f"\nEach model takes 1 input parameter:")
print(f"- CurrentLoad: Current energy load in kWh (float)")
print(f"\nEach model outputs 24 one-hour predictions for the next 24 hours")

print(f"\nAll models saved to: ../Models/")

# Example usage: Predict 24 hours using current load
print("\n" + "="*60)
print("EXAMPLE USAGE")
print("="*60)

# Example current load (in kWh)
example_current_load = 150.5  # Example: 150.5 kWh current load

print(f"\nExample: Predicting next 24 hours with current load = {example_current_load} kWh")
print("\nUsing Building 28 Model:")
building_28_predictions = predict_24_hours(model_28, example_current_load)
for i, pred in enumerate(building_28_predictions):
    print(f"  Hour {i+1:2d}: {pred:.2f} kWh")

print(f"\nUsing Combined Model:")
combined_predictions = predict_24_hours(model_combined, example_current_load)
for i, pred in enumerate(combined_predictions):
    print(f"  Hour {i+1:2d}: {pred:.2f} kWh")

# Example current load in kW (simulate Willow input)
example_current_load_kw = 150.5  # Example: 150.5 kW

print(f"\nExample: Predicting next 24 hours with current load = {example_current_load_kw} kW")

print("\nUsing North Campus Model:")
north_predictions_kw = predict_24_hours(model_csv, example_current_load_kw)

for i, pred in enumerate(north_predictions_kw):
    print(f"  Hour {i+1:2d}: {pred:.2f} kW")