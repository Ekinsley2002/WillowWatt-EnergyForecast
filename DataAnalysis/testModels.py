# Model Testing Script
# Tests all ONNX models in the Models folder with CSV data

import pandas as pd
import numpy as np
import onnxruntime as rt
import os

def load_csv_data():
    """Load and prepare CSV test data"""
    print("Loading CSV test data...")
    
    # Load CSV data
    csv_data = pd.read_csv("../Data/09-06-2024 -- 09-06-2025.csv")
    
    # Prepare data
    csv_data['Datetime'] = pd.to_datetime(csv_data['Time'])
    csv_data = csv_data.set_index('Datetime')
    csv_data = csv_data.rename(columns={'Average': 'Energy_kWh'})
    
    # Filter outliers (keep values between 0 and 100000 for kW data)
    csv_data = csv_data[(csv_data['Energy_kWh'] >= 0) & (csv_data['Energy_kWh'] < 100000)]
    
    # Resample to hourly and create features
    df = csv_data[['Energy_kWh']].resample('h').mean().ffill()
    
    # Create features
    df['prev_hour'] = df['Energy_kWh'].shift(1)
    df['prev_day'] = df['Energy_kWh'].shift(24)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['avg_3h'] = df['Energy_kWh'].rolling(window=3).mean()
    df['avg_24h'] = df['Energy_kWh'].rolling(window=24).mean()
    df.dropna(inplace=True)
    
    print(f"  Test data shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    return df

def test_model(model_path, test_data):
    """Test a single ONNX model with test data"""
    try:
        # Load ONNX model
        session = rt.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Prepare test features
        X = test_data[['prev_hour', 'prev_day', 'hour', 'day_of_week', 'month', 'avg_3h', 'avg_24h']].values
        y_true = test_data['Energy_kWh'].values
        
        # Make predictions
        predictions = []
        for i in range(len(X)):
            input_data = X[i:i+1].astype(np.float32)
            pred = session.run([output_name], {input_name: input_data})[0]
            predictions.append(pred[0])
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        mse = np.mean((y_true - predictions) ** 2)
        mae = np.mean(np.abs(y_true - predictions))
        r2 = 1 - (np.sum((y_true - predictions) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'actual': y_true
        }
        
    except Exception as e:
        print(f"  Error testing model: {e}")
        return None

def main():
    """Main testing function"""
    print("="*60)
    print("TESTING ENERGY PREDICTION MODELS")
    print("="*60)
    
    # Load test data
    test_data = load_csv_data()
    
    # Get all ONNX models in Models folder
    models_dir = "../Models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]
    
    print(f"\nFound {len(model_files)} models to test:")
    for model_file in model_files:
        print(f"  - {model_file}")
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    # Test each model
    results = {}
    for model_file in sorted(model_files):
        model_path = os.path.join(models_dir, model_file)
        model_name = model_file.replace('.onnx', '')
        
        print(f"\nTesting {model_name}...")
        
        result = test_model(model_path, test_data)
        if result:
            results[model_name] = result
            print(f"  R² Score: {result['r2']:.4f}")
            print(f"  MSE: {result['mse']:.2f}")
            print(f"  MAE: {result['mae']:.2f}")
        else:
            print(f"  Failed to test model")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results:
        # Sort by R² score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        print("\nModels ranked by R² Score:")
        for i, (model_name, result) in enumerate(sorted_results, 1):
            print(f"{i}. {model_name}: R² = {result['r2']:.4f}, MSE = {result['mse']:.2f}")
    
    print(f"\nTesting complete. {len(results)}/{len(model_files)} models tested successfully.")

if __name__ == "__main__":
    main()

