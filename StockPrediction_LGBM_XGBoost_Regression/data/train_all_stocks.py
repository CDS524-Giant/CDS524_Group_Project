#!/usr/bin/env python3
"""
Train all stock files in training_data directory

This script trains XGBoost and LightGBM models for each stock file in the training_data directory
and saves prediction results for each stock.
"""

import pandas as pd
import numpy as np
import os
import sys
from data_preprocessing import load_and_preprocess_data, create_features
from rolling_window import generate_rolling_windows, prepare_rolling_window_data, prepare_test_data
from xgboost_model import train_xgboost, predict_xgboost, evaluate_model as evaluate_xgboost
from lightgbm_model import train_lightgbm, predict_lightgbm, evaluate_model as evaluate_lightgbm

def train_single_stock(file_path, stock_name):
    """
    Train models for a single stock file
    
    Args:
        file_path: Path to the stock file
        stock_name: Name of the stock
    """
    try:
        print(f"\n=== Training {stock_name} ===")
        print(f"Loading data from {file_path}...")
        
        # Load and preprocess data
        df = load_and_preprocess_data(file_path)
        df = create_features(df)
        
        print(f"Data loaded successfully, total {len(df)} records")
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        # Generate rolling windows
        windows, test_start = generate_rolling_windows()
        
        print(f"\nGenerated {len(windows)} rolling windows")
        
        # Store performance metrics
        xgboost_metrics = []
        lightgbm_metrics = []
        
        print("\nStarting rolling window training...")
        
        for i, window in enumerate(windows):
            print(f"\nWindow {i+1}/{len(windows)}")
            print(f"Training set: {window['train_start'].date()} to {window['train_end'].date()}")
            print(f"Validation set: {window['val_start'].date()} to {window['val_end'].date()}")
            
            # Prepare data
            X_train, y_train, X_val, y_val = prepare_rolling_window_data(df, window)
            
            if len(X_train) == 0 or len(X_val) == 0:
                print("Warning: Insufficient data for current window, skipping")
                continue
            
            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            
            # Train XGBoost model
            xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
            xgb_pred = predict_xgboost(xgb_model, X_val)
            xgb_metric = evaluate_xgboost(y_val, xgb_pred)
            xgboost_metrics.append(xgb_metric)
            
            # Train LightGBM model
            lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
            lgb_pred = predict_lightgbm(lgb_model, X_val)
            lgb_metric = evaluate_lightgbm(y_val, lgb_pred)
            lightgbm_metrics.append(lgb_metric)
            
            # Print current window performance
            print(f"XGBoost RMSE: {xgb_metric['rmse']:.4f}, R2: {xgb_metric['r2']:.4f}")
            print(f"LightGBM RMSE: {lgb_metric['rmse']:.4f}, R2: {lgb_metric['r2']:.4f}")
        
        if not xgboost_metrics or not lightgbm_metrics:
            print(f"Error: Insufficient data for training {stock_name}")
            return
        
        # Calculate average performance
        avg_xgb = {k: np.mean([m[k] for m in xgboost_metrics]) for k in xgboost_metrics[0]}
        avg_lgb = {k: np.mean([m[k] for m in lightgbm_metrics]) for k in lightgbm_metrics[0]}
        
        print("\n=== Rolling Window Average Performance ===")
        print(f"XGBoost - RMSE: {avg_xgb['rmse']:.4f}, R2: {avg_xgb['r2']:.4f}")
        print(f"LightGBM - RMSE: {avg_lgb['rmse']:.4f}, R2: {avg_lgb['r2']:.4f}")
        
        # Prepare test set data
        X_test, y_test, test_dates = prepare_test_data(df, test_start)
        print(f"\nTest set samples: {len(X_test)}")
        print(f"Test set date range: {test_dates.min().date()} to {test_dates.max().date()}")
        
        if len(X_test) == 0:
            print(f"Error: No data in test set for {stock_name}")
            return
        
        # Use the last window's model for testing
        # Retrain the last window's model (using complete training data)
        last_window = windows[-1]
        X_train_final, y_train_final, _, _ = prepare_rolling_window_data(df, last_window)
        
        if len(X_train_final) == 0:
            print(f"Error: No data in final training set for {stock_name}")
            return
        
        # Train final models
        print("\nTraining final models...")
        final_xgb_model = train_xgboost(X_train_final, y_train_final, X_train_final, y_train_final)
        final_lgb_model = train_lightgbm(X_train_final, y_train_final, X_train_final, y_train_final)
        
        # Evaluate on test set
        xgb_test_pred = predict_xgboost(final_xgb_model, X_test)
        lgb_test_pred = predict_lightgbm(final_lgb_model, X_test)
        
        xgb_test_metric = evaluate_xgboost(y_test, xgb_test_pred)
        lgb_test_metric = evaluate_lightgbm(y_test, lgb_test_pred)
        
        print("\n=== Test Set Performance ===")
        print(f"XGBoost - RMSE: {xgb_test_metric['rmse']:.4f}, R2: {xgb_test_metric['r2']:.4f}")
        print(f"LightGBM - RMSE: {lgb_test_metric['rmse']:.4f}, R2: {lgb_test_metric['r2']:.4f}")
        
        # Compare the two algorithms
        if xgb_test_metric['rmse'] < lgb_test_metric['rmse']:
            print(f"\nXGBoost performs better on the test set for {stock_name}")
        else:
            print(f"\nLightGBM performs better on the test set for {stock_name}")
        
        # Save prediction results
        results = pd.DataFrame({
            'date': test_dates,
            'actual': y_test.values,
            'xgboost_pred': xgb_test_pred,
            'lightgbm_pred': lgb_test_pred
        })
        
        # Create results directory if it doesn't exist
        results_dir = 'prediction_results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Save results with stock name
        results_file = os.path.join(results_dir, f'prediction_results_{stock_name}.csv')
        results.to_csv(results_file, index=False)
        print(f"\nPrediction results saved to {results_file}")
        
        return True
        
    except Exception as e:
        print(f"Error training {stock_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to train all stocks
    """
    # Get training data directory
    training_dir = './Stock_predict_analytics_csv'
    
    # Check if training directory exists
    if not os.path.exists(training_dir):
        print(f"Error: Training directory {training_dir} not found")
        sys.exit(1)
    
    # Get all stock files
    stock_files = [f for f in os.listdir(training_dir) if f.startswith('Stock ') and f.endswith('.csv')]
    
    if not stock_files:
        print(f"Error: No stock files found in {training_dir}")
        sys.exit(1)
    
    print(f"Found {len(stock_files)} stock files to train")
    
    # Train each stock
    success_count = 0
    for stock_file in stock_files:
        # Extract stock name from filename
        stock_name = stock_file.replace('Stock ', '').replace('.csv', '')
        file_path = os.path.join(training_dir, stock_file)
        
        if train_single_stock(file_path, stock_name):
            success_count += 1
    
    print(f"\n=== Training Summary ===")
    print(f"Total stocks: {len(stock_files)}")
    print(f"Successfully trained: {success_count}")
    print(f"Failed: {len(stock_files) - success_count}")

if __name__ == "__main__":
    main()