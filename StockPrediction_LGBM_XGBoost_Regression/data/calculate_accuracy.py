#!/usr/bin/env python3
"""
Calculate accuracy for each stock's XGBoost and LightGBM models

This script reads prediction results from the prediction_results directory
and calculates accuracy metrics for each stock.
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_accuracy(results_file):
    """
    Calculate accuracy metrics for a single stock's prediction results
    
    Args:
        results_file: Path to the prediction results file
    
    Returns:
        Dictionary containing accuracy metrics
    """
    # Read results file
    df = pd.read_csv(results_file)
    
    # Calculate metrics for XGBoost
    xgb_rmse = np.sqrt(mean_squared_error(df['actual'], df['xgboost_pred']))
    xgb_mae = mean_absolute_error(df['actual'], df['xgboost_pred'])
    xgb_r2 = r2_score(df['actual'], df['xgboost_pred'])
    
    # Calculate metrics for LightGBM
    lgb_rmse = np.sqrt(mean_squared_error(df['actual'], df['lightgbm_pred']))
    lgb_mae = mean_absolute_error(df['actual'], df['lightgbm_pred'])
    lgb_r2 = r2_score(df['actual'], df['lightgbm_pred'])
    
    # Calculate directional accuracy
    xgb_directional = np.mean(np.sign(df['actual'].diff()) == np.sign(df['xgboost_pred'].diff()))
    lgb_directional = np.mean(np.sign(df['actual'].diff()) == np.sign(df['lightgbm_pred'].diff()))
    
    return {
        'xgb_rmse': xgb_rmse,
        'xgb_mae': xgb_mae,
        'xgb_r2': xgb_r2,
        'xgb_directional': xgb_directional,
        'lgb_rmse': lgb_rmse,
        'lgb_mae': lgb_mae,
        'lgb_r2': lgb_r2,
        'lgb_directional': lgb_directional
    }

def main():
    """
    Main function to calculate accuracy for all stocks
    """
    # Get prediction results directory
    results_dir = 'prediction_results'
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Results directory {results_dir} not found")
        return
    
    # Get all results files
    results_files = [f for f in os.listdir(results_dir) if f.startswith('prediction_results_') and f.endswith('.csv')]
    
    if not results_files:
        print(f"Error: No results files found in {results_dir}")
        return
    
    print(f"Found {len(results_files)} results files to process")
    
    # Create a list to store accuracy results
    accuracy_results = []
    
    # Process each results file
    for results_file in results_files:
        # Extract stock name from filename
        stock_name = results_file.replace('prediction_results_', '').replace('.csv', '')
        file_path = os.path.join(results_dir, results_file)
        
        try:
            # Calculate accuracy
            metrics = calculate_accuracy(file_path)
            
            # Add to results list
            accuracy_results.append({
                'stock_name': stock_name,
                **metrics
            })
            
            print(f"Processed {stock_name}")
            
        except Exception as e:
            print(f"Error processing {stock_name}: {str(e)}")
    
    # Create DataFrame from results
    accuracy_df = pd.DataFrame(accuracy_results)
    
    # Sort by stock name
    accuracy_df = accuracy_df.sort_values('stock_name').reset_index(drop=True)
    
    # Add average row
    avg_row = {
        'stock_name': 'Average',
        'xgb_rmse': accuracy_df['xgb_rmse'].mean(),
        'xgb_mae': accuracy_df['xgb_mae'].mean(),
        'xgb_r2': accuracy_df['xgb_r2'].mean(),
        'xgb_directional': accuracy_df['xgb_directional'].mean(),
        'lgb_rmse': accuracy_df['lgb_rmse'].mean(),
        'lgb_mae': accuracy_df['lgb_mae'].mean(),
        'lgb_r2': accuracy_df['lgb_r2'].mean(),
        'lgb_directional': accuracy_df['lgb_directional'].mean()
    }
    
    accuracy_df = pd.concat([accuracy_df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Save to CSV
    output_file = 'stock_model_accuracy.csv'
    accuracy_df.to_csv(output_file, index=False)
    print(f"\nAccuracy results saved to {output_file}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total stocks processed: {len(accuracy_results)}")
    print(f"Average XGBoost RMSE: {avg_row['xgb_rmse']:.4f}")
    print(f"Average LightGBM RMSE: {avg_row['lgb_rmse']:.4f}")
    print(f"Average XGBoost R2: {avg_row['xgb_r2']:.4f}")
    print(f"Average LightGBM R2: {avg_row['lgb_r2']:.4f}")
    print(f"Average XGBoost Directional Accuracy: {avg_row['xgb_directional']:.4f}")
    print(f"Average LightGBM Directional Accuracy: {avg_row['lgb_directional']:.4f}")

if __name__ == "__main__":
    main()