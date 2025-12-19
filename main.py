import os
import yaml
import argparse
import pandas as pd
import numpy as np
from src.data_loader import create_returns_matrix, load_raw_data
from src.features import generate_technical_features, apply_triple_barrier_labeling
from src.models import XGBoostTrainer, PortfolioOptimizer

def run_pipeline(mode):
    # load configuration
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # select specific config based on mode
    if mode == 'btc':
        run_config = config['btc_trading']
        print(f"- - - mode: bitcoin trading research - - -")
    elif mode == 'procurement':
        run_config = config['procurement_optimization']
        print(f"- - - mode: procurement cost optimization - - -")
    else:
        raise ValueError("invalid mode. choose 'btc' or 'procurement'.")

    # 1. data ingestion
    df = load_raw_data(run_config['data']['raw_path'])
    if df is None and mode == 'btc': # btc mode requires this df
        print("critical error: could not load raw data for btc mode.")
        return
    # 2. logic branch based on the study type
    if mode == 'btc':
        # technical features & triple barrier labeling
        df_features = generate_technical_features(df)
        df_final = apply_triple_barrier_labeling(
            df_features,
            profit_pct=run_config['labeling']['profit_margin'],
            loss_pct=run_config['labeling']['stop_loss'],
            days=run_config['labeling']['barrier_days']
        )
        
        # training with xgboost classifier
        trainer = XGBoostTrainer(df_final, features=run_config['features']['model_cols'])
        trainer.prepare_data(test_size=0.2)
        trainer.train()
        trainer.evaluate()
    else: # mode == 'procurement'
        # 1. build the returns matrix from all strategy exports 
        print("aligning strategies and creating returns matrix...")
        returns_matrix = create_returns_matrix(run_config['data']['raw_path'])
        if returns_matrix.empty:
            print("\ncritical error: no data available for optimization. check csv headers and paths.")
            return # stop pipeline execution
            
        optimizer = PortfolioOptimizer(returns_matrix, total_budget=run_config['optimization']['budget'])
        optimal_weights_dict = optimizer.optimize()
        
        if not optimal_weights_dict:
            print("optimization failed to return weights.")
            return
        # 2. run optimizer
        # total_budget defined in params.yaml or hardcoded as 25
        optimizer = PortfolioOptimizer(returns_matrix, total_budget=25)
        optimal_weights_dict = optimizer.optimize()
        
        # 3. create df_final to fix the UnboundLocalError
        df_final = pd.DataFrame(list(optimal_weights_dict.items()), columns=['strategy', 'optimal_weight'])
        
        print("\n- - - optimal strategy weights (linearized) - - -")
        # filter only strategies with significant weight
        active_allocations = df_final[df_final['optimal_weight'] > 0.001]
        print(active_allocations.to_string(index=False))

        # calculate final linearity score (r-squared of cumulative equity)
        portfolio_returns = (returns_matrix * optimizer.weights).sum(axis=1)
        cumulative_equity = portfolio_returns.cumsum()
        
        # simple r-squared calculation
        y = cumulative_equity.values
        x = np.arange(len(y)).reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(x, y)
        r2 = model.score(x, y)
        print(f"\nfinal portfolio linearity (r-squared): {r2:.4f}")

    # 3. persistence
    os.makedirs(os.path.dirname(run_config['data']['processed_path']), exist_ok=True)
    df_final.to_csv(run_config['data']['processed_path'])
    print(f"\npipeline complete. data saved at {run_config['data']['processed_path']}")

if __name__ == "__main__":
    # setup command line arguments
    parser = argparse.ArgumentParser(description="quantitative research platform")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=['btc', 'procurement'],
        help="choose the study mode: 'btc' for trading or 'procurement' for cost optimization"
    )
    
    args = parser.parse_args()
    run_pipeline(args.mode)