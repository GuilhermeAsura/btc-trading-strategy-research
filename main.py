import os
import yaml
import argparse
import pandas as pd
import numpy as np
from src.data_loader import create_returns_matrix, load_raw_data
from src.features import generate_technical_features, apply_triple_barrier_labeling
from src.models import XGBoostTrainer, PortfolioOptimizer, run_evolutionary_optimization 

def run_pipeline(mode):
    # load configuration
    config_path = "config/params.yaml" if os.path.exists("config/params.yaml") else "params.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # select specific config based on mode
    if mode == 'btc':
        run_config = config['btc_trading']
        print(f"- - - mode: bitcoin trading research - - -")
    elif mode == 'procurement':
        run_config = config['procurement_optimization']
        print(f"- - - mode: procurement cost optimization (linear) - - -")
    elif mode == 'genetic':
        run_config = config['procurement_optimization']
        ga_config = config['optimization_settings']
        print(f"- - - mode: evolutionary optimization (nsga-ii) - - -")
    else:
        raise ValueError("invalid mode. choose 'btc', 'procurement' or 'genetic'.")

    # 1. data ingestion
    if mode in ['procurement', 'genetic']:
        print("aligning strategies and creating returns matrix...")
        returns_matrix = create_returns_matrix(run_config['data']['raw_path'])
        print(f"matrix shape: {returns_matrix.shape}")
        if returns_matrix.empty:
            print("\ncritical error: no data available. check csv headers and paths.")
            return
    else:
        df = load_raw_data(run_config['data']['raw_path'])
        if df is None:
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
        trainer = XGBoostTrainer(df_final, features=run_config['features']['model_cols'])
        trainer.prepare_data(test_size=0.2)
        trainer.train()
        trainer.evaluate()

    elif mode == 'procurement':
        # standard linear optimization (slsqp)
        budget = run_config.get('constraints', {}).get('total_budget', 25)
        optimizer = PortfolioOptimizer(returns_matrix, total_budget=budget)
        optimal_weights_dict = optimizer.optimize()
        
        df_final = pd.DataFrame(list(optimal_weights_dict.items()), columns=['strategy', 'optimal_weight'])
        
        print("\n- - - optimal strategy weights (linearized) - - -")
        active_allocations = df_final[df_final['optimal_weight'] > 0.001]
        print(active_allocations.to_string(index=False))

        # calculate linearity (r-squared)
        portfolio_returns = (returns_matrix * optimizer.weights).sum(axis=1)
        cumulative_equity = portfolio_returns.cumsum()
        y = cumulative_equity.values
        x = np.arange(len(y)).reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        r2 = LinearRegression().fit(x, y).score(x, y)
        print(f"\nfinal portfolio linearity (r-squared): {r2:.4f}")

    elif mode == 'genetic':
        budget = run_config.get('constraints', {}).get('total_budget', 25)
        print(f"starting nsga-ii (pop: {ga_config['pop_size']}, gen: {ga_config['n_gen']}) with budget {budget}...")
        
        res = run_evolutionary_optimization(
            returns_matrix, 
            budget=budget, # pass the budget to the optimizer
            n_gen=ga_config['n_gen'], 
            pop_size=ga_config['pop_size']
        )
        
        # process results focusing on profit and linearity
        pareto_results = pd.DataFrame({
            'total_profit': -res.F[:, 0],
            'linearity_r2': 1 - res.F[:, 1]
        })
        
        print(f"\nevolution complete. {len(res.X)} solutions found.")
        # show top solutions by linearity
        print(pareto_results.sort_values('linearity_r2', ascending=False).head(5))
        
        # save and select best
        best_idx = pareto_results['linearity_r2'].idxmax()
        best_weights = res.X[best_idx]
        
        # apply the same integer logic for the final selection
        total_x = np.sum(best_weights)
        exact = best_weights * (budget / (total_x + 1e-9))
        final_ints = np.floor(exact).astype(int)
        rem = int(budget - np.sum(final_ints))
        if rem > 0:
            indices = np.argsort(exact - final_ints)[::-1][:rem]
            final_ints[indices] += 1
        
        df_final = pd.DataFrame({
            'strategy': returns_matrix.columns,
            'optimal_weight': final_ints # now perfectly integer
        })
        
        print("\n- - - final integer lot allocation (perfect budget) - - -")
        print(df_final[df_final['optimal_weight'] > 0].to_string(index=False))
        print(f"\ntotal lots allocated: {df_final['optimal_weight'].sum()}")

    # 3. persistence
    os.makedirs(os.path.dirname(run_config['data']['processed_path']), exist_ok=True)
    df_final.to_csv(run_config['data']['processed_path'], index=False)
    print(f"\npipeline complete. data saved at {run_config['data']['processed_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="quantitative research platform")
    parser.add_argument(
        "--mode", 
        type=str, 
        required=True, 
        choices=['btc', 'procurement', 'genetic'],
        help="choose study mode: 'btc', 'procurement' or 'genetic'"
    )
    args = parser.parse_args()
    run_pipeline(args.mode)