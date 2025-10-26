"""
Training script with hyperparameter optimization using Optuna.
Run from project root: python -m coffee_compass.scripts.train_with_optimization
"""

from pathlib import Path
import warnings
import optuna
from optuna.samplers import TPESampler

from coffee_compass.data.preprocess import CoffeePreprocessor
from coffee_compass.models.flavor_predictor import FlavorPredictor


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function to minimize."""
    
    # Define hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'base_score': 0.5
    }
    
    # Train model with these parameters
    predictor = FlavorPredictor(model_params=params)
    
    # Manual train/val split (instead of using predictor.train())
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np
    
    base_model = XGBRegressor(**params)
    model = MultiOutputRegressor(base_model)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    
    # Calculate RMSE (what we want to minimize)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse


def main():
    """Train with hyperparameter optimization."""
    
    # Get paths
    script_dir = Path(__file__).parent
    package_dir = script_dir.parent
    
    data_path = package_dir / "data" / "raw" / "arabica_data.csv"
    model_path = package_dir / "models" / "saved" / "flavor_predictor_optimized.joblib"
    
    print("="*60)
    print("COFFEE COMPASS - TRAINING WITH OPTIMIZATION")
    print("="*60)
    
    # Check if data exists
    if not data_path.exists():
        print(f"\n❌ ERROR: Dataset not found at {data_path}")
        return
    
    # Load and preprocess data
    print("\n📁 Loading and preprocessing data...")
    preprocessor = CoffeePreprocessor()
    X, y = preprocessor.preprocess_pipeline(str(data_path), is_training=True)
    
    print(f"✅ Data loaded: {len(X)} samples, {X.shape[1]} features")
    
    # Split into train/validation for optimization
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    
    # Run Optuna optimization
    print("\n🔍 Starting hyperparameter optimization...")
    print("   This will try 50 different parameter combinations")
    print("   Should take 5-10 minutes\n")
    
    study = optuna.create_study(
        direction='minimize',  # Minimize RMSE
        sampler=TPESampler(seed=42),
        study_name='coffee_compass_optimization'
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=50,  # Try 50 combinations
            show_progress_bar=True
        )
    
    # Get best parameters
    best_params = study.best_params
    best_rmse = study.best_value
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"\nBest RMSE: {best_rmse:.4f}")
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param:20} = {value}")
    
    # Train final model with best parameters on ALL data
    print("\n🚀 Training final model with optimized hyperparameters...")
    
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['base_score'] = 0.5
    
    predictor = FlavorPredictor(model_params=best_params)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = predictor.train(X, y, test_size=0.2, verbose=True)
    
    # Save model
    print(f"\n💾 Saving optimized model...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(str(model_path), preprocessor=preprocessor)
    
    # Display results
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Test R²: {metrics['test']['r2']:.4f}")
    print(f"Test RMSE: {metrics['test']['rmse']:.4f}")
    
    # Show feature importance
    print("\n📊 Top 10 Most Important Features:")
    importance_df = predictor.get_feature_importance(top_n=10)
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:40} {row['importance']:.4f}")
    
    print(f"\n💡 Model saved to: {model_path}")
    print("   Use this optimized model in production!")
    print("="*60)
    
    # Save best parameters to file
    import json
    params_file = package_dir / "models" / "saved" / "best_hyperparameters.json"
    with open(params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n📝 Best parameters saved to: {params_file}")


if __name__ == "__main__":
    main()