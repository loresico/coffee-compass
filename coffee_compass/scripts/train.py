"""
Training script for Coffee Compass flavor prediction model.
Run this to train the model from scratch and save it for the Gradio app.
"""

import sys
import warnings
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR  # This script should be in project root

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.preprocess import CoffeePreprocessor
from models.flavor_predictor import FlavorPredictor
import pandas as pd


def main():
    """Train and save the flavor prediction model."""
    
    print("="*60)
    print("COFFEE COMPASS - MODEL TRAINING")
    print("="*60)
    print(f"Running from: {Path.cwd()}")
    print(f"Project root: {PROJECT_ROOT}")
    
    # Paths relative to project root
    DATA_PATH = PROJECT_ROOT / "src" / "data" / "arabica_data.csv"
    MODEL_PATH = PROJECT_ROOT / "src" / "models" / "flavor_predictor.joblib"
    
    # Check if data exists
    if not DATA_PATH.exists():
        print(f"\n❌ Error: Dataset not found at {DATA_PATH}")
        print("Please download arabica_data.csv from:")
        print("https://github.com/jldbc/coffee-quality-database")
        print("\nRun:")
        print(f"  mkdir -p {DATA_PATH.parent}")
        print(f"  curl -o {DATA_PATH} https://raw.githubusercontent.com/jldbc/coffee-quality-database/master/data/arabica_data_cleaned.csv")
        return
    
    print(f"\n📁 Loading data from {DATA_PATH}")
    
    # Initialize preprocessor
    preprocessor = CoffeePreprocessor()
    
    # Load and preprocess data
    print("   Cleaning data...")
    X, y = preprocessor.preprocess_pipeline(str(DATA_PATH), is_training=True)
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Targets: {y.shape[1]}")
    
    # Check for any remaining NaNs
    if X.isna().any().any():
        print(f"\n⚠️  Warning: {X.isna().sum().sum()} NaN values in features")
    if y.isna().any().any():
        print(f"\n⚠️  Warning: {y.isna().sum().sum()} NaN values in targets")
    
    # Display some statistics (with warnings suppressed)
    print(f"\n📊 Target Statistics:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(y.describe())
    
    # Train model
    print(f"\n🚀 Training multi-output XGBoost model...")
    print("   This may take a few minutes...")
    
    predictor = FlavorPredictor()
    metrics = predictor.train(X, y, test_size=0.2, verbose=True)
    
    # Save model
    print(f"\n💾 Saving model to {MODEL_PATH}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictor.save(str(MODEL_PATH))
    
    # Display feature importance
    print("\n" + "="*60)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("="*60)
    importance_df = predictor.get_feature_importance(top_n=15)
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:45} {row['importance']:.4f}")
    
    # Make a sample prediction
    print("\n" + "="*60)
    print("SAMPLE PREDICTION")
    print("="*60)
    sample = X.head(1)
    prediction = predictor.predict(sample)
    print("\nInput features (first sample):")
    print(f"  Features: {X.columns.tolist()[:5]}...")  # Show first 5 features
    print("\nPredicted flavor profile:")
    print(prediction.T)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Test R²: {metrics['test']['r2']:.4f}")
    print(f"Test RMSE: {metrics['test']['rmse']:.4f}")
    print("\n🚀 Ready to launch Gradio app with: uv run python src/app.py")
    print("   (Make sure to run from project root!)")
    print("="*60)


if __name__ == "__main__":
    main()