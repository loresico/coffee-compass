"""
Training script for Coffee Compass flavor prediction model.
Run from project root: python -m coffee_compass.scripts.train
"""

from pathlib import Path
import warnings

from coffee_compass.data.preprocess import CoffeePreprocessor
from coffee_compass.models.flavor_predictor import FlavorPredictor


def main():
    """Train and save the flavor prediction model."""
    
    # Get paths
    script_dir = Path(__file__).parent  # coffee_compass/scripts/
    package_dir = script_dir.parent      # coffee_compass/
    
    data_path = package_dir / "data" / "raw" / "arabica_data.csv"
    model_path = package_dir / "models" / "saved" / "flavor_predictor.joblib"
    
    print("="*60)
    print("COFFEE COMPASS - MODEL TRAINING")
    print("="*60)
    print(f"\nData: {data_path}")
    print(f"Model will be saved to: {model_path}")
    
    # Check if data exists
    if not data_path.exists():
        print(f"\n❌ ERROR: Dataset not found at {data_path}")
        print("\nDownload it with:")
        print(f"  curl -o {data_path} \\")
        print("  https://raw.githubusercontent.com/jldbc/coffee-quality-database/master/data/arabica_data_cleaned.csv")
        return
    
    # Load and preprocess data
    print("\n📁 Loading and preprocessing data...")
    preprocessor = CoffeePreprocessor()
    X, y = preprocessor.preprocess_pipeline(str(data_path), is_training=True)
    
    print(f"✅ Data loaded: {len(X)} samples, {X.shape[1]} features")
    
    # Train model
    print("\n🚀 Training multi-output XGBoost model...")
    predictor = FlavorPredictor()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = predictor.train(X, y, test_size=0.2, verbose=True)
    
    # Save model
    print(f"\n💾 Saving model...")
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
    
    print("\n🚀 Ready to launch app: python -m coffee_compass.app")
    print("="*60)


if __name__ == "__main__":
    main()