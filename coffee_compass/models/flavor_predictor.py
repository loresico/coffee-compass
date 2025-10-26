"""
Multi-output XGBoost model for predicting coffee flavor profiles.
Predicts 6 sensory attributes: Aroma, Flavor, Aftertaste, Acidity, Body, Balance.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from typing import Dict, Tuple, Optional
import shap


class FlavorPredictor:
    """Multi-output flavor profile prediction model."""

    def __init__(self, model_params: Optional[Dict] = None):
        """
        Initialize the flavor predictor.

        Args:
            model_params: XGBoost hyperparameters. If None, uses sensible defaults.
        """
        if model_params is None:
            model_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "random_state": 42,
                "n_jobs": -1,
                "base_score": 0.5,
            }

        self.model_params = model_params
        self.model = None
        self.feature_names = None
        self.target_names = [
            "Aroma",
            "Flavor",
            "Aftertaste",
            "Acidity",
            "Body",
            "Balance",
        ]
        self.explainer = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = 0.2,
        verbose: bool = True,
    ) -> Dict:
        """
        Train the multi-output model.

        Args:
            X: Feature matrix
            y: Target matrix (6 sensory scores)
            test_size: Proportion of data for testing
            verbose: Whether to print training progress

        Returns:
            Dictionary with training metrics
        """
        self.feature_names = X.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        if verbose:
            print(
                f"Training on {len(X_train)} samples, testing on {len(X_test)} samples"
            )
            print(f"Features: {X.shape[1]}, Targets: {y.shape[1]}")

        # Create multi-output model
        base_model = XGBRegressor(**self.model_params)
        self.model = MultiOutputRegressor(base_model)

        # Train
        if verbose:
            print("\nTraining multi-output XGBoost model...")

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        metrics = self._calculate_metrics(
            y_train, y_test, y_pred_train, y_pred_test, verbose
        )

        if verbose:
            print("\nSHAP explainer: skipped")
        self.explainer = None

        return metrics

    def _calculate_metrics(
        self,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        y_pred_train: np.ndarray,
        y_pred_test: np.ndarray,
        verbose: bool = True,
    ) -> Dict:
        """Calculate and optionally print training metrics."""
        metrics = {"train": {}, "test": {}}

        # Overall metrics
        metrics["train"]["rmse"] = np.sqrt(mean_squared_error(y_train, y_pred_train))
        metrics["train"]["mae"] = mean_absolute_error(y_train, y_pred_train)
        metrics["train"]["r2"] = r2_score(y_train, y_pred_train)

        metrics["test"]["rmse"] = np.sqrt(mean_squared_error(y_test, y_pred_test))
        metrics["test"]["mae"] = mean_absolute_error(y_test, y_pred_test)
        metrics["test"]["r2"] = r2_score(y_test, y_pred_test)

        # Per-target metrics
        for i, target in enumerate(self.target_names):
            metrics["test"][f"{target}_rmse"] = np.sqrt(
                mean_squared_error(y_test.iloc[:, i], y_pred_test[:, i])
            )
            metrics["test"][f"{target}_r2"] = r2_score(
                y_test.iloc[:, i], y_pred_test[:, i]
            )

        if verbose:
            print("\n" + "=" * 50)
            print("MODEL PERFORMANCE")
            print("=" * 50)
            print(f"\nOverall Metrics:")
            print(f"  Train RMSE: {metrics['train']['rmse']:.4f}")
            print(f"  Test RMSE:  {metrics['test']['rmse']:.4f}")
            print(f"  Train MAE:  {metrics['train']['mae']:.4f}")
            print(f"  Test MAE:   {metrics['test']['mae']:.4f}")
            print(f"  Train R²:   {metrics['train']['r2']:.4f}")
            print(f"  Test R²:    {metrics['test']['r2']:.4f}")

            print(f"\nPer-Target Performance (Test Set):")
            for target in self.target_names:
                rmse = metrics["test"][f"{target}_rmse"]
                r2 = metrics["test"][f"{target}_r2"]
                print(f"  {target:12} - RMSE: {rmse:.4f}, R²: {r2:.4f}")

        return metrics

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict flavor profiles for new coffee samples.

        Args:
            X: Feature matrix

        Returns:
            DataFrame with predicted sensory scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X)
        return pd.DataFrame(predictions, columns=self.target_names)

    def predict_single(
        self, country: str, altitude: float, processing: str, variety: str
    ) -> Dict:
        """
        Predict flavor profile for a single coffee with user-friendly input.

        Args:
            country: Country of origin
            altitude: Altitude in meters
            processing: Processing method (Washed/Natural/Honey)
            variety: Coffee variety

        Returns:
            Dictionary with predictions and explanations
        """
        # This would need the preprocessor to create proper feature vector
        # For now, this is a placeholder showing the interface
        raise NotImplementedError(
            "Use predict() with properly preprocessed features, "
            "or call through the Gradio interface."
        )

    def explain_prediction(self, X: pd.DataFrame, index: int = 0) -> Dict:
        """
        Generate SHAP explanation for a prediction.

        Args:
            X: Feature matrix
            index: Index of sample to explain

        Returns:
            Dictionary with SHAP values and base value
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Train model first.")

        shap_values = self.explainer.shap_values(X.iloc[[index]])

        return {
            "shap_values": shap_values[0],
            "base_value": self.explainer.expected_value,
            "feature_values": X.iloc[index].values,
            "feature_names": self.feature_names,
        }

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance averaged across all target variables.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Average importance across all estimators
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)

        avg_importance = np.mean(importances, axis=0)

        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": avg_importance}
        ).sort_values("importance", ascending=False)

        return importance_df.head(top_n)

    def save(self, filepath: str, preprocessor=None):
        """Save model and metadata to disk."""
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "model_params": self.model_params,
            "preprocessor": preprocessor,
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "FlavorPredictor":
        """Load model from disk."""
        model_data = joblib.load(filepath)

        predictor = cls(model_params=model_data["model_params"])
        predictor.model = model_data["model"]
        predictor.feature_names = model_data["feature_names"]
        predictor.target_names = model_data["target_names"]

        print(f"Model loaded from {filepath}")
        return predictor, model_data.get("preprocessor")


def train_flavor_model(
    X: pd.DataFrame, y: pd.DataFrame, save_path: Optional[str] = None
) -> FlavorPredictor:
    """
    Convenience function to train and optionally save a flavor prediction model.

    Args:
        X: Feature matrix
        y: Target matrix
        save_path: Path to save model (optional)

    Returns:
        Trained FlavorPredictor instance
    """
    predictor = FlavorPredictor()
    predictor.train(X, y, verbose=True)

    if save_path:
        predictor.save(save_path)

    # Print feature importance
    print("\n" + "=" * 50)
    print("TOP FEATURES FOR FLAVOR PREDICTION")
    print("=" * 50)
    importance_df = predictor.get_feature_importance(top_n=15)
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']:40} {row['importance']:.4f}")

    return predictor


if __name__ == "__main__":
    # Example usage
    from preprocess import CoffeePreprocessor

    print("Loading and preprocessing data...")
    preprocessor = CoffeePreprocessor()
    X, y = preprocessor.preprocess_pipeline("data/arabica_data.csv", is_training=True)

    print("\nTraining flavor prediction model...")
    model = train_flavor_model(X, y, save_path="models/flavor_predictor.joblib")

    print("\nMaking sample prediction...")
    sample_prediction = model.predict(X.head(1))
    print("\nSample flavor profile:")
    print(sample_prediction)
