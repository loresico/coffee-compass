"""
Tests for FlavorPredictor model.
"""

import pytest
import pandas as pd
import numpy as np
from coffee_compass.models.flavor_predictor import FlavorPredictor, train_flavor_model

class TestFlavorPredictor:
    """Test suite for FlavorPredictor class."""
    
    def test_initialization_default_params(self):
        """Test predictor initializes with default parameters."""
        predictor = FlavorPredictor()
        
        assert predictor.model is None
        assert predictor.feature_names is None
        assert predictor.target_names == ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
        assert predictor.model_params is not None
        assert 'n_estimators' in predictor.model_params
    
    def test_initialization_custom_params(self):
        """Test predictor initializes with custom parameters."""
        custom_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1
        }
        predictor = FlavorPredictor(model_params=custom_params)
        
        assert predictor.model_params['n_estimators'] == 100
        assert predictor.model_params['max_depth'] == 5
        assert predictor.model_params['learning_rate'] == 0.1
    
    def test_train_basic(self, sample_features_targets):
        """Test basic model training."""
        X, y, _ = sample_features_targets
        
        predictor = FlavorPredictor()
        metrics = predictor.train(X, y, test_size=0.3, verbose=False)
        
        # Model should be trained
        assert predictor.model is not None
        
        # Should have feature names
        assert predictor.feature_names is not None
        assert len(predictor.feature_names) == X.shape[1]
        
        # Should have metrics
        assert 'train' in metrics
        assert 'test' in metrics
        assert 'rmse' in metrics['test']
        assert 'r2' in metrics['test']
    
    def test_train_metrics_reasonable(self, sample_features_targets):
        """Test that training metrics are in reasonable ranges."""
        X, y, _ = sample_features_targets
        
        predictor = FlavorPredictor()
        metrics = predictor.train(X, y, test_size=0.3, verbose=False)
        
        # RMSE should be positive and reasonable (< 2 on 7-10 scale)
        assert metrics['test']['rmse'] > 0
        assert metrics['test']['rmse'] < 2
        
        # R² should be between -inf and 1 (typically positive for decent models)
        assert metrics['test']['r2'] < 1.01  # Allow for small numerical errors
    
    def test_predict(self, sample_features_targets):
        """Test model prediction."""
        X, y, _ = sample_features_targets
        
        predictor = FlavorPredictor()
        predictor.train(X, y, test_size=0.3, verbose=False)
        
        # Predict on first 3 samples
        predictions = predictor.predict(X.head(3))
        
        # Should return DataFrame
        assert isinstance(predictions, pd.DataFrame)
        
        # Should have correct shape
        assert predictions.shape[0] == 3
        assert predictions.shape[1] == 6
        
        # Should have correct columns
        assert list(predictions.columns) == predictor.target_names
        
        # Predictions should be in reasonable range (6-10)
        assert predictions.min().min() >= 6
        assert predictions.max().max() <= 10
    
    def test_predict_before_training_raises_error(self, sample_features_targets):
        """Test that predicting before training raises an error."""
        X, y, _ = sample_features_targets
        
        predictor = FlavorPredictor()
        
        with pytest.raises(ValueError, match="Model not trained"):
            predictor.predict(X.head(1))
    
    def test_get_feature_importance(self, sample_features_targets):
        """Test feature importance extraction."""
        X, y, _ = sample_features_targets
        
        predictor = FlavorPredictor()
        predictor.train(X, y, test_size=0.3, verbose=False)
        
        importance_df = predictor.get_feature_importance(top_n=5)
        
        # Should return DataFrame
        assert isinstance(importance_df, pd.DataFrame)
        
        # Should have correct columns
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        
        # Should return top 5
        assert len(importance_df) == 5
        
        # Should be sorted by importance (descending)
        importances = importance_df['importance'].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))
        
        # Importances should be positive
        assert (importance_df['importance'] >= 0).all()
    
    def test_save_and_load(self, sample_features_targets, temp_model_path):
        """Test model saving and loading."""
        X, y, preprocessor = sample_features_targets
        
        # Train and save
        predictor = FlavorPredictor()
        predictor.train(X, y, test_size=0.3, verbose=False)
        predictor.save(str(temp_model_path), preprocessor=preprocessor)
        
        # Check file exists
        assert temp_model_path.exists()
        
        # Load model
        loaded_predictor, loaded_preprocessor = FlavorPredictor.load(str(temp_model_path))
        
        # Should have same attributes
        assert loaded_predictor.feature_names == predictor.feature_names
        assert loaded_predictor.target_names == predictor.target_names
        
        # Should make same predictions
        original_pred = predictor.predict(X.head(3))
        loaded_pred = loaded_predictor.predict(X.head(3))
        
        pd.testing.assert_frame_equal(original_pred, loaded_pred)
        
        # Preprocessor should be saved
        assert loaded_preprocessor is not None
    
    def test_model_deterministic(self, sample_features_targets):
        """Test that model produces deterministic results with fixed seed."""
        X, y, _ = sample_features_targets
        
        # Train two models with same seed
        predictor1 = FlavorPredictor()
        predictor1.train(X, y, test_size=0.3, verbose=False)
        pred1 = predictor1.predict(X.head(3))
        
        predictor2 = FlavorPredictor()
        predictor2.train(X, y, test_size=0.3, verbose=False)
        pred2 = predictor2.predict(X.head(3))
        
        # Predictions should be very similar (allowing for small numerical differences)
        np.testing.assert_array_almost_equal(pred1.values, pred2.values, decimal=4)
    
    def test_different_test_sizes(self, sample_features_targets):
        """Test training with different test sizes."""
        X, y, _ = sample_features_targets
        
        for test_size in [0.1, 0.2, 0.3]:
            predictor = FlavorPredictor()
            metrics = predictor.train(X, y, test_size=test_size, verbose=False)
            
            # Should complete without errors
            assert predictor.model is not None
            assert 'rmse' in metrics['test']
    
    def test_per_target_metrics(self, sample_features_targets):
        """Test that per-target metrics are calculated."""
        X, y, _ = sample_features_targets
        
        predictor = FlavorPredictor()
        metrics = predictor.train(X, y, test_size=0.3, verbose=False)
        
        # Should have per-target metrics
        for target in predictor.target_names:
            assert f'{target}_rmse' in metrics['test']
            assert f'{target}_r2' in metrics['test']
            
            # Metrics should be reasonable
            assert metrics['test'][f'{target}_rmse'] > 0
            assert metrics['test'][f'{target}_rmse'] < 3


class TestFlavorPredictorEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_train_with_insufficient_data(self):
        """Test training with very little data."""
        # Create minimal dataset
        X = pd.DataFrame(np.random.rand(5, 10))
        y = pd.DataFrame(np.random.rand(5, 6), columns=['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance'])
        
        predictor = FlavorPredictor()
        
        # Should handle small dataset
        # (might not perform well but shouldn't crash)
        try:
            metrics = predictor.train(X, y, test_size=0.2, verbose=False)
            assert predictor.model is not None
        except ValueError:
            # If it raises ValueError due to too little data, that's acceptable
            pass
    
    def test_predict_with_wrong_features(self, sample_features_targets):
        """Test prediction with mismatched features."""
        X, y, _ = sample_features_targets
        
        predictor = FlavorPredictor()
        predictor.train(X, y, test_size=0.3, verbose=False)
        
        # Try to predict with wrong number of features
        X_wrong = X.iloc[:, :5]  # Only 5 features instead of all
        
        with pytest.raises(Exception):  # XGBoost will raise some error
            predictor.predict(X_wrong)
    
    def test_explain_prediction_not_implemented(self, sample_features_targets):
        """Test that explain_prediction raises error when explainer is disabled."""
        X, y, _ = sample_features_targets
        
        predictor = FlavorPredictor()
        predictor.train(X, y, test_size=0.2, verbose=False)
        
        # Should raise error because we disabled SHAP
        with pytest.raises(ValueError, match="Explainer not initialized"):
            predictor.explain_prediction(X)
    
    def test_predict_single_not_implemented(self, sample_features_targets):
        """Test that predict_single raises NotImplementedError."""
        X, y, _ = sample_features_targets
        
        predictor = FlavorPredictor()
        predictor.train(X, y, test_size=0.2, verbose=False)
        
        with pytest.raises(NotImplementedError):
            predictor.predict_single("Ethiopia", 1800, "Washed", "Bourbon")
    
    def test_train_verbose_true(self, sample_features_targets, capsys):
        """Test training with verbose=True prints output."""
        X, y, _ = sample_features_targets
        
        predictor = FlavorPredictor()
        predictor.train(X, y, test_size=0.2, verbose=True)
        
        captured = capsys.readouterr()
        assert "Training on" in captured.out or len(captured.out) > 0
    
    def test_calculate_metrics_verbose_false(self, sample_features_targets):
        """Test metrics calculation with verbose=False."""
        X, y, _ = sample_features_targets
        
        predictor = FlavorPredictor()
        # verbose=False should not print
        metrics = predictor.train(X, y, test_size=0.2, verbose=False)
        
        # Should still return metrics
        assert 'train' in metrics
        assert 'test' in metrics
    
    def test_get_feature_importance_untrained(self):
        """Test feature importance before training raises error."""
        predictor = FlavorPredictor()
        
        with pytest.raises(ValueError, match="Model not trained"):
            predictor.get_feature_importance()
    
    def test_feature_importance_different_top_n(self, sample_features_targets):
        """Test feature importance with different top_n values."""
        X, y, _ = sample_features_targets
        
        predictor = FlavorPredictor()
        predictor.train(X, y, test_size=0.2, verbose=False)
        
        # Test different top_n values
        for top_n in [3, 10, 20]:
            importance_df = predictor.get_feature_importance(top_n=min(top_n, X.shape[1]))
            assert len(importance_df) <= top_n
            assert len(importance_df) <= X.shape[1]
    
    def test_save_without_preprocessor(self, sample_features_targets, temp_model_path):
        """Test saving model without preprocessor."""
        X, y, _ = sample_features_targets
        
        predictor = FlavorPredictor()
        predictor.train(X, y, test_size=0.2, verbose=False)
        
        # Save without preprocessor
        predictor.save(str(temp_model_path))
        
        # Load and verify
        loaded_predictor, loaded_preprocessor = FlavorPredictor.load(str(temp_model_path))
        
        assert loaded_predictor.model is not None
        assert loaded_preprocessor is None  # No preprocessor saved
    
    def test_load_model_with_preprocessor(self, sample_features_targets, temp_model_path):
        """Test loading model returns preprocessor when saved."""
        X, y, preprocessor = sample_features_targets
        
        predictor = FlavorPredictor()
        predictor.train(X, y, test_size=0.2, verbose=False)
        predictor.save(str(temp_model_path), preprocessor=preprocessor)
        
        loaded_predictor, loaded_preprocessor = FlavorPredictor.load(str(temp_model_path))
        
        assert loaded_preprocessor is not None

class TestTrainFlavorModelFunction:
    """Test the convenience train_flavor_model function."""
    
    def test_train_flavor_model_no_save(self, sample_features_targets, capsys):
        """Test train_flavor_model without saving."""
        X, y, _ = sample_features_targets
        
        predictor = train_flavor_model(X, y, save_path=None)
        
        # Should return trained predictor
        assert predictor.model is not None
        
        # Should print feature importance
        captured = capsys.readouterr()
        assert "TOP FEATURES" in captured.out
    
    def test_train_flavor_model_with_save(self, sample_features_targets, temp_model_path):
        """Test train_flavor_model with saving."""
        X, y, _ = sample_features_targets
        
        predictor = train_flavor_model(X, y, save_path=str(temp_model_path))
        
        # Should save model
        assert temp_model_path.exists()
        
        # Should be loadable
        loaded, _ = FlavorPredictor.load(str(temp_model_path))
        assert loaded.model is not None