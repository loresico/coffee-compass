"""
Integration tests for Coffee Compass end-to-end workflows.
"""

import pytest
from pathlib import Path
from coffee_compass.data.preprocess import CoffeePreprocessor
from coffee_compass.models.flavor_predictor import FlavorPredictor


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_full_training_pipeline(self, sample_coffee_data, tmp_path):
        """Test complete training pipeline from data to saved model."""
        # Save sample data
        data_path = tmp_path / "coffee_data.csv"
        sample_coffee_data.to_csv(data_path, index=False)
        
        # Preprocess
        preprocessor = CoffeePreprocessor()
        X, y = preprocessor.preprocess_pipeline(str(data_path), is_training=True)
        
        # Train
        predictor = FlavorPredictor()
        metrics = predictor.train(X, y, test_size=0.3, verbose=False)
        
        # Save
        model_path = tmp_path / "model.joblib"
        predictor.save(str(model_path), preprocessor=preprocessor)
        
        # Load and predict
        loaded_predictor, loaded_preprocessor = FlavorPredictor.load(str(model_path))
        predictions = loaded_predictor.predict(X.head(1))
        
        # Verify
        assert predictions.shape == (1, 6)
        assert model_path.exists()
        assert metrics['test']['rmse'] > 0
    
    def test_prediction_workflow(self, sample_features_targets, temp_model_path):
        """Test prediction workflow with saved model."""
        X, y, preprocessor = sample_features_targets
        
        # Train and save
        predictor = FlavorPredictor()
        predictor.train(X, y, test_size=0.3, verbose=False)
        predictor.save(str(temp_model_path), preprocessor=preprocessor)
        
        # Load model
        loaded_predictor, loaded_preprocessor = FlavorPredictor.load(str(temp_model_path))
        
        # Make predictions
        predictions = loaded_predictor.predict(X.head(5))
        
        # Verify predictions
        assert predictions.shape == (5, 6)
        assert (predictions >= 6).all().all()  # All values >= 6
        assert (predictions <= 10).all().all()  # All values <= 10
    
    def test_new_data_prediction(self, sample_coffee_data, sample_features_targets, temp_model_path):
        """Test prediction on completely new data."""
        X_train, y_train, preprocessor = sample_features_targets
        
        # Train model
        predictor = FlavorPredictor()
        predictor.train(X_train, y_train, test_size=0.3, verbose=False)
        predictor.save(str(temp_model_path), preprocessor=preprocessor)
        
        # Load model
        loaded_predictor, loaded_preprocessor = FlavorPredictor.load(str(temp_model_path))
        
        # Create new data (simulating user input)
        new_data = sample_coffee_data.iloc[:2].copy()
        new_data = loaded_preprocessor.clean_data(new_data)
        new_data = loaded_preprocessor.engineer_features(new_data)
        X_new, _ = loaded_preprocessor.prepare_features(new_data, is_training=False)
        
        # Predict
        predictions = loaded_predictor.predict(X_new)
        
        # Verify
        assert predictions.shape == (2, 6)
        assert all(col in predictions.columns for col in predictor.target_names)


@pytest.mark.integration
class TestDataQuality:
    """Integration tests for data quality checks."""
    
    def test_altitude_validation_integration(self, sample_coffee_data_with_issues, tmp_path):
        """Test that altitude validation works in full pipeline."""
        # Save data with issues
        data_path = tmp_path / "data_with_issues.csv"
        sample_coffee_data_with_issues.to_csv(data_path, index=False)
        
        # Process
        preprocessor = CoffeePreprocessor()
        X, y = preprocessor.preprocess_pipeline(str(data_path), is_training=True)
        
        # Train (should work despite data issues)
        predictor = FlavorPredictor()
        metrics = predictor.train(X, y, test_size=0.3, verbose=False)
        
        # Should complete successfully
        assert predictor.model is not None
        assert len(X) > 0  # Some data should remain after cleaning
    
    def test_missing_values_handling(self, sample_coffee_data_with_issues):
        """Test that missing values are handled throughout pipeline."""
        preprocessor = CoffeePreprocessor()
        df = preprocessor.clean_data(sample_coffee_data_with_issues)
        df = preprocessor.engineer_features(df)
        X, y = preprocessor.prepare_features(df, is_training=True)
        
        # No NaN in features or targets
        assert not X.isna().any().any()
        assert not y.isna().any().any()