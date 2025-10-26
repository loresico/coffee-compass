"""
Tests for data preprocessing and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from coffee_compass.data.preprocess import CoffeePreprocessor, get_preprocessor

class TestCoffeePreprocessor:
    """Test suite for CoffeePreprocessor class."""
    
    def test_initialization(self):
        """Test preprocessor initializes correctly."""
        preprocessor = CoffeePreprocessor()
        assert preprocessor.target_columns == ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
        assert preprocessor.categorical_encoders == {}
        assert preprocessor.feature_columns == []
    
    def test_clean_data_basic(self, sample_coffee_data):
        """Test basic data cleaning."""
        preprocessor = CoffeePreprocessor()
        cleaned = preprocessor.clean_data(sample_coffee_data)
        
        # Should filter to only Arabica
        assert (cleaned['Species'] == 'Arabica').all()
        
        # Should have all target columns
        for col in preprocessor.target_columns:
            assert col in cleaned.columns
        
        # No missing values in targets
        assert not cleaned[preprocessor.target_columns].isna().any().any()
    
    def test_clean_data_altitude_validation(self, sample_coffee_data_with_issues):
        """Test altitude outlier removal."""
        preprocessor = CoffeePreprocessor()
        cleaned = preprocessor.clean_data(sample_coffee_data_with_issues)
        
        # Should remove altitudes >3000m or <0
        assert (cleaned['altitude_mean_meters'] <= 3000).all()
        assert (cleaned['altitude_mean_meters'] >= 0).all()
    
    def test_clean_data_missing_altitude(self, sample_coffee_data_with_issues):
        """Test missing altitude value handling."""
        preprocessor = CoffeePreprocessor()
        cleaned = preprocessor.clean_data(sample_coffee_data_with_issues)
        
        # Should fill missing altitudes
        assert not cleaned['altitude_mean_meters'].isna().any()
    
    def test_clean_data_processing_method(self, sample_coffee_data_with_issues):
        """Test processing method standardization."""
        preprocessor = CoffeePreprocessor()
        cleaned = preprocessor.clean_data(sample_coffee_data_with_issues)
        
        # Should standardize processing methods
        assert 'Washed / Wet' not in cleaned['Processing.Method'].values
        assert 'Natural / Dry' not in cleaned['Processing.Method'].values
        assert 'Washed' in cleaned['Processing.Method'].values
        assert 'Natural' in cleaned['Processing.Method'].values
    
    def test_engineer_features_altitude_category(self, sample_coffee_data):
        """Test altitude category creation."""
        preprocessor = CoffeePreprocessor()
        df = preprocessor.clean_data(sample_coffee_data)
        df = preprocessor.engineer_features(df)
        
        assert 'altitude_category' in df.columns
        # Check categories are valid
        valid_categories = ['Low', 'Medium', 'High', 'Very High']
        assert all(cat in valid_categories for cat in df['altitude_category'].unique())
    
    def test_engineer_features_altitude_score(self, sample_coffee_data):
        """Test altitude score calculation."""
        preprocessor = CoffeePreprocessor()
        df = preprocessor.clean_data(sample_coffee_data)
        df = preprocessor.engineer_features(df)
        
        assert 'altitude_score' in df.columns
        # Altitude score should be normalized (altitude / 2000)
        assert df['altitude_score'].min() >= 0
        assert df['altitude_score'].max() <= 2  # Max reasonable altitude ~4000m / 2000 = 2
    
    def test_engineer_features_processing_complexity(self, sample_coffee_data):
        """Test processing complexity score."""
        preprocessor = CoffeePreprocessor()
        df = preprocessor.clean_data(sample_coffee_data)
        df = preprocessor.engineer_features(df)
        
        assert 'processing_complexity' in df.columns
        # Should be 1, 2, or 3
        assert set(df['processing_complexity'].unique()).issubset({1, 2, 3})
    
    def test_engineer_features_premier_origin(self, sample_coffee_data):
        """Test premier origin flag."""
        preprocessor = CoffeePreprocessor()
        df = preprocessor.clean_data(sample_coffee_data)
        df = preprocessor.engineer_features(df)
        
        assert 'premier_origin' in df.columns
        # Should be 0 or 1
        assert set(df['premier_origin'].unique()).issubset({0, 1})
        
        # Ethiopia, Kenya, Colombia should be premier
        ethiopia_rows = df[df['Country.of.Origin'] == 'Ethiopia']
        assert (ethiopia_rows['premier_origin'] == 1).all()
    
    def test_engineer_features_premium_variety(self, sample_coffee_data):
        """Test premium variety flag."""
        preprocessor = CoffeePreprocessor()
        df = preprocessor.clean_data(sample_coffee_data)
        df = preprocessor.engineer_features(df)
        
        assert 'premium_variety' in df.columns
        # Should be 0 or 1
        assert set(df['premium_variety'].unique()).issubset({0, 1})
        
        # Bourbon should be premium
        bourbon_rows = df[df['Variety'] == 'Bourbon']
        assert (bourbon_rows['premium_variety'] == 1).all()
    
    def test_prepare_features_one_hot_encoding(self, sample_coffee_data):
        """Test one-hot encoding of categorical features."""
        preprocessor = CoffeePreprocessor()
        df = preprocessor.clean_data(sample_coffee_data)
        df = preprocessor.engineer_features(df)
        X, y = preprocessor.prepare_features(df, is_training=True)
        
        # Should have more columns due to one-hot encoding
        assert X.shape[1] > 10  # More than just base features
        
        # Should have country columns
        country_cols = [col for col in X.columns if 'Country.of.Origin' in col]
        assert len(country_cols) > 0
        
        # Should have variety columns
        variety_cols = [col for col in X.columns if 'Variety_' in col]
        assert len(variety_cols) > 0
    
    def test_prepare_features_targets(self, sample_coffee_data):
        """Test target preparation."""
        preprocessor = CoffeePreprocessor()
        df = preprocessor.clean_data(sample_coffee_data)
        df = preprocessor.engineer_features(df)
        X, y = preprocessor.prepare_features(df, is_training=True)
        
        # Y should have 6 columns (sensory scores)
        assert y.shape[1] == 6
        assert list(y.columns) == preprocessor.target_columns
        
        # Y values should be in reasonable range (7-10 for specialty coffee)
        assert y.min().min() >= 7
        assert y.max().max() <= 10
    
    def test_prepare_features_prediction_mode(self, sample_coffee_data):
        """Test feature preparation in prediction mode (is_training=False)."""
        preprocessor = CoffeePreprocessor()
        
        # Train mode
        df = preprocessor.clean_data(sample_coffee_data)
        df = preprocessor.engineer_features(df)
        X_train, y_train = preprocessor.prepare_features(df, is_training=True)
        
        # Prediction mode with new data
        new_data = sample_coffee_data.iloc[:2].copy()
        new_data = preprocessor.clean_data(new_data)
        new_data = preprocessor.engineer_features(new_data)
        X_pred, y_pred = preprocessor.prepare_features(new_data, is_training=False)
        
        # Should have same number of features
        assert X_pred.shape[1] == X_train.shape[1]
        
        # Should have same columns
        assert list(X_pred.columns) == list(X_train.columns)
    
    def test_full_pipeline(self, sample_coffee_data, tmp_path):
        """Test complete preprocessing pipeline."""
        # Save sample data to temp file
        data_path = tmp_path / "test_data.csv"
        sample_coffee_data.to_csv(data_path, index=False)
        
        preprocessor = CoffeePreprocessor()
        X, y = preprocessor.preprocess_pipeline(str(data_path), is_training=True)
        
        # Should return valid features and targets
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        assert X.shape[0] == y.shape[0]  # Same number of samples
        assert X.shape[1] > 0  # Has features
        assert y.shape[1] == 6  # 6 sensory scores
    
    def test_feature_names_stored(self, sample_coffee_data):
        """Test that feature names are stored correctly."""
        preprocessor = CoffeePreprocessor()
        df = preprocessor.clean_data(sample_coffee_data)
        df = preprocessor.engineer_features(df)
        X, y = preprocessor.prepare_features(df, is_training=True)
        
        feature_names = preprocessor.get_feature_names()
        
        assert len(feature_names) == X.shape[1]
        assert feature_names == X.columns.tolist()
    
    def test_get_preprocessor_factory(self):
        """Test factory function."""
        preprocessor = get_preprocessor()
        assert isinstance(preprocessor, CoffeePreprocessor)
    
    def test_load_data_with_index_col(self, tmp_path):
        """Test loading CSV with index column."""
        # Create CSV with index
        data = pd.DataFrame({
            'Species': ['Arabica'] * 5,
            'Country.of.Origin': ['Ethiopia'] * 5,
            'altitude_mean_meters': [1800] * 5,
            'Processing.Method': ['Washed'] * 5,
            'Variety': ['Other'] * 5,
            'Aroma': [8.0] * 5,
            'Flavor': [8.0] * 5,
            'Aftertaste': [8.0] * 5,
            'Acidity': [8.0] * 5,
            'Body': [8.0] * 5,
            'Balance': [8.0] * 5,
        })
        
        # Save with index
        csv_path = tmp_path / "with_index.csv"
        data.to_csv(csv_path, index=True)
        
        preprocessor = CoffeePreprocessor()
        df = preprocessor.load_data(str(csv_path))
        
        assert len(df) == 5
        assert 'Species' in df.columns
    
    def test_clean_data_no_species_column(self):
        """Test cleaning data without Species column."""
        data = pd.DataFrame({
            'Country.of.Origin': ['Ethiopia'] * 5,
            'altitude_mean_meters': [1800] * 5,
            'Processing.Method': ['Washed'] * 5,
            'Variety': ['Other'] * 5,
            'Aroma': [8.0] * 5,
            'Flavor': [8.0] * 5,
            'Aftertaste': [8.0] * 5,
            'Acidity': [8.0] * 5,
            'Body': [8.0] * 5,
            'Balance': [8.0] * 5,
        })
        
        preprocessor = CoffeePreprocessor()
        cleaned = preprocessor.clean_data(data)
        
        # Should work without Species column
        assert len(cleaned) == 5
    
    def test_clean_data_no_altitude_column(self):
        """Test cleaning data without altitude column."""
        data = pd.DataFrame({
            'Species': ['Arabica'] * 5,
            'Country.of.Origin': ['Ethiopia'] * 5,
            'Processing.Method': ['Washed'] * 5,
            'Variety': ['Other'] * 5,
            'Aroma': [8.0] * 5,
            'Flavor': [8.0] * 5,
            'Aftertaste': [8.0] * 5,
            'Acidity': [8.0] * 5,
            'Body': [8.0] * 5,
            'Balance': [8.0] * 5,
        })
        
        preprocessor = CoffeePreprocessor()
        # Should handle missing altitude column gracefully
        cleaned = preprocessor.clean_data(data)
        assert len(cleaned) == 5
    
    def test_clean_data_no_processing_method_column(self):
        """Test cleaning data without Processing.Method column."""
        data = pd.DataFrame({
            'Species': ['Arabica'] * 5,
            'Country.of.Origin': ['Ethiopia'] * 5,
            'altitude_mean_meters': [1800] * 5,
            'Variety': ['Other'] * 5,
            'Aroma': [8.0] * 5,
            'Flavor': [8.0] * 5,
            'Aftertaste': [8.0] * 5,
            'Acidity': [8.0] * 5,
            'Body': [8.0] * 5,
            'Balance': [8.0] * 5,
        })
        
        preprocessor = CoffeePreprocessor()
        cleaned = preprocessor.clean_data(data)
        assert len(cleaned) == 5
    
    def test_clean_data_no_variety_column(self):
        """Test cleaning data without Variety column."""
        data = pd.DataFrame({
            'Species': ['Arabica'] * 5,
            'Country.of.Origin': ['Ethiopia'] * 5,
            'altitude_mean_meters': [1800] * 5,
            'Processing.Method': ['Washed'] * 5,
            'Aroma': [8.0] * 5,
            'Flavor': [8.0] * 5,
            'Aftertaste': [8.0] * 5,
            'Acidity': [8.0] * 5,
            'Body': [8.0] * 5,
            'Balance': [8.0] * 5,
        })
        
        preprocessor = CoffeePreprocessor()
        cleaned = preprocessor.clean_data(data)
        assert len(cleaned) == 5
    
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_clean_data_all_missing_altitude(self):
        """Test cleaning data with all missing altitudes."""
        data = pd.DataFrame({
            'Species': ['Arabica'] * 5,
            'Country.of.Origin': ['Ethiopia'] * 5,
            'altitude_mean_meters': [np.nan] * 5,
            'Processing.Method': ['Washed'] * 5,
            'Variety': ['Other'] * 5,
            'Aroma': [8.0] * 5,
            'Flavor': [8.0] * 5,
            'Aftertaste': [8.0] * 5,
            'Acidity': [8.0] * 5,
            'Body': [8.0] * 5,
            'Balance': [8.0] * 5,
        })
        
        preprocessor = CoffeePreprocessor()
        cleaned = preprocessor.clean_data(data)
        
        # Should fill with global median (which will be NaN, but shouldn't crash)
        assert len(cleaned) == 5
    
    def test_clean_data_drops_rows_with_missing_targets(self):
        """Test that rows with missing targets are dropped."""
        data = pd.DataFrame({
            'Species': ['Arabica'] * 10,
            'Country.of.Origin': ['Ethiopia'] * 10,
            'altitude_mean_meters': [1800] * 10,
            'Processing.Method': ['Washed'] * 10,
            'Variety': ['Other'] * 10,
            'Aroma': [8.0] * 10,
            'Flavor': [8.0, np.nan, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            'Aftertaste': [8.0] * 10,
            'Acidity': [8.0] * 10,
            'Body': [8.0] * 10,
            'Balance': [8.0, 8.0, np.nan, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
        })
        
        preprocessor = CoffeePreprocessor()
        cleaned = preprocessor.clean_data(data)
        
        # Should drop 2 rows (indices 1 and 2)
        assert len(cleaned) == 8
    
    def test_clean_data_empty_after_cleaning_raises_error(self):
        """Test that empty dataset after cleaning raises error."""
        data = pd.DataFrame({
            'Species': ['Arabica'] * 5,
            'Country.of.Origin': ['Ethiopia'] * 5,
            'altitude_mean_meters': [1800] * 5,
            'Processing.Method': ['Washed'] * 5,
            'Variety': ['Other'] * 5,
            'Aroma': [np.nan] * 5,  # All NaN
            'Flavor': [np.nan] * 5,
            'Aftertaste': [np.nan] * 5,
            'Acidity': [np.nan] * 5,
            'Body': [np.nan] * 5,
            'Balance': [np.nan] * 5,
        })
        
        preprocessor = CoffeePreprocessor()
        
        with pytest.raises(ValueError, match="No valid data remaining"):
            preprocessor.clean_data(data)
    
    def test_engineer_features_no_altitude_data(self):
        """Test feature engineering when altitude is all NaN."""
        data = pd.DataFrame({
            'Species': ['Arabica'] * 5,
            'Country.of.Origin': ['Ethiopia'] * 5,
            'altitude_mean_meters': [np.nan] * 5,
            'Processing.Method': ['Washed'] * 5,
            'Variety': ['Other'] * 5,
        })
        
        preprocessor = CoffeePreprocessor()
        engineered = preprocessor.engineer_features(data)
        
        # Should create altitude_category as 'Medium' for all
        assert 'altitude_category' in engineered.columns
        # altitude_score should handle NaN
        assert 'altitude_score' in engineered.columns
    
    def test_engineer_features_unknown_processing_method(self):
        """Test feature engineering with unknown processing method."""
        data = pd.DataFrame({
            'Species': ['Arabica'] * 5,
            'Country.of.Origin': ['Ethiopia'] * 5,
            'altitude_mean_meters': [1800] * 5,
            'Processing.Method': ['Unknown'] * 5,  # Unknown method
            'Variety': ['Other'] * 5,
        })
        
        preprocessor = CoffeePreprocessor()
        engineered = preprocessor.engineer_features(data)
        
        # Should default to 1 for unknown
        assert (engineered['processing_complexity'] == 1).all()
    
    def test_prepare_features_maintains_column_order(self, sample_coffee_data):
        """Test that column order is consistent."""
        preprocessor = CoffeePreprocessor()
        df = preprocessor.clean_data(sample_coffee_data)
        df = preprocessor.engineer_features(df)
        
        # Train mode
        X1, _ = preprocessor.prepare_features(df, is_training=True)
        columns1 = X1.columns.tolist()
        
        # Predict mode with same data
        X2, _ = preprocessor.prepare_features(df, is_training=False)
        columns2 = X2.columns.tolist()
        
        # Columns should be in same order
        assert columns1 == columns2
    
    def test_preprocess_pipeline_file_not_found(self):
        """Test pipeline with non-existent file."""
        preprocessor = CoffeePreprocessor()
        
        # Should raise FileNotFoundError or similar
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            preprocessor.preprocess_pipeline("nonexistent.csv", is_training=True)