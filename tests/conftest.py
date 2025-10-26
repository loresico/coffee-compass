"""
Pytest configuration and shared fixtures for Coffee Compass tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_coffee_data():
    """Create sample coffee data for testing."""
    # Create larger dataset (50 samples) to avoid R² warnings
    np.random.seed(42)
    n_samples = 50
    
    countries = ['Ethiopia', 'Brazil', 'Colombia', 'Kenya', 'Guatemala', 'Costa Rica', 'Honduras', 'Mexico']
    processing = ['Washed', 'Natural', 'Honey']
    varieties = ['Other', 'Bourbon', 'Caturra', 'SL28', 'Typica', 'Catuai']
    
    data = {
        'Species': ['Arabica'] * n_samples,
        'Country.of.Origin': np.random.choice(countries, n_samples),
        'altitude_mean_meters': np.random.randint(1000, 2500, n_samples).astype(float),
        'Processing.Method': np.random.choice(processing, n_samples),
        'Variety': np.random.choice(varieties, n_samples),
        'Aroma': np.random.uniform(7.5, 9.0, n_samples),
        'Flavor': np.random.uniform(7.5, 9.0, n_samples),
        'Aftertaste': np.random.uniform(7.5, 9.0, n_samples),
        'Acidity': np.random.uniform(7.5, 9.0, n_samples),
        'Body': np.random.uniform(7.5, 9.0, n_samples),
        'Balance': np.random.uniform(7.5, 9.0, n_samples),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_coffee_data_with_issues():
    """Create sample data with quality issues for testing data cleaning."""
    data = {
        'Species': ['Arabica'] * 8,
        'Country.of.Origin': ['Ethiopia', 'Brazil', None, 'Kenya', 'Guatemala', 'Colombia', 'Peru', 'Honduras'],
        'altitude_mean_meters': [1800, 10000, 1600, -100, np.nan, 1700, 1500, 2500],  # Issues: >3000, <0, NaN
        'Processing.Method': ['Washed', 'Natural', 'Washed / Wet', None, 'Honey', 'Washed', 'Natural / Dry', 'Washed'],
        'Variety': ['Other', 'Bourbon', 'Caturra', None, 'Bourbon', 'Typica', 'Caturra', 'Other'],
        'Aroma': [8.2, 7.8, 8.1, 8.5, 8.0, 8.3, 7.9, 8.2],
        'Flavor': [8.3, 7.9, 8.2, 8.6, 8.1, 8.4, 8.0, 8.3],
        'Aftertaste': [8.1, 7.7, 8.0, 8.4, 7.9, 8.2, 7.8, 8.1],
        'Acidity': [8.4, 7.6, 8.3, 8.7, 8.2, 8.5, 7.7, 8.4],
        'Body': [8.0, 8.2, 8.1, 8.3, 8.0, 8.1, 8.3, 8.2],
        'Balance': [8.2, 7.9, 8.1, 8.5, 8.0, 8.3, 8.0, 8.2],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_features_targets(sample_coffee_data):
    """Return preprocessed features and targets."""
    from coffee_compass.data.preprocess import CoffeePreprocessor
    
    preprocessor = CoffeePreprocessor()
    df = preprocessor.clean_data(sample_coffee_data)
    df = preprocessor.engineer_features(df)
    X, y = preprocessor.prepare_features(df, is_training=True)
    
    return X, y, preprocessor


@pytest.fixture
def temp_model_path(tmp_path):
    """Create temporary path for model saving."""
    return tmp_path / "test_model.joblib"
