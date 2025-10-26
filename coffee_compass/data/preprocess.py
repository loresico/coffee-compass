"""
Data preprocessing and feature engineering for coffee quality prediction.
Applies domain knowledge about specialty coffee to create meaningful features.
"""

from tabnanny import verbose
import pandas as pd
import numpy as np
from typing import Tuple, List

import matplotlib.pyplot as plt


class CoffeePreprocessor:
    """Handles data cleaning and feature engineering for coffee dataset."""
    
    def __init__(self):
        self.categorical_encoders = {}
        self.feature_columns = []
        self.target_columns = [
            'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance'
        ]
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load coffee dataset from CSV."""
        try:
            # Try reading with first column as index (it's usually an unnamed index column)
            df = pd.read_csv(filepath, index_col=0, encoding='utf-8')
        except Exception as e:
            print(f"Error with index_col=0, trying default: {e}")
            try:
                # Try without index_col but specify encoding
                df = pd.read_csv(filepath, encoding='utf-8')
            
            except Exception as e2:
                print(f"Error with utf-8, trying latin1: {e2}")
                # Try different encoding
                df = pd.read_csv(filepath, encoding='latin1', index_col=0)
        
        # If first column is unnamed or empty, it's likely an index
        if df.columns[0] == '' or 'Unnamed' in str(df.columns[0]):
            df = df.iloc[:, 1:]  # Skip first column
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare raw coffee data."""
        df = df.copy()
        
        # Filter only Arabica (if dataset contains both species)
        if 'Species' in df.columns:
            df = df[df['Species'] == 'Arabica'].copy()
        
        # Handle altitude missing values
        # Use mean altitude if available, otherwise regional average
        if 'altitude_mean_meters' in df.columns and df['altitude_mean_meters'].isna().any():
            
            # Remove unrealistic altitudes (coffee typically grows 500-3000m)
            invalid_altitude = (df['altitude_mean_meters'] > 3000) | (df['altitude_mean_meters'] < 0)
            if invalid_altitude.any():
                n_removed = invalid_altitude.sum()
                print(f"   Removing {n_removed} rows with invalid altitude (>3000m or <0)")
                df = df[~invalid_altitude].copy()
        
            # Fill with country-specific median (only for countries with data)
            country_medians = df.groupby('Country.of.Origin')['altitude_mean_meters'].median()
            df['altitude_mean_meters'] = df.apply(
                lambda row: country_medians.get(row['Country.of.Origin'], np.nan) 
                if pd.isna(row['altitude_mean_meters']) else row['altitude_mean_meters'],
                axis=1
            )
            # If still missing, use global median
            global_median = df['altitude_mean_meters'].median()
            if pd.notna(global_median):
                df = df.fillna({'altitude_mean_meters': global_median})
        
        # Clean processing method - standardize names
        if 'Processing.Method' in df.columns:
            df['Processing.Method'] = df['Processing.Method'].str.strip()
            # Map variations to standard names
            process_mapping = {
                'Washed / Wet': 'Washed',
                'Natural / Dry': 'Natural',
                'Semi-washed / Semi-pulped': 'Honey',
                'Pulped natural / honey': 'Honey'
            }
            df['Processing.Method'] = df['Processing.Method'].replace(process_mapping)
            # Fill missing with most common
            df = df.fillna({'Processing.Method': 'Washed'})
        
        # Clean variety names
        if 'Variety' in df.columns:
            df = df.fillna({'Variety': 'Other'})
            df['Variety'] = df['Variety'].str.strip()
        
        # Remove rows with missing target values
        missing_targets = df[self.target_columns].isna().any(axis=1)
        if missing_targets.any():
            n_dropped = missing_targets.sum()
            print(f"   Dropping {n_dropped} rows with missing target values")
            df = df[~missing_targets].copy()
        
        # Validate we have data left
        if len(df) == 0:
            raise ValueError("No valid data remaining after cleaning!")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features based on coffee expertise."""
        df = df.copy()
        
        # 1. Altitude categories (specialty coffee grading)
        # High altitude -> higher density -> better quality potential
        df['altitude_category'] = pd.cut(
            df['altitude_mean_meters'],
            bins=[0, 1200, 1500, 1800, 3000],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # 2. Altitude score (normalized)
        df['altitude_score'] = df['altitude_mean_meters'] / 2000  # Normalize around typical specialty coffee altitude
        
        # 3. Processing complexity score
        # Natural processing is more complex/risky but can yield unique flavors
        process_complexity = {
            'Natural': 3,
            'Honey': 2,
            'Washed': 1
        }
        df['processing_complexity'] = df['Processing.Method'].map(process_complexity).fillna(1)
        
        # 4. Country tier (based on specialty coffee reputation)
        # This is subjective but based on competition results and market
        premier_origins = ['Ethiopia', 'Kenya', 'Panama', 'Colombia', 'Costa Rica']
        df['premier_origin'] = df['Country.of.Origin'].isin(premier_origins).astype(int)
        
        # 5. Variety quality indicator
        # Some varieties are known for exceptional quality
        premium_varieties = ['Geisha', 'Bourbon', 'Typica', 'SL28', 'SL34']
        df['premium_variety'] = df['Variety'].apply(
            lambda x: 1 if any(var in str(x) for var in premium_varieties) else 0
        )
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features for modeling.
        
        Args:
            df: Cleaned and engineered dataframe
            is_training: Whether this is training data (fit encoders) or prediction (transform only)
        
        Returns:
            X: Feature matrix
            y: Target matrix (if training)
        """
        df = df.copy()
        
        # Select base features
        base_features = [
            'Country.of.Origin',
            'altitude_mean_meters',
            'Processing.Method',
            'Variety'
        ]
        
        # Add engineered features
        engineered_features = [
            'altitude_category',
            'altitude_score',
            'processing_complexity',
            'premier_origin',
            'premium_variety'
        ]
        
        # Combine all features
        feature_df = df[base_features + engineered_features].copy()
        
        # One-hot encode categorical variables
        categorical_cols = ['Country.of.Origin', 'Processing.Method', 'Variety', 'altitude_category']
        
        if is_training:
            # Store the categorical values for later
            for col in categorical_cols:
                self.categorical_encoders[col] = feature_df[col].unique().tolist()
        
        # One-hot encode
        feature_df = pd.get_dummies(
            feature_df,
            columns=categorical_cols,
            drop_first=False  # Keep all categories for interpretability
        )
        
        if is_training:
            self.feature_columns = feature_df.columns.tolist()
        else:
            # Ensure prediction data has same columns as training
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            feature_df = feature_df[self.feature_columns]
        
        X = feature_df
        y = df[self.target_columns] if is_training else None
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Return the list of feature names used in the model."""
        return self.feature_columns
    
    def preprocess_pipeline(self, filepath: str, is_training: bool = True, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            filepath: Path to CSV file
            is_training: Whether this is training data
            
        Returns:
            X: Feature matrix
            y: Target matrix (None if not training)
        """
        df = self.load_data(filepath)
        df = self.clean_data(df)
        df = self.engineer_features(df)
        X, y = self.prepare_features(df, is_training=is_training)
        
        if verbose:
            print(f'Feature matrix = ', X.columns)
            print('Target = ', y)
            
            fig, axes = plt.subplots(1, 1, figsize=(18, 5))
            axes.plot(X['Country.of.Origin_Honduras'], 'b*', linewidth=2, label='Analytical')
            plt.tight_layout()
            plt.show()
        
        return X, y


def get_preprocessor() -> CoffeePreprocessor:
    """Factory function to get a preprocessor instance."""
    return CoffeePreprocessor()


if __name__ == "__main__":
    # Example usage
    file_path = "src/data/arabica_data.csv"
    preprocessor = CoffeePreprocessor()
    X, y = preprocessor.preprocess_pipeline(file_path, is_training=True)
    
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"\nFeature columns: {len(preprocessor.get_feature_names())}")
    print(f"\nFirst few feature names: {preprocessor.get_feature_names()[:10]}")
    print(f"\nTarget columns: {preprocessor.target_columns}")