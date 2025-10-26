"""
Coffee Compass - Specialty Coffee Flavor Profile Predictor
"""

__version__ = "0.1.0"
__author__ = "Lorenzo Siconolfi"

from coffee_compass.models.flavor_predictor import FlavorPredictor
from coffee_compass.data.preprocess import CoffeePreprocessor

__all__ = ["FlavorPredictor", "CoffeePreprocessor"]