# ☕ Coffee Compass 🧭

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/badge/uv-enabled-blue)](https://github.com/astral-sh/uv)
[![Gradio](https://img.shields.io/badge/gradio-4.0+-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



**Specialty Coffee Flavor Profile Predictor**

Predict the sensory characteristics of specialty arabica coffee based on origin, altitude, processing method, and variety. Built with domain expertise in specialty coffee and machine learning.

🔗 **[Try the Demo](your-huggingface-space-link)** | 📊 **[Dataset Source](https://github.com/jldbc/coffee-quality-database)**

---

## 🎯 What It Does

Coffee Compass predicts six sensory attributes that coffee professionals use to evaluate specialty coffee:

- **Aroma** - The fragrance of ground coffee
- **Flavor** - The taste characteristics 
- **Aftertaste** - The lingering taste after swallowing
- **Acidity** - The brightness and liveliness (like wine acidity)
- **Body** - The texture and weight in the mouth
- **Balance** - The harmony between flavors

### Why This Matters

For coffee buyers, roasters, and enthusiasts, understanding expected flavor profiles before purchase helps:
- Buyers make informed sourcing decisions
- Roasters match beans to their desired profiles
- Consumers discover coffees matching their taste preferences

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- UV (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/coffee-compass.git
cd coffee-compass

# Install dependencies with UV
uv sync

# Create data and model directories
mkdir -p coffee_compass/data/raw coffee_compass/models/saved

# Download the dataset
curl -o coffee_compass/data/raw/arabica_data.csv \
  https://raw.githubusercontent.com/jldbc/coffee-quality-database/master/data/arabica_data_cleaned.csv

# Train the model
python -m coffee_compass.scripts.train

# Launch the Gradio interface
python -m coffee_compass.app
```

**⚠️ Important:** Run all commands from the **project root directory** (`coffee-compass/`).

Visit `http://localhost:7860` to use the app!

---

## 🏗️ Project Structure

```
coffee-compass/                          # Repository (GitHub name with hyphen)
├── coffee_compass/                      # Package (Python name with underscore)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py               # Feature engineering
│   │   └── raw/
│   │       └── arabica_data.csv        # Dataset
│   ├── models/
│   │   ├── __init__.py
│   │   ├── flavor_predictor.py         # XGBoost model
│   │   └── saved/
│   │       └── flavor_predictor.joblib # Trained model
│   ├── scripts/
│   │   ├── __init__.py
│   │   └── train.py                    # Training script
│   ├── utils/
│   │   ├── __init__.py
│   │   └── visualization.py
│   └── app.py                           # Gradio interface
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## 🧠 How It Works

### 1. Feature Engineering

The model uses domain expertise to create meaningful features:

- **Altitude Categories**: High (>1800m), Medium (1400-1800m), Low (<1400m)
  - *Why:* Higher altitude → slower maturation → denser beans → more complex flavors
  
- **Processing Complexity**: Natural (3) > Honey (2) > Washed (1)
  - *Why:* Natural processing is riskier but can produce unique fruity flavors
  
- **Premier Origins**: Ethiopia, Kenya, Panama, Colombia, Costa Rica
  - *Why:* Historical reputation for high-quality specialty coffee
  
- **Premium Varieties**: Geisha, Bourbon, Typica, SL28, SL34
  - *Why:* Genetic varieties known for exceptional cup quality

### 2. Model Architecture

**Multi-Output XGBoost Regressor**
- Predicts 6 sensory scores simultaneously
- Captures relationships between growing conditions and flavor
- ~200 trees with careful regularization to avoid overfitting

### 3. Performance

- **Test R²**: ~0.75 (varies by attribute)
- **Test RMSE**: ~0.25 points (on 0-10 scale)
- Best predictions: Flavor, Balance
- Most challenging: Body (more subjective)

---

## 🎨 Features

### Interactive Prediction
- Select country, altitude, processing method, and variety
- Get instant flavor profile predictions
- Visualize results with radar charts

### Explainability
- SHAP values show which features drive predictions
- Feature importance rankings
- Coffee expertise annotations

### Example Predictions

**Ethiopian Natural @ 2000m**
- High acidity (bright, tea-like)
- Fruity, wine-like characteristics
- Complex flavor profile

**Colombian Washed @ 1700m**
- Balanced profile
- Clean, sweet cup
- Medium body

---

## 📊 Dataset

**Coffee Quality Institute (CQI) Arabica Reviews**
- 1,300+ specialty coffee evaluations
- Professional cupper scores (certified Q Graders)
- Features: origin, altitude, variety, processing, sensory scores

**Data Source**: [Coffee Quality Database](https://github.com/jldbc/coffee-quality-database)

---

## 🔬 Technical Details

### Dependencies
- **gradio**: Web interface
- **xgboost**: Gradient boosting model
- **scikit-learn**: ML utilities
- **pandas**: Data manipulation
- **shap**: Model interpretability
- **plotly**: Interactive visualizations

### Model Hyperparameters
```python
{
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

---

## 🚧 Roadmap

### Phase 2: Quality Predictor
- [ ] Add overall quality score prediction (Total.Cup.Points)
- [ ] Confidence intervals for predictions
- [ ] Quality grade classification (Specialty vs Commodity)

### Future Enhancements
- [ ] Coffee recommendation system ("Find me a bright, fruity coffee")
- [ ] Batch predictions from CSV upload
- [ ] Price prediction based on quality
- [ ] Region-specific models for better accuracy
- [ ] Integration with coffee retailer APIs

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional feature engineering ideas
- Model performance optimization
- UI/UX enhancements
- Documentation

---

## 👤 Author

Lorenzo Siconolfi
- Aerospace Engineer → ML Engineering
- PhD in Computational Fluid Dynamics
- CFD Software Developer @ Stake F1 Team
- Specialty coffee enthusiast

Combining computational optimization expertise with machine learning and a passion for specialty coffee.

🔗 [LinkedIn](your-linkedin) | 💻 [GitHub](your-github) | ✉️ [Email](your-email)

---

## 📄 License

MIT License - feel free to use for learning and projects!

---

## 🙏 Acknowledgments

- Coffee Quality Institute for the dataset
- Specialty coffee community for domain knowledge
- Open source ML community

---

**Built with ❤️ and ☕**

*Note: This is a machine learning model trained on historical data. Actual coffee flavor depends on many factors including roasting, brewing method, freshness, and water quality. Use predictions as a guide, not gospel!*

