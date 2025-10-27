# ☕ Coffee Compass 🧭

**Specialty Coffee Flavor Profile Predictor using Machine Learning**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-yellow.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Predict the sensory characteristics of specialty arabica coffee based on origin, altitude, processing method, and variety. Built with domain expertise in specialty coffee and modern ML engineering practices.

🔗 **[Try the Demo](https://huggingface.co/spaces/loresico/coffee-compass)** | 📊 **[View Dataset](https://github.com/jldbc/coffee-quality-database)**

---

## 🎯 What It Does

Coffee Compass predicts **six sensory attributes** that professional cuppers use to evaluate specialty coffee:

- **Aroma** - The fragrance of ground coffee
- **Flavor** - The taste characteristics 
- **Aftertaste** - The lingering taste after swallowing
- **Acidity** - The brightness and liveliness (like wine acidity)
- **Body** - The texture and weight in the mouth
- **Balance** - The harmony between flavors

### Why This Matters

For coffee buyers, roasters, and enthusiasts, understanding expected flavor profiles helps:
- ✅ Buyers make informed sourcing decisions
- ✅ Roasters match beans to their desired profiles
- ✅ Consumers discover coffees matching their taste preferences
- ✅ Understand how origin and processing impact flavor

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+ (Python 3.13 has compatibility issues with some ML libraries)
- [UV package manager](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/loresico/coffee-compass.git
cd coffee-compass

# Install dependencies with UV
./setup.sh

# Activate the local virtual environment 
source .venv/bin/activate

# Train the model (with hyperparameter optimization)
python -m coffee_compass.scripts.train

# Launch the Gradio interface
python -m coffee_compass.app
```

Visit `http://127.0.0.1:7860/` to use the app!

---

## 📊 Model Performance

**Multi-Output XGBoost Regressor**
- **Test R²:** 0.1894 (explains ~19% of variance)
- **Test RMSE:** 0.2775 points (on 7-10 scale)
- **Training samples:** ~1,100 coffees
- **Features:** 77 (after one-hot encoding)
- **Targets:** 6 sensory scores

### Interpretation

**R² = 0.19** means our features (origin, altitude, processing, variety) explain 19% of coffee flavor variance. The remaining 81% comes from factors not in the dataset:
- Roast level (huge impact!)
- Freshness & storage
- Processing details (fermentation time, drying method)
- Specific farm practices & terroir
- Weather conditions during growing season
- Brewing method & water quality

**RMSE = 0.28** means predictions are typically within ±0.28 points on a 7-10 scale - solid accuracy given the limited features available.

### Top Features by Importance

| Feature | Importance | Insight |
|---------|-----------|---------|
| Ethiopia (origin) | 0.0796 | Distinctive bright, floral, tea-like profiles |
| Variety: Caturra | 0.0515 | Genetic variety significantly influences taste |
| Premier origin flag | 0.0509 | Top origins (Ethiopia/Kenya/Panama) matter |
| Honduras (origin) | 0.0489 | Balanced, sweet characteristics |
| Mexico (origin) | 0.0457 | Mild, nutty profiles |
| Variety: Typica | 0.0350 | Heritage variety with clean flavor |
| Altitude: Medium | 0.0348 | 1200-1500m sweet spot for many origins |
| Processing complexity | 0.0323 | Natural > Honey > Washed impact |
| Altitude (meters) | 0.0309 | Higher altitude → denser beans |
| Variety: Catuai | 0.0285 | Brazilian variety, smooth profile |

**Key insight:** Origin (Ethiopia, Honduras, Mexico) dominates, but domain-engineered features (premier_origin, altitude_category, processing_complexity) all appear in top 10, validating the feature engineering approach!

---

## 🏗️ Technical Architecture

### Project Structure

```
coffee-compass/                       # Repository
├── coffee_compass/                   # Python package
│   ├── data/
│   │   ├── preprocess.py            # Feature engineering
│   │   └── raw/
│   │       └── arabica_data.csv     # CQI dataset
│   ├── models/
│   │   ├── flavor_predictor.py      # Multi-output XGBoost
│   │   └── saved/
│   │       └── flavor_predictor.joblib
│   ├── scripts/
│   │   └── train.py                 # Training with optimization
│   ├── utils/
│   ├── app.py                        # Gradio interface (optimized UI)
├── pyproject.toml                    # Dependencies
└── README.md
```

### Tech Stack

- **ML Framework:** XGBoost 2.0+ (gradient boosting)
- **Optimization:** Optuna (Bayesian hyperparameter tuning)
- **Interface:** Gradio 4.0+ (web UI)
- **Data Processing:** Pandas, NumPy, scikit-learn
- **Visualization:** Plotly (interactive radar charts)
- **Package Manager:** UV (fast Python packaging)

---

## 🧠 How It Works

### 1. Feature Engineering

Starting with **4 base features**:
- Country of Origin (36 unique countries)
- Altitude (meters above sea level)
- Processing Method (Washed, Natural, Honey)
- Variety (28 coffee varieties)

We apply **domain-driven feature engineering**:

```python
# Altitude categories (specialty coffee grading)
- Low: <1200m
- Medium: 1200-1500m  
- High: 1500-1800m
- Very High: >1800m

# Processing complexity score
- Natural: 3 (complex, risky, unique flavors)
- Honey: 2 (medium complexity)
- Washed: 1 (clean, consistent)

# Premier origin indicator
- Ethiopia, Kenya, Panama, Colombia, Costa Rica = 1
- Others = 0

# Premium variety indicator
- Geisha, Bourbon, Typica, SL28, SL34 = 1
- Others = 0
```

### 2. One-Hot Encoding

Categorical features are converted to binary columns (one-hot encoding):
- **4 input features** → **77 numerical features** after encoding
- Allows tree-based models to learn category-specific patterns
- Maintains interpretability (see exactly which countries matter)

### 3. Multi-Output XGBoost

```python
# Predicts all 6 sensory scores simultaneously
model = MultiOutputRegressor(XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
))
```

**Why XGBoost?**
- Handles non-linear relationships
- Robust to missing data
- Feature importance built-in
- Fast training & prediction

---

## 🎨 Features

### Interactive Web Interface

- **Modern Coffee-Themed UI** - Custom CSS with brown/cream color scheme
- **Responsive Design** - Works on mobile and desktop
- **Real-Time Predictions** - Instant feedback on parameter changes
- **Interactive Radar Charts** - Plotly visualization with reference lines
- **Contextual Insights** - Coffee knowledge annotations based on inputs
- **Example Presets** - Quick start with specialty coffee origins

### Data Quality

- ✅ Altitude validation (removes >3000m or <0m outliers - **improved R² by 10%!**)
- ✅ Missing value handling (country-specific medians)
- ✅ Processing method standardization
- ✅ Variety normalization

---

## 🔬 Advanced Usage

### Adjusting Hyperparameter Optimization

The training script uses **Bayesian optimization (Optuna)** by default with 50 trials. To adjust:

```python
# In coffee_compass/scripts/train.py, modify:
study.optimize(
    lambda trial: objective(trial, X_train, y_train, X_val, y_val),
    n_trials=50,  # Change this number
    show_progress_bar=True
)
```

**Trade-offs:**
- More trials (100+) = better parameters but slower training
- Fewer trials (20-30) = faster but might miss optimal values
- 50 trials is a good balance (~10 minutes)

### Alternative Encoding Methods

The project uses one-hot encoding for interpretability. For high-cardinality features or when accuracy matters more than interpretability, consider:

- **Target Encoding** - Replace categories with average target values
- **Label Encoding** - Assign ordinal numbers (works with trees)
- **Embeddings** - Neural network learned representations

See `docs/feature_encoding_guide.html` for detailed comparison.

---

## 📊 Dataset

**Coffee Quality Institute (CQI) Arabica Reviews**
- **Source:** [Coffee Quality Database](https://github.com/jldbc/coffee-quality-database)
- **Samples:** 1,311 specialty coffee evaluations
- **Evaluators:** Certified Q Graders (professional cuppers)
- **Features:** Origin, altitude, variety, processing, sensory scores
- **License:** Open source

---

## 🛠️ Development

### Running Tests

```bash
# Coming soon
pytest tests/
```

### Code Quality

```bash
# Format code
black coffee_compass/

# Lint
ruff check coffee_compass/

# Type checking
mypy coffee_compass/
```

---

## 🚀 Deployment

### Hugging Face Spaces (Recommended)

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Push your code:
```bash
git push hf main
```
3. Add `app.py` as the entry point
4. Space will auto-deploy!

### Docker

```bash
# Build
docker build -t coffee-compass .

# Run
docker run -p 7860:7860 coffee-compass
```

---

## 🎓 Technical Insights

### Why One-Hot Encoding?

**Chosen approach:**
- ✅ No ordinal assumptions (countries aren't "ranked")
- ✅ Highly interpretable (see exact country impact)
- ✅ Works great with XGBoost
- ✅ 77 features is manageable

**When to use alternatives:**
- Target encoding: High cardinality (500+ categories)
- Embeddings: Deep learning, massive datasets
- Label encoding: Truly ordinal data only

### Model Limitations

**What we DON'T account for:**
- Roast level (light/medium/dark)
- Bean freshness (weeks since roasting)
- Processing details (fermentation hours, drying method)
- Brewing parameters (temperature, grind size, water)
- Cupper individual preferences

**Result:** Model captures ~19% of variance. The other 81% comes from factors not in the dataset.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add more features (if available in extended datasets)
- Experiment with alternative models (Random Forest, Neural Networks)
- Improve UI/UX
- Add data visualization dashboard
- Implement quality score prediction (Phase 2)

---

## 📄 License

MIT License - feel free to use for learning and projects!

---

## 👤 Author

**Lorenzo Siconolfi**
- Aerospace Engineer → ML Engineering
- PhD in Computational Fluid Dynamics, University of Pisa
- Postdoctoral Researcher @ EPFL
- CFD Software Developer @ Stake F1 Team
- Specialty coffee enthusiast ☕

Combining computational optimization expertise with machine learning and a passion for specialty coffee.

🔗 [LinkedIn](www.linkedin.com/in/lorenzo-siconolfi-1449255b) | 💻 [GitHub](your-github) | ✉️ [Email](siconolfi.lorenzo@gmail.com)

---

## 🙏 Acknowledgments

- **Coffee Quality Institute** for the dataset
- **Specialty coffee community** for domain knowledge
- **Open source ML community** (XGBoost, scikit-learn, Gradio)

---

**Built with ❤️ and ☕**

*Note: This is a machine learning model trained on historical data. Actual coffee flavor depends on many factors including roasting, brewing method, freshness, and water quality. Use predictions as a guide, not gospel!*
