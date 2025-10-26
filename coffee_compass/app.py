"""
Improved Gradio interface for Coffee Compass with better UI/UX.
Run from project root: python -m coffee_compass.app_improved
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Get paths
SCRIPT_DIR = Path(__file__).parent  # coffee_compass/
PROJECT_ROOT = SCRIPT_DIR.parent     # coffee-compass/

from coffee_compass.models.flavor_predictor import FlavorPredictor
from coffee_compass.data.preprocess import CoffeePreprocessor


# Custom CSS for coffee-themed styling
custom_css = """
/* Coffee-themed color scheme */
:root {
    --primary-color: #8B4513;
    --secondary-color: #D2691E;
    --accent-color: #CD853F;
    --bg-light: #FFF8DC;
    --text-dark: #3E2723;
}

/* Header styling */
.app-header {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #8B4513 0%, #FFF8DC 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.app-header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    font-weight: bold;
}

.app-header p {
    font-size: 1.1rem;
    opacity: 0.9;
}

/* Input section */
.input-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-left: 4px solid var(--primary-color);
}

/* Results section */
.results-card {
    background: var(--bg-light);
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    border-left: 4px solid var(--accent-color);
}

/* Button styling */
.predict-button {
    background: linear-gradient(135deg, #8B4513 0%, #D2691E 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    font-size: 1.1rem !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
    transition: transform 0.2s !important;
}

.predict-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
}

/* Info boxes */
.info-box {
    background: #E3F2FD;
    border-left: 4px solid #2196F3;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

/* Examples styling */
.examples {
    margin-top: 2rem;
    padding: 1rem;
    background: #F5F5F5;
    border-radius: 8px;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 2px solid #DDD;
    color: #666;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .app-header h1 {
        font-size: 1.8rem;
    }
}
"""


class CoffeeCompassApp:
    """Enhanced Gradio application for coffee flavor prediction."""
    
    def __init__(self, model_path: str, data_path: str):
        """Initialize the app with trained model and preprocessor."""
        print("Loading model and preprocessor...")
        self.predictor, self.preprocessor = FlavorPredictor.load(model_path)
        if self.preprocessor is None:
            self.preprocessor = CoffeePreprocessor()
        
        # Load data to get valid options
        df = self.preprocessor.load_data(data_path)
        df = self.preprocessor.clean_data(df)
        
        # Extract unique values - filter out NaN
        self.countries = sorted([c for c in df['Country.of.Origin'].unique() if pd.notna(c)])
        self.processing_methods = sorted([p for p in df['Processing.Method'].unique() if pd.notna(p)])
        self.varieties = sorted([v for v in df['Variety'].unique() if pd.notna(v)])
        
        # Altitude range
        self.altitude_min = int(df['altitude_mean_meters'].min())
        self.altitude_max = int(df['altitude_mean_meters'].max())
        
        print("App initialized successfully!")
    
    def create_feature_vector(self, country: str, altitude: float, processing: str, variety: str) -> pd.DataFrame:
        """Create a properly formatted feature vector for prediction."""
        input_data = pd.DataFrame([{
            'Species': 'Arabica',
            'Country.of.Origin': country,
            'altitude_mean_meters': altitude,
            'Processing.Method': processing,
            'Variety': variety,
            'Aroma': 0, 'Flavor': 0, 'Aftertaste': 0,
            'Acidity': 0, 'Body': 0, 'Balance': 0
        }])
        
        input_data = self.preprocessor.engineer_features(input_data)
        X, _ = self.preprocessor.prepare_features(input_data, is_training=False)
        
        return X
    
    def predict_flavor(self, country: str, altitude: float, processing: str, variety: str):
        """Make prediction and return results with visualization."""
        try:
            X = self.create_feature_vector(country, altitude, processing, variety)
            prediction = self.predictor.predict(X)
            
            fig = self.create_radar_chart(prediction.iloc[0])
            
            # Split output into separate components
            characteristics = self.format_characteristics(country, altitude, processing, variety)
            scores = self.format_scores(prediction.iloc[0])
            insights = self.format_insights(prediction.iloc[0], country, altitude, processing)
            
            return fig, characteristics, scores, insights
            
        except Exception as e:
            error_msg = f"❌ Error making prediction: {str(e)}"
            print(error_msg)
            return None, error_msg, "", ""
    
    def create_radar_chart(self, scores: pd.Series) -> go.Figure:
        """Create an enhanced radar chart for flavor profile."""
        categories = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
        values = [scores[cat] for cat in categories]
        
        # Close the radar chart
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]
        
        fig = go.Figure()
        
        # Main trace with gradient fill
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name='Predicted Profile',
            line=dict(color='#8B4513', width=3),
            fillcolor='rgba(139, 69, 19, 0.4)',
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.2f}<extra></extra>'
        ))
        
        # Add reference circle at 8.0 (good specialty coffee)
        reference = [8.0] * 7
        fig.add_trace(go.Scatterpolar(
            r=reference,
            theta=categories_closed,
            name='Specialty Grade (8.0)',
            line=dict(color='rgba(0, 150, 0, 0.5)', width=2, dash='dash'),
            showlegend=True
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[7, 9.5],
                    tickfont=dict(size=11),
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(0,0,0,0.1)'
                )
            ),
            showlegend=True,
            legend=dict(
            x=1.0,
            y=.5,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="black"
                ),
            ),
            height=600,
            font=dict(size=12),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=80, b=40)
        )
        
        return fig
    
    def format_characteristics(self, country: str, altitude: float, processing: str, variety: str) -> str:
        """Format coffee characteristics."""
        return f"""
    ### 📍 Coffee Characteristics

    - **Origin:** {country} 🌍
    - **Altitude:** {altitude:.0f}m ⛰️
    - **Processing:** {processing} ⚙️
    - **Variety:** {variety} 🌱
    """

    def format_scores(self, scores: pd.Series) -> str:
        """Format sensory scores as a clean table."""
        avg_score = scores.mean()
        
        # Quality rating
        if avg_score >= 8.5:
            quality = "⭐ Exceptional Quality"
        elif avg_score >= 8.0:
            quality = "✨ Specialty Grade"
        else:
            quality = "☕ Good Quality"
        
        return f"""
    ### {quality}
    **Average: {avg_score:.2f}/10**

    | Attribute | Score |
    |-----------|-------|
    | **Aroma** 👃 | {scores['Aroma']:.2f} |
    | **Flavor** 👅 | {scores['Flavor']:.2f} |
    | **Aftertaste** ⏱️ | {scores['Aftertaste']:.2f} |
    | **Acidity** ⚡ | {scores['Acidity']:.2f} |
    | **Body** 💪 | {scores['Body']:.2f} |
    | **Balance** ⚖️ | {scores['Balance']:.2f} |
    """

    def format_insights(self, scores: pd.Series, country: str, altitude: float, processing: str) -> str:
        """Format tasting notes and insights."""
        output = "### 💡 Tasting Notes\n\n"
        
        if scores['Acidity'] > 8.5:
            output += "✨ **Bright & Lively** - High acidity typical of high-altitude coffees\n\n"
        
        if processing == "Natural":
            output += "🍓 **Fruity & Complex** - Natural processing brings wine-like characteristics\n\n"
        
        if altitude > 1800:
            output += "⛰️ **High Altitude** - Dense beans with complex, nuanced flavors\n\n"
        
        if country in ['Ethiopia', 'Kenya']:
            output += f"🌍 **{country}** - Known for distinctive, bright, and often floral notes\n\n"
        
        output += "\n*Predictions based on growing conditions. Actual flavor depends on roast, freshness, and brewing.*"
        
        return output
    
    def create_interface(self) -> gr.Blocks:
        """Create the enhanced Gradio interface."""
        
        # Custom theme
        theme = gr.themes.Soft(
            primary_hue="stone",
            secondary_hue="stone",
            neutral_hue="stone",
            font=("Inter", "sans-serif")
        ).set(
            button_primary_background_fill="#8B4513",
            button_primary_background_fill_hover="#A0522D",
        )
        
        with gr.Blocks(theme=theme, css=custom_css, title="Coffee Compass ☕") as app:
            
            # Header
            gr.HTML("""
                <div class="app-header">
                    <h1>☕ Coffee Compass 🧭</h1>
                    <p>Predict specialty coffee flavor profiles using machine learning</p>
                    <p style="font-size: 0.9rem; opacity: 0.8;">
                        Origin • Altitude • Processing • Variety → Flavor Predictions
                    </p>
                </div>
            """)
            
            # Main interface
            with gr.Row():
                # Left column - Inputs
                with gr.Column(scale=1):
                    gr.HTML('<h3 style="color: #5D4037; font-weight: 700; font-size: 1.3rem; margin-bottom: 1.5rem; padding-bottom: 0.75rem; border-bottom: 2px solid #E0E0E0;">⚙️ Coffee Parameters</h3>')
                                        
                    country_input = gr.Dropdown(
                        choices=self.countries,
                        label="🌍 Country of Origin",
                        value="Ethiopia",
                        info="Where the coffee was grown"
                    )
                    
                    altitude_input = gr.Slider(
                        minimum=self.altitude_min,
                        maximum=self.altitude_max,
                        value=1800,
                        step=50,
                        label="⛰️ Altitude (meters)",
                        info="Higher altitude → denser beans → more complexity"
                    )
                    
                    processing_input = gr.Dropdown(
                        choices=self.processing_methods,
                        label="⚙️ Processing Method",
                        value="Washed",
                        info="How the coffee cherry was processed after picking"
                    )
                    
                    variety_input = gr.Dropdown(
                        choices=self.varieties,
                        label="🌱 Coffee Variety",
                        value="Other",
                        info="Genetic variety of the coffee plant"
                    )
                    
                    predict_btn = gr.Button(
                        "🔮 Predict Flavor Profile",
                        variant="primary",
                        size="lg",
                        elem_classes=["predict-button"]
                    )
                    gr.HTML('</div>')
                
                # Right column - Results
                with gr.Column(scale=1):
                    gr.HTML('''<h3 style="color: #5D4037; 
                            font-weight: 700; 
                            font-size: 1.3rem; 
                            margin-bottom: 1.5rem; 
                            padding-bottom: 0.75rem; 
                            border-bottom: 2px solid #E0E0E0;">📊 Predicted Flavor Profile</h3>'
                            ''')
                    radar_plot = gr.Plot(label="")
                    gr.HTML('</div>')
            
            # Detailed results below
            with gr.Row():
                with gr.Column(scale=1):
                    characteristics_output = gr.Markdown(label="")
                
                with gr.Column(scale=1):
                    scores_output = gr.Markdown(label="")
                
                with gr.Column(scale=1):
                    insights_output = gr.Markdown(label="")
            
            # Examples
            gr.Markdown("### 🎯 Try These Examples:")
            gr.Examples(
                examples=[
                    ["Ethiopia", 2000, "Natural", "Other"],
                    ["Colombia", 1700, "Washed", "Bourbon"],
                    ["Kenya", 1800, "Washed", "SL28"],
                    ["Brazil", 1200, "Natural", "Bourbon"],
                    ["Costa Rica", 1600, "Honey", "Caturra"],
                    ["Guatemala", 1500, "Washed", "Bourbon"],
                ],
                inputs=[country_input, altitude_input, processing_input, variety_input],
                label="Quick start with specialty coffee origins",
                examples_per_page=6
            )
            
            predict_btn.click(
                fn=self.predict_flavor,
                inputs=[country_input, altitude_input, processing_input, variety_input],
                outputs=[radar_plot, characteristics_output, scores_output, insights_output]
            )
            
            # Footer with info
            gr.HTML("""
            <div class="app-footer">
                <h3>About Coffee Compass</h3>
                <p>Built with domain expertise in specialty coffee and machine learning (XGBoost).</p>
                <p>How it works:</strong> Our ML model predicts 6 sensory attributes based on coffee growing conditions. 
            Trained on 1,300+ specialty coffee evaluations from the Coffee Quality Institute.</p>
                <p><strong>Model:</strong> Multi-output XGBoost Regressor | 
                   <strong>Features:</strong> 77 | 
                   <strong>Targets:</strong> 6 sensory attributes</p>
                <p style="margin-top: 1rem;">
                    <strong>Data Source:</strong> Coffee Quality Institute (CQI) Arabica Reviews<br>
                    <strong>Created by:</strong> Lorenzo Siconolfi | 
                    <a href="https://github.com/your-username/coffee-compass" style="color: #8B4513;">
                        GitHub
                    </a> | 
                    <a href="https://linkedin.com/in/your-profile" style="color: #8B4513;">
                        LinkedIn
                    </a>
                </p>
                <p style="font-size: 0.9rem; color: #999; margin-top: 1rem;">
                    ⚠️ Predictions are estimates. Actual flavor depends on roasting, brewing, and freshness.
                </p>
            </div>
            """)
        
        return app


def main():
    """Main function to launch the improved Gradio app."""
    # Paths
    MODEL_PATH = SCRIPT_DIR / "models" / "saved" / "flavor_predictor_optimized.joblib"
    DATA_PATH = SCRIPT_DIR / "data" / "raw" / "arabica_data.csv"
    
    print(f"Package directory: {SCRIPT_DIR}")
    print(f"Looking for model at: {MODEL_PATH}")
    print(f"Looking for data at: {DATA_PATH}")
    
    if not MODEL_PATH.exists():
        print(f"\n❌ ERROR: Model not found at {MODEL_PATH}")
        print("Please train the model first:")
        print("  python -m coffee_compass.scripts.train")
        return
    
    if not DATA_PATH.exists():
        print(f"\n❌ ERROR: Data not found at {DATA_PATH}")
        return
    
    # Initialize and launch app
    app = CoffeeCompassApp(str(MODEL_PATH), str(DATA_PATH))
    interface = app.create_interface()
    
    # Launch
    interface.launch(
        share=False,
        show_error=True,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()