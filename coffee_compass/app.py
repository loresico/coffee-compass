"""
Gradio interface for Coffee Compass - Specialty Coffee Flavor Predictor
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


class CoffeeCompassApp:
    """Gradio application for coffee flavor prediction."""
    
    def __init__(self, model_path: str, data_path: str):
        """Initialize the app with trained model and preprocessor."""
        print("Loading model and preprocessor...")
        self.predictor, self.preprocessor = FlavorPredictor.load(model_path)
        if self.preprocessor is None:
            self.preprocessor = CoffeePreprocessor()
        
        # Load data to get valid options
        df = self.preprocessor.load_data(data_path)
        df = self.preprocessor.clean_data(df)
        
        # Extract unique values for dropdowns
        # Extract unique values for dropdowns - filter out NaN
        self.countries = sorted([c for c in df['Country.of.Origin'].unique() if pd.notna(c)])
        self.processing_methods = sorted([p for p in df['Processing.Method'].unique() if pd.notna(p)])
        self.varieties = sorted([v for v in df['Variety'].unique() if pd.notna(v)])
        
        # Altitude range
        self.altitude_min = int(df['altitude_mean_meters'].min())
        self.altitude_max = int(df['altitude_mean_meters'].max())
        
        print("App initialized successfully!")
    
    def create_feature_vector(
        self, 
        country: str, 
        altitude: float, 
        processing: str, 
        variety: str
    ) -> pd.DataFrame:
        """Create a properly formatted feature vector for prediction."""
        # Create a minimal dataframe with user inputs
        input_data = pd.DataFrame([{
            'Species': 'Arabica',
            'Country.of.Origin': country,
            'altitude_mean_meters': altitude,
            'Processing.Method': processing,
            'Variety': variety,
            # Add dummy values for required columns
            'Aroma': 0, 'Flavor': 0, 'Aftertaste': 0,
            'Acidity': 0, 'Body': 0, 'Balance': 0
        }])
        
        # Apply feature engineering
        input_data = self.preprocessor.engineer_features(input_data)
        
        # Prepare features (don't fit, just transform)
        X, _ = self.preprocessor.prepare_features(input_data, is_training=False)
        
        return X
    
    def predict_flavor(
        self,
        country: str,
        altitude: float,
        processing: str,
        variety: str
    ):
        """Make prediction and return results with visualization."""
        try:
            # Create feature vector
            X = self.create_feature_vector(country, altitude, processing, variety)
            
            # Make prediction
            prediction = self.predictor.predict(X)
            
            # Create radar chart
            fig = self.create_radar_chart(prediction.iloc[0])
            
            # Format text output
            output_text = self.format_prediction(prediction.iloc[0], country, altitude, processing)
            
            return fig, output_text
            
        except Exception as e:
            error_msg = f"Error making prediction: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def create_radar_chart(self, scores: pd.Series) -> go.Figure:
        """Create a radar chart for flavor profile."""
        categories = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
        values = [scores[cat] for cat in categories]
        
        # Close the radar chart
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name='Predicted Profile',
            line=dict(color='#8B4513', width=2),
            fillcolor='rgba(139, 69, 19, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[7, 9.5],  # Typical range for specialty coffee
                    tickfont=dict(size=10)
                )
            ),
            showlegend=False,
            title="Predicted Flavor Profile",
            height=500,
            font=dict(size=12)
        )
        
        return fig
    
    def format_prediction(
        self, 
        scores: pd.Series, 
        country: str, 
        altitude: float, 
        processing: str
    ) -> str:
        """Format prediction results as readable text."""
        output = f"""
### 🎯 Predicted Flavor Profile

**Coffee Characteristics:**
- **Origin:** {country}
- **Altitude:** {altitude:.0f}m
- **Processing:** {processing}

**Sensory Scores (Scale: 0-10):**
- **Aroma:** {scores['Aroma']:.2f} - The fragrance of the coffee
- **Flavor:** {scores['Flavor']:.2f} - The taste characteristics
- **Aftertaste:** {scores['Aftertaste']:.2f} - The lingering taste
- **Acidity:** {scores['Acidity']:.2f} - The brightness and liveliness
- **Body:** {scores['Body']:.2f} - The texture and weight
- **Balance:** {scores['Balance']:.2f} - The harmony of flavors

**Overall Quality Indicator:** {scores.mean():.2f}/10

---

*💡 Tip: Specialty coffee scores typically range from 80-90+ points. 
Higher altitude and careful processing generally lead to more complex flavors.*
"""
        
        # Add contextual interpretation
        if scores['Acidity'] > 8.5:
            output += "\n✨ This coffee is predicted to have bright, vibrant acidity - typical of high-altitude African coffees!"
        
        if processing == "Natural":
            output += "\n🍓 Natural processing often brings fruity, wine-like characteristics."
        
        if altitude > 1800:
            output += "\n⛰️ High altitude typically produces denser beans with more complex flavors."
        
        return output
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(title="Coffee Compass ☕🧭", theme=gr.themes.Soft()) as app:
            gr.Markdown("""
            # ☕ Coffee Compass 🧭
            ## Specialty Coffee Flavor Profile Predictor
            
            Predict the sensory characteristics of specialty coffee based on origin, altitude, processing method, and variety.
            Built with domain expertise and machine learning (XGBoost).
            
            *Created by [Your Name] - Combining computational engineering with coffee passion*
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Coffee Parameters")
                    
                    country_input = gr.Dropdown(
                        choices=self.countries,
                        label="Country of Origin",
                        value="Ethiopia",
                        info="Where the coffee was grown"
                    )
                    
                    altitude_input = gr.Slider(
                        minimum=self.altitude_min,
                        maximum=self.altitude_max,
                        value=1800,
                        step=50,
                        label="Altitude (meters)",
                        info="Higher altitude → denser beans → more complexity"
                    )
                    
                    processing_input = gr.Dropdown(
                        choices=self.processing_methods,
                        label="Processing Method",
                        value="Washed",
                        info="How the coffee cherry was processed"
                    )
                    
                    variety_input = gr.Dropdown(
                        choices=self.varieties,
                        label="Coffee Variety",
                        value="Other",
                        info="Genetic variety of the coffee plant"
                    )
                    
                    predict_btn = gr.Button("🔮 Predict Flavor Profile", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("### Predicted Flavor Profile")
                    radar_plot = gr.Plot(label="Sensory Radar Chart")
            
            with gr.Row():
                output_text = gr.Markdown(label="Prediction Details")
            
            # Connect the prediction function
            predict_btn.click(
                fn=self.predict_flavor,
                inputs=[country_input, altitude_input, processing_input, variety_input],
                outputs=[radar_plot, output_text]
            )
            
            # Add examples
            gr.Examples(
                examples=[
                    ["Ethiopia", 2000, "Natural", "Other"],
                    ["Colombia", 1700, "Washed", "Bourbon"],
                    ["Kenya", 1800, "Washed", "SL28"],
                    ["Brazil", 1200, "Natural", "Bourbon"],
                    ["Costa Rica", 1600, "Honey", "Caturra"],
                ],
                inputs=[country_input, altitude_input, processing_input, variety_input],
                label="Try these examples:"
            )
            
            gr.Markdown("""
            ---
            ### About This Tool
            
            **Coffee Compass** uses machine learning to predict specialty coffee flavor profiles based on:
            - **Origin:** Different terroirs produce distinct characteristics
            - **Altitude:** Higher elevations typically yield denser, more complex beans
            - **Processing:** Washed, Natural, or Honey processing dramatically affects flavor
            - **Variety:** Genetic differences influence taste potential
            
            The model is trained on Coffee Quality Institute cupping scores from specialty coffee evaluations.
            
            **Note:** Predictions are estimates based on historical data. Actual flavor depends on many factors
            including roasting, brewing method, and individual coffee lot characteristics.
            
            📊 **Model:** Multi-output XGBoost Regressor  
            🎯 **Targets:** 6 sensory attributes (Aroma, Flavor, Aftertaste, Acidity, Body, Balance)  
            📈 **Features:** Origin, altitude, processing, variety + engineered features  
            
            ---
            *Built with ❤️ and ☕ by [Your Name] | [GitHub](your-github-link) | [LinkedIn](your-linkedin)*
            """)
        
        return app


def main():
    """Main function to launch the Gradio app."""
    # Paths - work from project root
    MODEL_PATH = SCRIPT_DIR / "models" / "saved" / "flavor_predictor.joblib"
    DATA_PATH = SCRIPT_DIR / "data" / "raw" / "arabica_data.csv"
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Looking for model at: {MODEL_PATH}")
    print(f"Looking for data at: {DATA_PATH}")
    
    if not MODEL_PATH.exists():
        print(f"\n❌ ERROR: Model not found at {MODEL_PATH}")
        print("Please train the model first:")
        print("  uv run python train.py")
        return
    
    if not DATA_PATH.exists():
        print(f"\n❌ ERROR: Data not found at {DATA_PATH}")
        print("Please download the dataset first")
        return
    
    # Initialize and launch app
    app = CoffeeCompassApp(str(MODEL_PATH), str(DATA_PATH))
    interface = app.create_interface()
    
    # Launch with public link for sharing
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        share=False,
        inbrowser=True,
    )


if __name__ == "__main__":
        
    main()