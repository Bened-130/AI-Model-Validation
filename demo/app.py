"""
Streamlit Demo Application
Interactive dashboard for AI Validation Framework
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
import requests
import io

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ai_validation.core.pipeline import AIValidationPipeline
from ai_validation.core.data_quality import DataQualityController
from ai_validation.core.model_builder import PredictiveModelBuilder

st.set_page_config(
    page_title="AI Validation Framework",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; color: #1f77b4; text-align: center; }
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 20px; text-align: center; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


def render_gauge(score, title):
    """Create quality gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 60], 'color': "#ffcccc"},
                {'range': [60, 80], 'color': "#ffffcc"},
                {'range': [80, 100], 'color': "#ccffcc"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': 95}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    st.markdown('<div class="main-header">🔍 AI Validation Framework</div>', unsafe_allow_html=True)
    st.markdown("**Enterprise-grade ML validation pipeline**")
    
    # Sidebar
    st.sidebar.header("Configuration")
    mode = st.sidebar.radio("Data Source", ["Generate Synthetic", "Upload CSV", "Use API"])
    
    if mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df):,} records")
        else:
            st.info("Upload a CSV file")
            return
    elif mode == "Use API":
        st.sidebar.info("Using backend API")
        df = None
    else:
        n_samples = st.sidebar.slider("Samples", 1000, 10000, 5000, 500)
        if st.button("🚀 Generate & Validate", type="primary"):
            with st.spinner("Generating..."):
                df = PredictiveModelBuilder().generate_sample_data(n_samples)
            st.success(f"Generated {len(df):,} records")
        else:
            return
    
    # Run validation
    if 'df' in locals() and df is not None:
        with st.spinner("Running validation pipeline..."):
            progress = st.progress(0)
            
            # Quality check
            progress.progress(25)
            controller = DataQualityController()
            quality = controller.validate_dataset(df)
            
            # Full pipeline
            progress.progress(50)
            pipeline = AIValidationPipeline()
            results = pipeline.run_complete_validation(df)
            
            progress.progress(100)
        
        # Display results
        st.markdown("---")
        
        # Quality metrics
        cols = st.columns(4)
        metrics = [
            (quality['overall_quality_score'], "Overall Quality"),
            (quality['checks']['completeness']['score'], "Completeness"),
            (quality['checks']['consistency']['score'], "Consistency"),
            (quality['checks']['accuracy']['score'], "Accuracy")
        ]
        
        for col, (score, title) in zip(cols, metrics):
            with col:
                st.plotly_chart(render_gauge(score, title), use_container_width=True)
        
        # Model comparison
        st.subheader("Model Performance")
        st.dataframe(results['comparison'], use_container_width=True)
        
        # Improvement
        st.metric(
            "Accuracy Improvement", 
            f"{results['improvement']:.2f}%",
            delta=f"{results['improvement']:.2f}%"
        )
        
        # Feature importance
        if results['feature_importance'] is not None:
            st.subheader("Feature Importance")
            fig = px.bar(
                results['feature_importance'].head(10),
                x='importance',
                y='feature',
                orientation='h',
                color='importance'
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()