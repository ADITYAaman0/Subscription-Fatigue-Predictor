"""
Subscription Fatigue Predictor - Project Documentation

## Overview

This project provides a comprehensive analysis system for subscription services,
focusing on:

1. **Price Elasticity Analysis**: Measure how demand responds to price changes
2. **Market Saturation Detection**: Identify when a service has reached its pricing limit
3. **Churn Risk Prediction**: Forecast cancellation rates for different price scenarios
4. **Competitive Analysis**: Analyze pricing strategies across competing services

## Key Components

### Economic Models
- Bertrand Competition: Nash equilibrium pricing
- Elasticity Calculation: Price sensitivity metrics
- Consumer Surplus: Welfare impact analysis

### Machine Learning
- XGBoost Churn Predictor
- Heterogeneous Treatment Effects
- Temporal Forecasting

### Statistical Analysis
- Change Point Detection (PELT algorithm)
- Causal Analysis (Granger causality)
- Structural Break Identification

### Data Visualization
- Interactive Streamlit dashboard
- Plotly interactive charts
- Real-time analytics

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Generate sample data: `python main.py`
3. Run dashboard: `streamlit run src/visualization/dashboard.py`

## Project Structure

- `src/models/`: Core ML and statistical models
- `src/data/`: Data collection and processing
- `src/visualization/`: Streamlit dashboard
- `notebooks/`: Jupyter notebooks for analysis
- `tests/`: Unit tests
"""
