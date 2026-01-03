# 💰 Subscription Fatigue Predictor v2.0

A high-performance intelligence engine designed to analyze subscription pricing patterns, identify market saturation points, and simulate competitive market shifts using advanced economic and machine learning models.

![Premium Dashboard](https://img.shields.io/badge/UI-Premium_Glassmorphic-blueviolet)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Project Overview

In an era of "Subscription Fatigue," businesses must navigate price increases with surgical precision. This project provides a state-of-the-art suite of tools to:
- **Quantify Demand**: Measure exact price elasticity for digital services.
- **Predict Churn**: Use XGBoost to forecast subscriber loss under price hike scenarios.
- **Map Market Shifts**: Understand where your defecting users go (Churn Diversion).
- **Optimize Bundles**: Find the mathematically optimal bundle configurations to maximize NPV.

---

## ✨ Key Features

### 🏦 Economic & Competitive Intelligence
- **Bertrand Competition Model**: Nash Equilibrium solver for oligopolistic markets.
- **Cross-Elasticity Analysis**: Measure competitive resonance between rival services.
- **Churn Diversion Mapping**: Predict subscriber migration paths (e.g., Netflix → Disney+).
- **Consumer Surplus Analyzer**: Quantify the welfare impact of pricing changes.

### 🔮 Predictive Analytics
- **XGBoost Churn Predictor**: High-accuracy regressor for cancellation forecasting.
- **Weekly Churn Detector**: Real-time Z-score based anomaly detection for search trends.
- **Market Saturation Simulator**: Interactive "What-If" analysis for price increases.
- **Psychographic Segmenter**: Cluster users by price sensitivity and churn risk.

### 💎 Premium Dashboard
- **Glassmorphic UI**: High-end dark mode inspired by modern fintech tools.
- **Interactive Plotly Visualization**: Dynamic, branded charts for all analytical tabs.
- **Strategic Insights**: AI-generated summaries for market shifts and ROI.

---

## 📁 Project Structure

```bash
subscription-fatigue-predictor/
├── src/                        # Core Application Source
│   ├── models/                 # Analytical Engines
│   │   ├── economic/           # Bertrand & Elasticity
│   │   ├── ml/                 # XGBoost & Causal Forest
│   │   ├── statistical/        # Change Point Detection
│   │   └── advanced_models.py  # Consolidated Competitive Logic
│   ├── visualization/          # Streamlit Premium Dashboard
│   ├── data/                   # Data Ingestion & Processing
│   └── utils/                  # Global Config & Constants
├── data/                       # Local SQLite Database
├── config/                     # Environment Configurations
├── docs/                       # Technical Methodology & API Ref
├── notebooks/                  # Research & Development
├── tests/                      # Comprehensive Unit Tests
├── requirements.txt            # Operational Dependencies
└── README.md                   # Project Manifesto
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/subscription-fatigue-predictor.git
cd subscription-fatigue-predictor

# Create & activate a virtual environment
# Windows:
python -m venv venv
venv\Scripts\activate
# macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialization

Initialize the database and generate high-fidelity sample data:
```bash
python setup.py
```

### 3. Launch Dashboard

Run the premium interactive intelligence suite:
```bash
streamlit run src/visualization/dashboard.py
```

---

## 🧪 Testing

Maintain architectural integrity with the built-in test suite:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest --cov=src tests/
```

---

## 📖 Methodology

This project integrates classical economic theory with modern data science:
- **Pricing Theory**: Based on Bertrand (1883) strategic interaction models.
- **Anomaly Detection**: Implements the PELT (2012) algorithm for structural break detection.
- **Causal Inference**: Uses Generalized Random Forests (2019) for segment-level treatment effects.

For a deep dive into the math, see [docs/methodology.md](docs/methodology.md).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Last Updated**: January 2026 | **Version**: 2.0.0
