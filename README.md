# 💰 Subscription Fatigue Predictor v2.0

**Optimize pricing, mitigate churn, and navigate market saturation with AI-driven economic intelligence.**

![Premium Dashboard](https://img.shields.io/badge/UI-Premium_Glassmorphic-blueviolet)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🚩 The Problem: Subscription Fatigue
In a crowded digital landscape, consumers are reaching a breaking point. Indiscriminate price increases often trigger catastrophic churn. Businesses lack the microscopic visibility needed to understand:
- **Saturation Points**: At what exact price do we lose the majority of our base?
- **Competitive Resonance**: How do our rivals' price changes impact *our* retention?
- **Diversion Paths**: Where do our users go when they leave?

## 💡 The Solution: Economic Intelligence Engine
The **Subscription Fatigue Predictor** transforms raw metrics into strategic foresight. It combines classical economic models with modern machine learning to:
- **Simulate Market Shifts**: Predict final market shares after concurrent price changes.
- **Detect Early Warning Signals**: Identify cancellation intent via search trend anomalies.
- **Optimize Revenue**: Calculate the mathematically optimal bundle price to maximize Net Present Value (NPV).

## 📊 Project Results & Impact
- **Churn Mitigation**: Forecast retention ROI to deploy the most cost-effective rescue campaigns.
- **Precision Pricing**: Identify "Consumer Surplus" to capture value without triggering fatigue.
- **Segment Intelligence**: Cluster users by psychographic sensitivity to tailor pricing strategies.

---

## 🔬 Data Universe & Methodology
The system operates on a robust data architecture, utilizing both historical records and real-time signals.

### Data Sources
- **Pricing History**: Tracks SKU-level price evolution across the competitive landscape.
- **Subscriber Metrics**: Ingests counts, ARPU, and baseline churn rates.
- **Search Intensity**: Monitors high-intent keywords (e.g., "Cancel Netflix") to capture pre-churn sentiment.
- **Synthetic Generation**: Includes a high-fidelity data generator (`setup.py`) for benchmarking and demonstration.

### Key Analytical Metrics
- **XGBoost Risk Score**: Probability of churn for specific price-hike scenarios.
- **Bertrand Nash Equilibrium**: Theoretical optimal pricing in a competitive oligopoly.
- **Z-Score Anomaly detection**: Statistical verification of search volume spikes.
- **NPV (Net Present Value)**: 12-month financial projection of bundle strategies.

---

## 🎨 Visualization Gallery
The system features a **Premium Glassmorphic Dashboard** built with Streamlit and Plotly:
- **Market Overview**: Real-time KPIs for total reach, revenue, and share.
- **Pricing Timeline**: Comparative evolution of rival service costs.
- **Market Shift Simulator**: Interactive "What-If" tool for multi-service competition.
- **Signal Heatmaps**: Intensity mapping of cancellation intent over time.

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
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialization & Execution
```bash
# Initialize the database and generate high-fidelity sample data
python setup.py

# Launch the premium interactive intelligence suite
streamlit run src/visualization/dashboard.py
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Last Updated**: January 2026 | **Version**: 2.0.0
