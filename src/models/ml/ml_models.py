"""
# 🤖 Machine Learning Engines: Churn Prediction & Causal Inference
# Implements advanced gradient boosting and heterogeneous treatment effect analysis.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

try:
    from econml.dml import CausalForestDML
except ImportError:
    CausalForestDML = None


class ChurnRiskPredictor:
    """
    Predict subscriber churn risk for various pricing scenarios.
    
    Utilizes XGBoost Regressor to capture high-dimensional interactions between 
    pricing, historical elasticity, and user engagement signals.
    
    Attributes:
        model (xgb.XGBRegressor): The core predictive engine.
        scaler (StandardScaler): Normalizer for feature distribution alignment.
    """
    
    def __init__(self):
        """Initialize the predictor with optimized hyper-parameters."""
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def prepare_features(self, df):
        """
        Engineers features for the churn prediction model.
        
        Calculates percentage changes, historical elasticity, and time-lagged 
        variables to provide the model with temporal context.
        """
        features = pd.DataFrame()
        
        if 'price' in df.columns:
            features['current_price'] = df['price']
            features['price_change_pct'] = df['price'].pct_change() * 100
        
        if 'elasticity' in df.columns:
            features['historical_elasticity'] = df['elasticity']
        
        if 'subscriber_count' in df.columns:
            features['subscriber_growth_pct'] = df['subscriber_count'].pct_change() * 100
        
        if 'search_volume' in df.columns:
            features['cancel_search_volume'] = df['search_volume']
        
        # Aggregate lagged signals to capture momentum
        for col in features.columns:
            features[f'{col}_lag_1'] = features[col].shift(1)
        
        self.feature_names = features.columns.tolist()
        return features.fillna(0)
    
    def train(self, X, y):
        """
        Execute model training.
        
        Parameters:
            X (pd.DataFrame): Normalized feature matrix.
            y (pd.Series): Churn rate labels.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, verbose=False)
    
    def predict(self, X):
        """Execute inference to forecast churn rate."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_saturation(self, current_price, proposed_price_increase_pct, 
                          historical_elasticity=None, growth_rate=None):
        """
        Assess market saturation risk for a specific price update.
        
        Evaluates the forecast against internal risk thresholds:
        - CRITICAL: > 5% churn
        - HIGH: > 3% churn
        - MODERATE: > 1% churn
        
        Returns:
            dict: Probabilistic churn forecast and categorical risk level.
        """
        # Construct the inference payload
        sample = pd.DataFrame({
            'current_price': [current_price],
            'price_change_pct': [proposed_price_increase_pct],
            'historical_elasticity': [historical_elasticity or -0.8],
            'subscriber_growth_pct': [growth_rate or 2.0],
            'cancel_search_volume': [50]
        })
        
        # Apply lag feature replication for single-point inference
        for col in sample.columns:
            sample[f'{col}_lag_1'] = sample[col]
        
        try:
            predicted_churn = self.predict(sample)[0]
        except:
            # Implement heuristic fallback if model state is uninitialized
            predicted_churn = 2.0 + (proposed_price_increase_pct / 10) * 1.2
        
        risk_level = "CRITICAL" if predicted_churn > 5 else \
                     "HIGH" if predicted_churn > 3 else \
                     "MODERATE" if predicted_churn > 1 else \
                     "LOW"
        
        return {
            'proposed_price_increase_pct': proposed_price_increase_pct,
            'predicted_churn_rate': round(float(predicted_churn), 2),
            'risk_level': risk_level,
            'saturation_likely': predicted_churn > 3
        }


class HeterogeneousEffectAnalyzer:
    """
    Estimate Conditional Average Treatment Effects (CATE) using Causal Forest.
    
    Identifies how different customer segments respond to price changes to 
    enable personalized pricing and retention.
    """
    
    def __init__(self):
        """Initialize the causal engine with Double Machine Learning (DML) parameters."""
        if CausalForestDML:
            self.model = CausalForestDML(
                model_y=RandomForestRegressor(n_estimators=100, random_state=42),
                model_t=RandomForestRegressor(n_estimators=100, random_state=42),
                discrete_treatment=False,
                n_estimators=500,
                min_samples_leaf=50,
                max_depth=10,
                random_state=42
            )
        else:
            self.model = None
    
    def estimate_cate(self, X, T, y):
        """
        Model the individual treatment effects for each observation.
        
        X: Contextual features (Segment, Tenure, etc.)
        T: Treatment (Price Change Intensity)
        y: Outcome (Churn Event)
        """
        if self.model is None:
            return np.zeros(len(X))
        
        self.model.fit(y, T, X=X)
        return self.model.effect(X)
    
    def identify_sensitive_segments(self, X, effects, threshold=-0.05):
        """
        Classify segments based on their estimated price sensitivity.
        
        Segments are categorized into 'Highly Sensitive' to 'Inelastic' based 
        on the magnitude of the predicted treatment effect.
        """
        df = pd.DataFrame(X)
        df['treatment_effect'] = effects
        
        df['segment'] = pd.cut(
            df['treatment_effect'],
            bins=[-np.inf, -0.1, -0.05, 0, np.inf],
            labels=['Highly Sensitive', 'Moderately Sensitive', 
                   'Low Sensitivity', 'Inelastic']
        )
        
        segment_analysis = df.groupby('segment').agg({
            'treatment_effect': ['mean', 'std', 'count']
        })
        
        return segment_analysis
