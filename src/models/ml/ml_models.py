"""
Machine Learning models for subscription fatigue prediction.
Includes XGBoost churn predictor and heterogeneous treatment effect analysis.
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
    Predicts subscriber churn risk given a proposed price increase.
    Uses XGBoost for high-dimensional feature interactions.
    """
    
    def __init__(self):
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
        Engineer features for churn prediction.
        """
        features = pd.DataFrame()
        
        if 'price' in df.columns:
            features['current_price'] = df['price']
        
        if 'price' in df.columns:
            features['price_change_pct'] = df['price'].pct_change() * 100
        
        if 'elasticity' in df.columns:
            features['historical_elasticity'] = df['elasticity']
        
        if 'subscriber_count' in df.columns:
            features['subscriber_growth_pct'] = df['subscriber_count'].pct_change() * 100
        
        if 'search_volume' in df.columns:
            features['cancel_search_volume'] = df['search_volume']
        
        # Lag features
        for col in features.columns:
            features[f'{col}_lag_1'] = features[col].shift(1)
        
        self.feature_names = features.columns.tolist()
        return features.fillna(0)
    
    def train(self, X, y):
        """
        Train the churn prediction model.
        
        Args:
            X: Feature matrix
            y: Churn rate labels
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, verbose=False)
    
    def predict(self, X):
        """Predict churn rate."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_saturation(self, current_price, proposed_price_increase_pct, 
                          historical_elasticity=None, growth_rate=None):
        """
        Predict saturation risk for a price increase.
        
        Returns:
            dict with predicted churn and risk assessment
        """
        # Create feature vector
        sample = pd.DataFrame({
            'current_price': [current_price],
            'price_change_pct': [proposed_price_increase_pct],
            'historical_elasticity': [historical_elasticity or -0.8],
            'subscriber_growth_pct': [growth_rate or 2.0],
            'cancel_search_volume': [50]  # baseline
        })
        
        # Add lag features
        for col in sample.columns:
            sample[f'{col}_lag_1'] = sample[col]
        
        try:
            predicted_churn = self.predict(sample)[0]
        except:
            # Fallback to heuristic
            predicted_churn = 2.0 + (proposed_price_increase_pct / 10) * 1.2
        
        # Risk assessment
        risk_level = "CRITICAL" if predicted_churn > 5 else \
                     "HIGH" if predicted_churn > 3 else \
                     "MODERATE" if predicted_churn > 1 else \
                     "LOW"
        
        return {
            'proposed_price_increase_pct': proposed_price_increase_pct,
            'predicted_churn_rate': predicted_churn,
            'risk_level': risk_level,
            'saturation_likely': predicted_churn > 3
        }


class HeterogeneousEffectAnalyzer:
    """
    Estimates heterogeneous treatment effects using Causal Forest.
    Different customer segments respond differently to price changes.
    """
    
    def __init__(self):
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
        Estimate Conditional Average Treatment Effect.
        
        Args:
            X: Customer features (age, tenure, engagement, etc.)
            T: Treatment (price change %)
            y: Outcome (churn or retention)
        
        Returns:
            Individual treatment effects for each customer segment
        """
        if self.model is None:
            return np.zeros(len(X))
        
        self.model.fit(y, T, X=X)
        return self.model.effect(X)
    
    def identify_sensitive_segments(self, X, effects, threshold=-0.05):
        """
        Find customer segments most sensitive to price increases.
        """
        df = pd.DataFrame(X)
        df['treatment_effect'] = effects
        
        # Segment by sensitivity
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
