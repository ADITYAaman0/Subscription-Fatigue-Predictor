"""
# 🤖 Machine Learning Engines: Churn Prediction & Causal Inference
# Implements advanced gradient boosting and heterogeneous treatment effect analysis.
"""

from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

if TYPE_CHECKING:
    from econml.dml import CausalForestDML
else:
    try:
        from econml.dml import CausalForestDML
    except ImportError:
        CausalForestDML = None  # type: ignore[assignment]


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
        Engineers features for the churn prediction model with REAL DATA PRIORITY.
        
        Calculates percentage changes, historical elasticity, and time-lagged 
        variables to provide the model with temporal context.
        
        REAL DATA HIERARCHY (in priority order):
        1. Google Trends - Real search volume for churn intent
        2. News API Sentiment - Real media sentiment scores
        3. Kaggle Datasets - Real subscriber/pricing data
        4. Synthetic data - ONLY for missing features
        
        Feature categories:
        - News sentiment scores (from news_articles table) [REAL]
        - Real search volume (from Google Trends) [REAL]
        - Competitor pricing (from real_global_streaming) [REAL]
        - Fallback synthetic data only when above are unavailable
        """
        features = pd.DataFrame()
        self.real_data_usage = {}  # Track real vs synthetic for this batch
        
        # === PRICE FEATURES (Prioritize Real Kaggle Data) ===
        if 'price' in df.columns:
            features['current_price'] = df['price']
            features['price_change_pct'] = df['price'].pct_change() * 100
            self.real_data_usage['price'] = True
        else:
            # Fallback: Generate from competitor baseline
            if 'competitor_avg_price' in df.columns:
                features['current_price'] = df['competitor_avg_price'] * 0.95
            else:
                features['current_price'] = 10.99  # Netflix baseline
            features['price_change_pct'] = np.random.uniform(-2, 5, len(df))
            self.real_data_usage['price'] = False
        
        # === ELASTICITY (Real from Kaggle) ===
        if 'elasticity' in df.columns:
            features['historical_elasticity'] = df['elasticity']
            self.real_data_usage['elasticity'] = True
        else:
            # Synthetic: Industry standard elasticity
            features['historical_elasticity'] = np.random.normal(-0.8, 0.3, len(df))
            self.real_data_usage['elasticity'] = False
        
        # === SUBSCRIBER METRICS (Real from Kaggle) ===
        if 'subscriber_count' in df.columns:
            features['subscriber_growth_pct'] = df['subscriber_count'].pct_change() * 100
            self.real_data_usage['subscriber_growth'] = True
        else:
            features['subscriber_growth_pct'] = np.random.uniform(0, 10, len(df))
            self.real_data_usage['subscriber_growth'] = False
        
        # === REAL DATA PRIORITY 1: Google Trends Search Volume ===
        # This is the STRONGEST real data signal for churn intent
        if 'search_volume' in df.columns and (df['search_volume'] > 0).any():
            features['cancel_search_volume'] = df['search_volume']
            self.real_data_usage['search_volume'] = 'REAL_GOOGLE_TRENDS'
        elif 'google_trends_volume' in df.columns and (df['google_trends_volume'] > 0).any():
            features['cancel_search_volume'] = df['google_trends_volume']
            self.real_data_usage['search_volume'] = 'REAL_GOOGLE_TRENDS'
        else:
            # SYNTHETIC FALLBACK: Base on price change + subscriber health
            if 'price_change_pct' in features.columns:
                synthetic_vol = np.maximum(
                    20 + (features['price_change_pct'] * 0.5),  # Higher price → more searches
                    10
                )
                features['cancel_search_volume'] = synthetic_vol
            else:
                features['cancel_search_volume'] = np.random.uniform(15, 50, len(df))
            self.real_data_usage['search_volume'] = 'SYNTHETIC'
        
        # === REAL DATA PRIORITY 2: News Sentiment ===
        # Real sentiment from NewsAPI or web scraping
        if 'news_sentiment' in df.columns and (df['news_sentiment'] != 0).any():
            features['news_sentiment_score'] = df['news_sentiment']
            self.real_data_usage['sentiment'] = 'REAL_NEWSAPI'
        elif 'sentiment_score' in df.columns and (df['sentiment_score'] != 0).any():
            features['news_sentiment_score'] = df['sentiment_score']
            self.real_data_usage['sentiment'] = 'REAL_NEWSAPI'
        elif 'news_api_sentiment' in df.columns and (df['news_api_sentiment'] != 0).any():
            features['news_sentiment_score'] = df['news_api_sentiment']
            self.real_data_usage['sentiment'] = 'REAL_NEWSAPI'
        else:
            # SYNTHETIC FALLBACK: Derive from price changes and search trends
            price_shock = features['price_change_pct'].abs() / 100
            search_concern = features['cancel_search_volume'] / 100
            features['news_sentiment_score'] = -1 * np.maximum(price_shock * 0.6 + search_concern * 0.4, 0)
            features['news_sentiment_score'] = np.clip(features['news_sentiment_score'], -1, 1)
            self.real_data_usage['sentiment'] = 'SYNTHETIC'
        
        # === REAL DATA PRIORITY 3: Competitor Pricing ===
        # Real pricing from Kaggle or streaming service datasets
        if 'competitor_avg_price' in df.columns and (df['competitor_avg_price'] > 0).any():
            features['price_vs_competitors'] = df['price'] - df['competitor_avg_price']
            features['price_ratio_competitors'] = df['price'] / (df['competitor_avg_price'] + 0.01)
            self.real_data_usage['competitor_price'] = 'REAL'
        else:
            # SYNTHETIC: Use industry-standard competitor benchmarks
            if 'current_price' in features.columns:
                comp_avg = features['current_price'].mean() * 1.05  # Competitors slightly higher
                features['price_vs_competitors'] = features['current_price'] - comp_avg
                features['price_ratio_competitors'] = features['current_price'] / (comp_avg + 0.01)
            else:
                features['price_vs_competitors'] = 0
                features['price_ratio_competitors'] = 1.0
            self.real_data_usage['competitor_price'] = 'SYNTHETIC'
        
        # === LAG FEATURES FOR TEMPORAL CONTEXT ===
        # Capture momentum in all signals
        for col in features.columns:
            features[f'{col}_lag_1'] = features[col].shift(1)
        
        # Store original feature names for weight calculation later
        self.base_feature_names = [c for c in features.columns if not c.endswith('_lag_1')]
        self.feature_names = features.columns.tolist()
        return features.fillna(0)
    
    def train(self, X, y):
        """
        Execute model training with real data weighting.
        
        Records that use predominantly real data get higher weight in training,
        encouraging the model to learn from real data signals more heavily.
        
        Parameters:
            X (pd.DataFrame): Normalized feature matrix.
            y (pd.Series): Churn rate labels.
        """
        # Calculate sample weights based on real data availability
        sample_weights = self._calculate_sample_weights(X)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, sample_weight=sample_weights, verbose=False)
    
    def predict(self, X):
        """Execute inference to forecast churn rate."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def _calculate_sample_weights(self, X):
        """
        Calculate per-sample weights based on real data content.
        
        Samples with more real data features get higher weights.
        This encourages the model to fit more closely to real data patterns.
        
        Returns:
            np.ndarray: Sample weights (default 1.0, up to 2.0 for fully-real samples)
        """
        n_samples = len(X)
        weights = np.ones(n_samples)
        
        # Identify which columns come from real data vs synthetic
        real_data_cols = []
        for col in X.columns:
            base_col = col.replace('_lag_1', '')
            if hasattr(self, 'real_data_usage') and self.real_data_usage:
                # Check if the base feature or the column itself is marked as real
                if base_col in self.real_data_usage:
                    status = self.real_data_usage[base_col]
                    # Consider anything that's not exactly 'SYNTHETIC' as real
                    if isinstance(status, bool) and status:  # True = real
                        real_data_cols.append(col)
                    elif isinstance(status, str) and status != 'SYNTHETIC':
                        real_data_cols.append(col)
        
        # Boost weight for samples with more real data
        if real_data_cols and len(real_data_cols) > 0:
            real_data_count = X[real_data_cols].notna().sum(axis=1)
            max_real = len(real_data_cols)
            # Weight ranges from 1.0 (all synthetic) to 2.0 (all real)
            weights = 1.0 + (real_data_count / max_real).values
        
        return weights
    
    def get_real_data_percentage(self):
        """
        Calculate the percentage of features that use real vs synthetic data.
        
        Returns:
            dict: Real data usage statistics
        """
        if not hasattr(self, 'real_data_usage') or not self.real_data_usage:
            return {'real_percentage': 0, 'features_analyzed': 0, 'real_features': []}
        
        real_features = [k for k, v in self.real_data_usage.items() 
                        if v != 'SYNTHETIC']
        total_features = len(self.real_data_usage)
        real_pct = (len(real_features) / total_features * 100) if total_features > 0 else 0
        
        return {
            'real_percentage': round(real_pct, 1),
            'features_analyzed': total_features,
            'real_features': real_features,
            'synthetic_features': [k for k, v in self.real_data_usage.items() 
                                  if v == 'SYNTHETIC'],
            'real_data_sources': {k: v for k, v in self.real_data_usage.items() 
                                 if v != 'SYNTHETIC'}
        }
    
    def predict_saturation(self, current_price, proposed_price_increase_pct, 
                          historical_elasticity=None, growth_rate=None,
                          news_sentiment=None, search_volume=None,
                          competitor_avg_price=None):
        """
        Assess market saturation risk for a specific price update.
        
        REAL DATA PRIORITY: Uses actual signals when available (Google Trends search volume,
        NewsAPI sentiment), fills gaps with derived synthetic data.
        
        Evaluates the forecast against internal risk thresholds:
        - CRITICAL: > 5% churn
        - HIGH: > 3% churn
        - MODERATE: > 1% churn
        
        Real-world signals used (in priority order):
        1. news_sentiment: Real sentiment from news articles (range: -1 to 1)
        2. search_volume: Real search volume from Google Trends
        3. competitor_avg_price: Real competitor pricing for context
        4. historical_elasticity: Real elasticity from Kaggle data
        
        Returns:
            dict: Probabilistic churn forecast, risk level, and real data breakdown.
        """
        # Track which signals are real vs synthetic
        signal_sources = {}
        
        # Construct the inference payload with real data priority
        sample = pd.DataFrame({
            'current_price': [current_price],
            'price_change_pct': [proposed_price_increase_pct],
            'historical_elasticity': [historical_elasticity or -0.8],
            'subscriber_growth_pct': [growth_rate or 2.0],
            'cancel_search_volume': [search_volume or 50],
            'news_sentiment_score': [news_sentiment or 0.0]
        })
        
        # Track signal sources (REAL vs synthetic)
        signal_sources['search_volume'] = 'REAL_GOOGLE_TRENDS' if search_volume else 'SYNTHETIC'
        signal_sources['news_sentiment'] = 'REAL_NEWSAPI' if news_sentiment else 'SYNTHETIC'
        signal_sources['elasticity'] = 'REAL_KAGGLE' if historical_elasticity else 'SYNTHETIC'
        signal_sources['competitor_price'] = 'REAL' if competitor_avg_price else 'SYNTHETIC'
        signal_sources['growth_rate'] = 'REAL_KAGGLE' if growth_rate else 'SYNTHETIC'
        
        # Add competitor pricing features if available (REAL data preferred)
        if competitor_avg_price is not None:
            sample['competitor_avg_price'] = [competitor_avg_price]
            sample['price_vs_competitors'] = [current_price - competitor_avg_price]
            sample['price_ratio_competitors'] = [current_price / (competitor_avg_price + 0.01)]
        
        # Apply lag feature replication for single-point inference
        for col in sample.columns:
            sample[f'{col}_lag_1'] = sample[col]
        
        try:
            predicted_churn = self.predict(sample)[0]
        except:
            # Implement heuristic fallback if model state is uninitialized
            # Rely more heavily on real signals if available
            base_churn = 2.0 + (proposed_price_increase_pct / 10) * 1.2
            
            # REAL signal: news sentiment (strongest impact)
            if news_sentiment is not None:
                base_churn += abs(news_sentiment) * 0.7 if news_sentiment < 0 else 0
            
            # REAL signal: search volume (strong impact)
            if search_volume is not None:
                base_churn += (search_volume / 100) * 0.15
            
            # REAL signal: competitor pricing
            if competitor_avg_price is not None and current_price > competitor_avg_price:
                price_gap_pct = ((current_price - competitor_avg_price) / competitor_avg_price) * 100
                base_churn += price_gap_pct * 0.02
            
            predicted_churn = base_churn
        
        # Calculate real vs synthetic breakdown
        real_signal_count = sum(1 for v in signal_sources.values() if v != 'SYNTHETIC')
        total_signals = len(signal_sources)
        real_pct = (real_signal_count / total_signals * 100) if total_signals > 0 else 0
        
        risk_level = "CRITICAL" if predicted_churn > 5 else \
                     "HIGH" if predicted_churn > 3 else \
                     "MODERATE" if predicted_churn > 1 else \
                     "LOW"
        
        return {
            'proposed_price_increase_pct': proposed_price_increase_pct,
            'predicted_churn_rate': round(float(predicted_churn), 2),
            'risk_level': risk_level,
            'saturation_likely': predicted_churn > 3,
            'real_data_signals': {k: v for k, v in signal_sources.items() if v != 'SYNTHETIC'},
            'synthetic_signals': [k for k, v in signal_sources.items() if v == 'SYNTHETIC'],
            'real_signal_percentage': round(real_pct, 1),
            'features_used': {
                'news_sentiment': news_sentiment is not None,
                'real_search_volume': search_volume is not None,
                'competitor_pricing': competitor_avg_price is not None,
                'elasticity': historical_elasticity is not None
            }
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
