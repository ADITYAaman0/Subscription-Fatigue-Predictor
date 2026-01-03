"""
Unit tests for Subscription Fatigue Predictor.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.economic.economic_models import ElasticityCalculator, BertrandCompetitionModel
from src.models.ml.ml_models import ChurnRiskPredictor
from src.models.statistical.statistical_models import ChangePointDetector, CausalAnalyzer


class TestElasticityCalculator:
    """Tests for elasticity calculations."""
    
    def test_arc_elasticity(self):
        """Test arc elasticity calculation."""
        calculator = ElasticityCalculator()
        
        # Test case: 10% price increase, 5% quantity decrease
        # Expected elasticity: -0.5 (inelastic)
        elasticity = calculator.calculate_arc_elasticity(
            price_old=100,
            price_new=110,
            qty_old=1000,
            qty_new=950
        )
        
        assert -0.6 < elasticity < -0.4  # Close to -0.5
    
    def test_point_elasticity(self):
        """Test rolling elasticity calculation."""
        calculator = ElasticityCalculator()
        
        dates = pd.date_range('2020-01-01', '2024-12-01', freq='MS')
        df = pd.DataFrame({
            'date': dates,
            'price': np.linspace(10, 16, len(dates)),
            'subscriber_count': np.linspace(200e6, 210e6, len(dates))
        })
        
        result = calculator.calculate_point_elasticity(df, window_months=3)
        
        assert len(result) > 0
        assert 'elasticity' in result.columns
        assert 'revenue_change_pct' in result.columns


class TestChurnRiskPredictor:
    """Tests for churn risk prediction."""
    
    def test_predict_saturation(self):
        """Test saturation risk prediction."""
        predictor = ChurnRiskPredictor()
        
        result = predictor.predict_saturation(
            current_price=15.99,
            proposed_price_increase_pct=20
        )
        
        assert 'predicted_churn_rate' in result
        assert 'risk_level' in result
        assert result['risk_level'] in ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
        assert result['predicted_churn_rate'] >= 0


class TestChangePointDetector:
    """Tests for change point detection."""
    
    def test_pelt_detection(self):
        """Test PELT algorithm."""
        # Create data with clear breakpoint
        dates = pd.date_range('2020-01-01', '2024-12-01', freq='MS')
        trend1 = np.linspace(200, 250, 36)
        trend2 = np.linspace(250, 200, 24)  # 36 + 24 = 60 months
        data = np.concatenate([trend1, trend2]) * 1e6
        
        series = pd.Series(data, index=dates)
        
        detector = ChangePointDetector(min_size=10)
        change_points = detector.pelt_detection(series, penalty=5)
        
        assert isinstance(change_points, list)


class TestCausalAnalyzer:
    """Tests for causal analysis."""
    
    def test_find_optimal_lag(self):
        """Test lag finding."""
        dates = pd.date_range('2020-01-01', '2024-12-01', freq='W')
        monthly_dates = dates[::4]
        
        pricing_df = pd.DataFrame({
            'date': monthly_dates,
            'price': np.linspace(10, 16, len(monthly_dates))
        })
        
        search_trends_df = pd.DataFrame({
            'date': dates,
            'search_volume': 50 + np.random.randint(-10, 20, len(dates))
        })
        
        analyzer = CausalAnalyzer(pricing_df, search_trends_df)
        result = analyzer.find_optimal_lag()
        
        assert 'optimal_lag_weeks' in result
        assert 'correlation' in result
        assert 0 <= result['optimal_lag_weeks'] <= 8


class TestBertrandCompetition:
    """Tests for Bertrand competition model."""
    
    def test_nash_equilibrium(self):
        """Test Nash equilibrium calculation."""
        model = BertrandCompetitionModel(n_competitors=3)
        
        marginal_costs = np.array([5.0, 4.5, 6.0])
        differentiation = np.array([
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.4],
            [0.2, 0.4, 1.0]
        ])
        elasticities = np.array([
            [-0.8, -0.2, -0.1],
            [-0.2, -0.9, -0.15],
            [-0.1, -0.15, -0.85]
        ])
        
        result = model.calculate_nash_equilibrium_price(
            marginal_costs, differentiation, elasticities
        )
        
        assert len(result) == 3
        assert all(result >= marginal_costs)  # Prices > costs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
