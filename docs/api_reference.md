# Subscription Fatigue Predictor - API Reference

## Economic Models

### ElasticityCalculator

```python
from src.models.economic.economic_models import ElasticityCalculator

calculator = ElasticityCalculator()
```

#### Methods

**calculate_arc_elasticity(price_old, price_new, qty_old, qty_new)**
- Calculates midpoint elasticity
- Returns: float (elasticity coefficient)

**calculate_point_elasticity(df, window_months=3)**
- Calculates rolling elasticity over periods
- Args:
  - df: DataFrame with 'date', 'price', 'subscriber_count'
  - window_months: Calculation window in months
- Returns: DataFrame with elasticity metrics

### BertrandCompetitionModel

```python
from src.models.economic.economic_models import BertrandCompetitionModel

model = BertrandCompetitionModel(n_competitors=4)
```

#### Methods

**calculate_nash_equilibrium_price(marginal_costs, differentiation_params, demand_elasticities)**
- Solves for Nash equilibrium prices
- Args:
  - marginal_costs: Array of costs per competitor
  - differentiation_params: Matrix of product differentiation
  - demand_elasticities: Matrix of demand elasticities
- Returns: Array of equilibrium prices

## ML Models

### ChurnRiskPredictor

```python
from src.models.ml.ml_models import ChurnRiskPredictor

predictor = ChurnRiskPredictor()
```

#### Methods

**train(X, y)**
- Trains XGBoost churn model
- Args:
  - X: Feature DataFrame
  - y: Target (churn rate)

**predict_saturation(current_price, proposed_price_increase_pct, historical_elasticity=None, growth_rate=None)**
- Predicts saturation risk for price increase
- Returns: Dict with keys:
  - `proposed_price_increase_pct`
  - `predicted_churn_rate`
  - `risk_level` ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')
  - `saturation_likely` (bool)

## Statistical Models

### ChangePointDetector

```python
from src.models.statistical.statistical_models import ChangePointDetector

detector = ChangePointDetector(min_size=30)
```

#### Methods

**pelt_detection(series, penalty=10)**
- PELT algorithm for change point detection
- Args:
  - series: pandas Series with DatetimeIndex
  - penalty: Higher values = fewer breakpoints
- Returns: List of change point dates

**binary_segmentation(series, n_bkps=3)**
- Binary segmentation algorithm
- Args:
  - series: pandas Series
  - n_bkps: Number of breakpoints
- Returns: List of change point dates

### CausalAnalyzer

```python
from src.models.statistical.statistical_models import CausalAnalyzer

analyzer = CausalAnalyzer(pricing_df, search_trends_df)
```

#### Methods

**find_optimal_lag()**
- Finds lag with maximum correlation
- Returns: Dict with keys:
  - `optimal_lag_weeks`
  - `correlation`
  - `all_correlations` (dict)

## Dashboard

Run with: `streamlit run src/visualization/dashboard.py`

### Features
- KPI metrics
- Pricing timeline chart
- Subscriber growth trends
- Search trend analysis
- Elasticity analysis
- Change point visualization
- Saturation simulator
- Raw data explorer

## Configuration

Edit `src/utils/config.py` to customize:

```python
# Services to analyze
SERVICES = ['Netflix', 'Spotify', 'Disney Plus']

# Analysis parameters
ELASTICITY_WINDOW_MONTHS = 6
CHANGE_POINT_PENALTY = 10

# Model parameters
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1
}
```

## Data Format

### Pricing DataFrame
```
company_id | effective_date | price | currency
```

### Metrics DataFrame
```
company_id | date | subscriber_count | arpu | churn_rate
```

### Search Trends DataFrame
```
company_id | date | search_term | search_volume | region
```

---

Last updated: December 2025
