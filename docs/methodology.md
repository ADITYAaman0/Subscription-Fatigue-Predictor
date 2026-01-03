# Subscription Fatigue Predictor - Methodology

## Economic Theory Foundation

### Price Elasticity of Demand

Price elasticity measures the responsiveness of quantity demanded to price changes:

$$E_p = \frac{\% \text{ Change in Quantity}}{\% \text{ Change in Price}} = \frac{\Delta Q/Q}{\Delta P/P}$$

**Interpretation:**
- $E_p < -1$: **ELASTIC** - Price increase reduces revenue (demand is price-sensitive)
- $-1 < E_p < 0$: **INELASTIC** - Price increase increases revenue (demand is price-insensitive)
- $E_p = -1$: **UNIT ELASTIC** - Price increase maintains revenue

### Market Saturation

A market reaches saturation when:
1. Price elasticity approaches or exceeds -1
2. Subscriber growth rate slows significantly
3. Churn rate increases with price announcements
4. Consumer search volume for "cancel [service]" spikes

### Bertrand Competition Model

In oligopolistic markets with differentiated products:

$$p_i^* = MC_i + \frac{1}{|\varepsilon_{ii} + \sum_{j \neq i} \theta_{ij} \varepsilon_{ji}|}$$

Where:
- $p_i^*$: Optimal price for service $i$
- $MC_i$: Marginal cost
- $\varepsilon_{ii}$: Own-price elasticity
- $\varepsilon_{ji}$: Cross-price elasticity (demand for service $j$ w.r.t. price of service $i$)
- $\theta_{ij}$: Strategic interaction parameter

## Statistical Methods

### Change Point Detection (PELT)

PELT (Pruned Exact Linear Time) algorithm identifies structural breaks in time series:

$$\min_{\tau} \left[ \sum_{t} L(y_t) + \beta |\tau| \right]$$

Where:
- $L(y_t)$: Loss function (e.g., sum of squared errors)
- $|\tau|$: Number of breakpoints
- $\beta$: Penalty coefficient (higher = fewer breakpoints)

### Causal Analysis

Test if price changes **Granger-cause** search volume increases:

$$y_t = \alpha + \sum_{i=1}^{p} \beta_i y_{t-i} + \sum_{i=1}^{p} \gamma_i x_{t-i} + \epsilon_t$$

Where:
- $y_t$: Search volume (cancellation intent)
- $x_t$: Price changes
- Reject null hypothesis if adding $x$ significantly improves prediction

## Machine Learning Approach

### XGBoost Churn Predictor

Features engineered:
- Historical elasticity
- Price change percentage
- Subscriber growth rate
- Search volume (cancellation signals)
- Competitive pricing
- Macroeconomic indicators

Model trained on historical price increases and their churn outcomes.

### Heterogeneous Treatment Effects

Using Causal Forests to identify which customer segments are most price-sensitive:
- Low-income users: Typically elastic (price-sensitive)
- High engagement users: Typically inelastic (loyal)
- New subscribers: More elastic (haven't built loyalty)

## Data Sources

### Primary Data
- Kaggle: Historical pricing and subscriber metrics
- Yahoo Finance: Stock prices and volumes
- Company investor relations: SEC filings and earnings calls

### Alternative Data
- Google Trends: Search volume for cancellation keywords
- App Store reviews: User sentiment
- Social media: Discussion volumes and sentiment
- News APIs: Media coverage

### Macroeconomic Data
- Federal Reserve: Interest rates, unemployment
- Bureau of Labor Statistics: CPI inflation
- FRED API: 400+ economic indicators

## Feature Engineering

### Temporal Features
- Quarters since service launch
- Seasonality indices (quarterly patterns)
- Year-over-year growth rates

### Competitive Features
- Herfindahl-Hirschman Index (market concentration)
- Relative pricing (vs. competitors)
- Feature differentiation index

### User Behavior Features
- Engagement score (hours watched/month)
- Content exclusivity ratio
- Cross-service penetration rate

## Model Validation

### Backtesting Strategy
1. Train on 2020-2023 data
2. Test on 2024-2025 holdout
3. Evaluate:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Directional accuracy

### Cross-Validation
- Time series cross-validation (walk-forward)
- Stratified by market cycle phase

## Saturation Risk Score

$$\text{Risk} = w_1 E_p + w_2 \Delta \text{Search} + w_3 \Delta \text{Churn} + w_4 \text{Competition}$$

Where:
- $E_p$: Price elasticity
- $\Delta \text{Search}$: Change in cancellation search volume
- $\Delta \text{Churn}$: Change in churn rate
- $\text{Competition}$: Competitive intensity

Weight assignment reflects relative importance of each factor.

## Limitations & Considerations

1. **Historical Bias**: Past elasticity may not predict future behavior
2. **External Shocks**: COVID-19, competitor launches affect elasticity
3. **Data Quality**: Search volume trends depend on keyword selection
4. **Market Dynamics**: Elasticity varies with market maturity phase
5. **Unobserved Factors**: Content quality, UI changes not always captured

---

Last updated: December 2025
