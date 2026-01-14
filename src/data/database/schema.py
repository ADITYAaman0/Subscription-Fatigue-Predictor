"""
Database schema and initialization SQL.
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    company_id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    sector TEXT,
    country TEXT,
    stock_symbol TEXT
);

CREATE TABLE IF NOT EXISTS pricing_history (
    price_id INTEGER PRIMARY KEY,
    company_id INTEGER,
    tier_id INTEGER,
    price REAL,
    currency TEXT,
    country TEXT,
    effective_date DATE,
    previous_price REAL,
    change_percentage REAL,
    FOREIGN KEY (company_id) REFERENCES companies(company_id)
);

CREATE TABLE IF NOT EXISTS subscriber_metrics (
    metric_id INTEGER PRIMARY KEY,
    company_id INTEGER,
    date DATE,
    subscriber_count INTEGER,
    premium_subscribers INTEGER,
    market_share REAL,
    arpu REAL,
    churn_rate REAL,
    FOREIGN KEY (company_id) REFERENCES companies(company_id),
    UNIQUE(company_id, date)
);

CREATE TABLE IF NOT EXISTS search_trends (
    trend_id INTEGER PRIMARY KEY,
    company_id INTEGER,
    search_term TEXT,
    date DATE,
    search_volume INTEGER,
    region TEXT,
    FOREIGN KEY (company_id) REFERENCES companies(company_id)
);

CREATE TABLE IF NOT EXISTS elasticity_analysis (
    elasticity_id INTEGER PRIMARY KEY,
    company_id INTEGER,
    period_start DATE,
    period_end DATE,
    price_change_percentage REAL,
    quantity_change_percentage REAL,
    elasticity_coefficient REAL,
    revenue_impact_percentage REAL,
    is_elastic BOOLEAN,
    FOREIGN KEY (company_id) REFERENCES companies(company_id)
);

CREATE TABLE IF NOT EXISTS change_points (
    changepoint_id INTEGER PRIMARY KEY,
    company_id INTEGER,
    metric_type TEXT,
    detected_date DATE,
    confidence REAL,
    change_magnitude REAL,
    associated_price_change REAL,
    is_saturation BOOLEAN,
    FOREIGN KEY (company_id) REFERENCES companies(company_id)
);
"""
