"""
Streamlit dashboard for Subscription Fatigue Predictor.
Main visualization and interaction interface with advanced models.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import models
try:
    from src.models.statistical.statistical_models import ChangePointDetector, CausalAnalyzer
    from src.models.economic.economic_models import ElasticityCalculator, BertrandCompetitionModel, ConsumerSurplusAnalyzer
    from src.models.ml.ml_models import ChurnRiskPredictor, HeterogeneousEffectAnalyzer
    # Import new advanced models
    from src.models.advanced_models import (
        CompetitiveResonanceModel,
        WeeklyChurnDetector,
        PsychographicSegmenter,
        BundleOptimizer
    )
    # Import new analytics dashboard components
    from src.visualization.analytics_dashboard import (
        render_content_value_tab,
        ANALYTICS_AVAILABLE
    )
except ImportError as e:
    st.warning(f"Some modules not available: {e}")
    # Create dummy classes for missing imports
    class DummyModel:
        def __init__(self, *args, **kwargs): pass
        def __getattr__(self, name): return lambda *args, **kwargs: None
    
    ChangePointDetector = DummyModel
    ElasticityCalculator = DummyModel
    CompetitiveResonanceModel = DummyModel
    WeeklyChurnDetector = DummyModel
    PsychographicSegmenter = DummyModel
    BundleOptimizer = DummyModel
    CausalAnalyzer = DummyModel
    ChurnRiskPredictor = DummyModel
    HeterogeneousEffectAnalyzer = DummyModel
    BertrandCompetitionModel = DummyModel
    ConsumerSurplusAnalyzer = DummyModel
    ANALYTICS_AVAILABLE = False
    
    def render_content_value_tab(): st.info("Content Value module not available")
    def render_attribution_tab(): st.info("Attribution module not available")
    def render_ab_testing_tab(): st.info("A/B Testing module not available")
    def render_network_effects_tab(): st.info("Network Effects module not available")
    def render_regulatory_tab(): st.info("Regulatory Simulation module not available")

# Page configuration
st.set_page_config(
    page_title="Subscription Fatigue Predictor v2.0",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for Premium UI/UX
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">

<style>
    /* Global Variables & Base Styles */
    :root {
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.1);
        --accent-red: #FF2E2E;
        --accent-blue: #00D1FF;
        --accent-green: #39FF14;
        --bg-deep: #0B0E11;
        --text-primary: #FFFFFF;
        --text-secondary: rgba(255, 255, 255, 0.6);
        --font-main: 'Outfit', sans-serif;
        --font-sub: 'Inter', sans-serif;
    }

    .stApp {
        background-color: var(--bg-deep);
        font-family: var(--font-main);
    }

    /* Premium Typography */
    h1, h2, h3, h4, h5, h6, .main-header {
        font-family: var(--font-main);
        letter-spacing: -0.02em;
        font-weight: 800;
        color: var(--text-primary);
    }

    p, span, div, .stMarkdown {
        font-family: var(--font-sub);
    }

    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #FFFFFF 0%, #A0A0A0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-top: 2rem;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        font-size: 1.2rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    /* Glassmorphism Containers */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 24px;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    /* Specific Component Overrides */
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent-blue);
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 8px;
        padding: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 100px;
        color: var(--text-secondary);
        padding: 8px 24px;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1);
        color: white;
    }

    .stTabs [aria-selected="true"] {
        background: white !important;
        color: black !important;
        border: none;
        font-weight: 600;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0F1216;
        border-right: 1px solid var(--glass-border);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary);
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# UI Helpers for Glassmorphic Cards
def kpi_card(label, value, color, icon):
    return f"""
    <div class="glass-card" style="text-align: center; border-bottom: 2px solid {color}; padding: 20px;">
        <p style="margin:0; font-size: 0.75rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.1em; font-family: 'Inter'; font-weight: 600;">{label}</p>
        <h3 style="margin: 8px 0 0 0; font-size: 2rem; color: white; display: flex; align-items: center; justify-content: center; gap: 10px;">
            <span style="font-size: 1.4rem;">{icon}</span> {value}
        </h3>
    </div>
    """

def insight_card(title, service, val, color, icon):
    return f"""
    <div class="glass-card" style="border-left: 4px solid {color}; padding: 20px;">
        <p style="margin:0; font-size: 0.75rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.08em; font-family: 'Inter';">{title}</p>
        <h4 style="margin: 8px 0 0 0; font-size: 1.4rem; color: white; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px;">{icon} {service}</h4>
        <p style="margin:12px 0 0 0; color: {color}; font-weight: 700; font-size: 1.1rem;">{val:+.2f}% <span style='font-size: 0.8rem; font-weight: 400; opacity: 0.7; color: white;'>Net Change</span></p>
    </div>
    """

def diversion_card(label, value, color, icon):
    return f"""
    <div class="glass-card" style="border-top: 4px solid {color}; padding: 18px; text-align: center;">
        <p style="margin:0; font-size: 0.7rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.08em; font-family: 'Inter';">{label}</p>
        <h4 style="margin: 8px 0 0 0; font-size: 1.5rem; color: white;">{icon} {value}</h4>
    </div>
    """

# Company Icons Mapping
COMPANY_ICONS = {
    'Netflix': '🎬',
    'Disney Plus': '🏰',
    'HBO Max': '📺',
    'Amazon Prime': '📦'
}

# Company Color Mapping
COMPANY_COLORS = {
    'Netflix': '#E50914',      # Red
    'Disney Plus': '#113CCF',  # Royal Blue
    'HBO Max': '#5822b4',      # Purple
    'Amazon Prime': '#00A8E1'  # Light Blue
}

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

@st.cache_data(ttl=1)
def load_and_prepare_data():
    """Load and prepare all data for the dashboard. Robust error handling with fallbacks."""
    # Strictly use the main database - resolve relative to CWD (x:\spc)
    db_path = Path('data/subscription_fatigue.db').resolve()
    
    if not db_path.exists():
        st.error(f"Database not found at: {db_path}")
        return generate_sample_data()

    conn = sqlite3.connect(str(db_path))
    
    # Load companies (essential)
    try:
        companies = pd.read_sql("SELECT * FROM companies", conn)
    except:
        companies = pd.DataFrame()
        
    # Load pricing
    try:
        pricing = pd.read_sql("SELECT * FROM pricing_history", conn)
        if not pricing.empty:
            # Safely rename 'date' to 'effective_date' only if it won't cause duplicates
            if 'date' in pricing.columns:
                if 'effective_date' not in pricing.columns:
                    pricing = pricing.rename(columns={'date': 'effective_date'})
                else:
                    # Both exist, prefer 'effective_date' and drop 'date'
                    pricing = pricing.drop(columns=['date'])
    except:
        pricing = pd.DataFrame()
        
    # Load metrics  
    try:
        metrics = pd.read_sql("SELECT * FROM subscriber_metrics", conn)
    except:
        metrics = pd.DataFrame()
        
    # Load trends
    try:
        trends = pd.read_sql("SELECT * FROM search_trends", conn)
    except:
        trends = pd.DataFrame()
    
    # Load news and provenance
    try:
        news_data = pd.read_sql("SELECT * FROM news_articles", conn)
    except:
        news_data = pd.DataFrame()
            
    try:
        provenance = pd.read_sql("SELECT * FROM data_provenance", conn)
    except:
        provenance = pd.DataFrame()
    
    # Load global streaming
    try:
        global_streaming = pd.read_sql("SELECT * FROM real_global_streaming", conn)
        if global_streaming.empty:
             global_streaming = pd.read_sql("SELECT * FROM global_streaming", conn) # Fallback
    except:
        global_streaming = pd.DataFrame()
    
    # Load ecommerce
    try:
        ecommerce = pd.read_sql("SELECT * FROM ecommerce_data", conn)
    except:
        ecommerce = pd.DataFrame()
    
    # Initialize metrics_summary (needed by notebook integration)
    metrics_summary = pd.DataFrame()
    
    # Load kaggle data - PRIORITIZE real_world_churn_data
    try:
        kaggle_data = pd.read_sql("SELECT * FROM real_world_churn_data", conn)
    except Exception as e:
        st.error(f"KAGGLE LOAD ERROR: {e}")
        kaggle_data = pd.DataFrame()

    if kaggle_data.empty:
        try:
            kaggle_data = pd.read_sql("SELECT * FROM kaggle_telco_churn_WA_Fn_UseC__Telco_Customer_Churn", conn)
        except:
            pass
    conn.close()
    
    # DEDUPLICATE ALL DATAFRAMES to prevent assemble errors
    pricing = pricing.loc[:, ~pricing.columns.duplicated()].copy()
    metrics = metrics.loc[:, ~metrics.columns.duplicated()].copy()
    trends = trends.loc[:, ~trends.columns.duplicated()].copy()
    companies = companies.loc[:, ~companies.columns.duplicated()].copy()
    kaggle_data = kaggle_data.loc[:, ~kaggle_data.columns.duplicated()].copy()
    
    # Ensure all dataframes have dates in proper format
    for df in [pricing, metrics, trends]:
        if not df.empty:
            for col in df.columns:
                if 'date' in col.lower():
                    # df[col] is now guaranteed to be a Series
                    df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # If essential data is missing, use sample
    if pricing.empty or metrics.empty:
        st.warning("Essential data missing from database, falling back to sample data.")
        return generate_sample_data()
    
    return pricing, metrics, trends, companies, kaggle_data, news_data, provenance, global_streaming, ecommerce, metrics_summary


def render_data_health_sidebar(provenance_df: pd.DataFrame):
    """
    Render data health panel in sidebar showing real vs synthetic data usage.
    
    Args:
        provenance_df: DataFrame from data_provenance table
    """
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Data Health")
    
    if provenance_df.empty:
        st.sidebar.warning("No provenance data available")
        st.sidebar.caption("Run the data pipeline to populate")
        return
    
    # Calculate real vs synthetic percentages
    # DEDUPLICATION: Keep only the latest entry for each table to avoid skewing by historical runs
    if 'table_name' in provenance_df.columns and 'ingestion_timestamp' in provenance_df.columns:
        latest_provenance = provenance_df.sort_values('ingestion_timestamp', ascending=False).drop_duplicates('table_name')
    else:
        latest_provenance = provenance_df
    
    source_counts = latest_provenance['source_type'].value_counts()
    total = source_counts.sum()
    
    # Treat everything except 'synthetic' as Real
    synthetic_sources = source_counts.get('synthetic', 0)
    real_sources = total - synthetic_sources
    
    real_pct = (real_sources / total * 100) if total > 0 else 0
    synthetic_pct = (synthetic_sources / total * 100) if total > 0 else 0
    
    # Display metrics
    col1, col2 = st.sidebar.columns(2)
    with col1:
        color = "#39FF14" if real_pct >= 80 else ("#FFA500" if real_pct >= 50 else "#FF2E2E")
        st.sidebar.markdown(f"""
        <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 10px; border-left: 3px solid {color};">
            <p style="margin:0; font-size: 0.75rem; opacity: 0.7;">REAL DATA</p>
            <p style="margin:0; font-size: 1.5rem; font-weight: bold; color: {color};">{real_pct:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.sidebar.markdown(f"""
        <div style="text-align: center; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 10px; border-left: 3px solid #6366f1;">
            <p style="margin:0; font-size: 0.75rem; opacity: 0.7;">SYNTHETIC</p>
            <p style="margin:0; font-size: 1.5rem; font-weight: bold; color: #6366f1;">{synthetic_pct:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Last refresh time
    if 'ingestion_timestamp' in provenance_df.columns:
        try:
            latest = pd.to_datetime(provenance_df['ingestion_timestamp']).max()
            if pd.notna(latest):
                time_ago = datetime.now() - latest
                if time_ago.days > 0:
                    freshness = f"{time_ago.days}d ago"
                    fresh_color = "#FF2E2E" if time_ago.days > 7 else "#FFAA00"
                else:
                    hours = time_ago.seconds // 3600
                    freshness = f"{hours}h ago" if hours > 0 else "Just now"
                    fresh_color = "#39FF14"
                
                st.sidebar.markdown(f"""
                <div style="margin-top: 10px; padding: 8px; background: rgba(255,255,255,0.03); border-radius: 8px;">
                    <p style="margin:0; font-size: 0.7rem; opacity: 0.6;">LAST REFRESH</p>
                    <p style="margin:0; font-size: 1rem; color: {fresh_color};">{freshness}</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            pass
    
    # Source breakdown expander
    with st.sidebar.expander("📋 Source Details"):
        for source_type in ['kaggle', 'web_scrape', 'synthetic']:
            source_data = provenance_df[provenance_df['source_type'] == source_type]
            if not source_data.empty:
                icon = {"kaggle": "📦", "web_scrape": "🌐", "synthetic": "🔧"}.get(source_type, "📄")
                st.markdown(f"**{icon} {source_type.title()}** ({len(source_data)} tables)")
                for _, row in source_data.iterrows():
                    table = row.get('table_name', 'Unknown')
                    records = row.get('record_count', 0)
                    st.caption(f"  • {table}: {records:,} records")


def generate_sample_data():
    """Generate comprehensive sample data for demonstration.
    Returns 10 items to match load_and_prepare_data() expectations."""
    # Companies
    companies = pd.DataFrame({
        'company_id': [1, 3, 4, 5],
        'name': ['Netflix', 'Disney Plus', 'HBO Max', 'Amazon Prime'],
        'sector': ['Streaming', 'Streaming', 'Streaming', 'Streaming'],
        'country': ['US', 'US', 'US', 'US'],
        'stock_symbol': ['NFLX', 'DIS', 'WBD', 'AMZN']
    })
    
    # Generate dates up to current month
    now = datetime.now()
    end_date_current = datetime(now.year, now.month, 1)
    dates = pd.date_range('2020-01-01', end_date_current, freq='MS')
    weekly_dates = pd.date_range('2020-01-01', end_date_current, freq='W')
    
    # Pricing data
    pricing_data = []
    base_prices = {
        1: 10.99,  # Netflix
        3: 7.99,   # Disney Plus
        4: 14.99,  # HBO Max
        5: 12.99   # Amazon Prime
    }
    
    for company_id, base_price in base_prices.items():
        for i, date in enumerate(dates):
            # Simulate price increases
            price = base_price
            if i >= 12:  # After 1 year
                price += 1.50
            if i >= 24:  # After 2 years
                price += 1.50
            if i >= 36:  # After 3 years
                price += 1.50
            
            # Add some randomness
            price += np.random.uniform(-0.5, 0.5)
            
            pricing_data.append({
                'company_id': company_id,
                'effective_date': date,
                'price': round(price, 2),
                'currency': 'USD',
                'country': 'US',
                'previous_price': round(max(price - np.random.uniform(0, 1), 5), 2),
                'change_percentage': np.random.uniform(0, 10)
            })
    
    pricing = pd.DataFrame(pricing_data)
    
    # Metrics data
    metrics_data = []
    base_subs = {
        1: 220_000_000,  # Netflix
        3: 110_000_000,  # Disney Plus
        4: 80_000_000,   # HBO Max
        5: 200_000_000   # Amazon Prime
    }
    
    for company_id, base_sub in base_subs.items():
        subs = base_sub
        for i, date in enumerate(dates):
            # Simulate growth with seasonality
            growth = np.random.uniform(-0.01, 0.03)
            if i % 12 in [10, 11, 0, 1]:  # Holiday season
                growth += 0.02
            
            subs = int(subs * (1 + growth))
            
            metrics_data.append({
                'company_id': company_id,
                'date': date,
                'subscriber_count': subs,
                'premium_subscribers': int(subs * 0.7),
                'market_share': np.random.uniform(15, 40),
                'arpu': np.random.uniform(10, 16),
                'churn_rate': np.random.uniform(1.5, 4.5)
            })
    
    metrics = pd.DataFrame(metrics_data)
    
    # Trends data
    trends_data = []
    search_terms = {
        1: ['Cancel Netflix', 'Netflix price', 'Netflix expensive'],
        3: ['Cancel Disney Plus', 'Disney Plus price'],
        4: ['Cancel HBO Max', 'HBO Max price'],
        5: ['Cancel Amazon Prime', 'Prime Video price']
    }
    
    for company_id, terms in search_terms.items():
        for date in weekly_dates:
            for term in terms:
                # Base volume with trends
                base = np.random.randint(20, 40)
                
                # Add spikes around known price increase dates
                if any(price_date.month == date.month and price_date.year == date.year 
                      for price_date in dates[::12]):
                    base += np.random.randint(20, 40)
                
                # Add some randomness
                base += np.random.randint(-10, 10)
                base = max(0, base)
                
                trends_data.append({
                    'company_id': company_id,
                    'date': date,
                    'search_term': term,
                    'search_volume': base,
                    'region': 'US'
                })
    
    trends = pd.DataFrame(trends_data)
    
    # Return 10 items to match load_and_prepare_data expectations
    return (pricing, metrics, trends, companies, pd.DataFrame(), pd.DataFrame(), 
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

def prepare_data_for_models(pricing, metrics, trends, companies, news_data=None, global_streaming=None, ecommerce=None):
    """Prepare data in formats needed by different models.
    
    Enhanced to include:
    - news_data: For sentiment analysis and co-occurrence
    - global_streaming: For competitor pricing and subscriber data
    - ecommerce: For price sensitivity
    """
    # Merge company names into dataframes
    pricing_named = pricing.merge(companies[['company_id', 'name']], on='company_id')
    metrics_named = metrics.merge(companies[['company_id', 'name']], on='company_id')
    trends_named = trends.merge(companies[['company_id', 'name']], on='company_id')
    
    result = {
        'pricing_named': pricing_named,
        'metrics_named': metrics_named,
        'trends_named': trends_named,
        'companies': companies
    }
    
    # Add optional real data sources
    if news_data is not None: result['news_data'] = news_data
    if global_streaming is not None: result['global_streaming'] = global_streaming
    if ecommerce is not None: result['ecommerce'] = ecommerce
    
    return result

# =============================================================================
# SIDEBAR FILTERS
# =============================================================================

def create_sidebar(data):
    """Create sidebar with all filters and controls."""
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Service selection
    company_names = data['companies']['name'].tolist()
    selected_services = st.sidebar.multiselect(
        "Select Services",
        company_names,
        default=company_names[:3],
        help="Choose which subscription services to analyze"
    )
    
    # Date range - handle potential NaT values
    if 'effective_date' in data['pricing_named'].columns:
        data['pricing_named']['effective_date'] = pd.to_datetime(data['pricing_named']['effective_date'])
        min_date_val = data['pricing_named']['effective_date'].min()
        max_date_val = data['pricing_named']['effective_date'].max()
    else:
        # Fallback if column still missing
        min_date_val = pd.NaT
        max_date_val = pd.NaT
    
    # Handle NaT values with fallback dates
    if pd.isna(min_date_val):
        min_date = datetime(2020, 1, 1).date()
    else:
        min_date = min_date_val.date()
    
    if pd.isna(max_date_val):
        max_date = datetime.now().date()
    else:
        max_date = max_date_val.date()
    
    date_range = st.sidebar.date_input(
        "Analysis Period",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Analysis parameters
    st.sidebar.header("📊 Analysis Settings")
    
    analysis_window = st.sidebar.select_slider(
        "Analysis Window (months)",
        options=[3, 6, 12, 24, 36],
        value=12
    )
    
    sensitivity = st.sidebar.select_slider(
        "Model Sensitivity",
        options=["Conservative", "Moderate", "Aggressive"],
        value="Moderate"
    )
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        change_point_penalty = st.slider(
            "Change Point Penalty",
            min_value=1,
            max_value=20,
            value=10,
            help="Higher values detect fewer change points"
        )
        
        elasticity_method = st.selectbox(
            "Elasticity Method",
            ["Arc Elasticity", "Point Elasticity", "Rolling Elasticity"]
        )
        
        forecast_horizon = st.slider(
            "Forecast Horizon (weeks)",
            min_value=4,
            max_value=52,
            value=12
        )
    
    return {
        'services': selected_services,
        'date_range': date_range,
        'analysis_window': analysis_window,
        'sensitivity': sensitivity,
        'change_point_penalty': change_point_penalty,
        'elasticity_method': elasticity_method,
        'forecast_horizon': forecast_horizon
    }

# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================

def create_header():
    """Create dashboard header."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<p class="main-header">💰 Subscription Fatigue Predictor</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-powered market intelligence for subscription businesses</p>', unsafe_allow_html=True)
    
    st.markdown("---")

def create_kpi_metrics(data, filters):
    """Display key performance indicators."""
    st.subheader("📊 Market Overview")
    
    # Calculate metrics
    latest_date = data['metrics_named']['date'].max()
    latest_metrics = data['metrics_named'][data['metrics_named']['date'] == latest_date]
    
    # Filter for selected services
    selected_metrics = latest_metrics[latest_metrics['name'].isin(filters['services'])]
    
    m1, m2, m3, m4, m5 = st.columns(5)

    # Safe column access using reindexing
    required_cols = ['subscriber_count', 'arpu', 'churn_rate', 'market_share']
    metrics_safe = selected_metrics.reindex(columns=required_cols, fill_value=0)
    
    total_subs = metrics_safe['subscriber_count'].sum()
    avg_arpu = metrics_safe['arpu'].mean()
    total_rev = total_subs * avg_arpu / 1e6
    avg_churn = metrics_safe['churn_rate'].mean() * 100
    total_share = metrics_safe['market_share'].sum()

    with m1:
        st.markdown(kpi_card("Total Reach", f"{total_subs/1e6:.1f}M", "#00D1FF", "💎"), unsafe_allow_html=True)
    with m2:
        st.markdown(kpi_card("Monthly Rev", f"${total_rev:.1f}M", "#39FF14", "📈"), unsafe_allow_html=True)
    with m3:
        st.markdown(kpi_card("Avg Churn", f"{avg_churn:.1f}%", "#FF2E2E", "📉"), unsafe_allow_html=True)
    with m4:
        st.markdown(kpi_card("Avg ARPU", f"${avg_arpu:.2f}", "#FFFFFF", "💳"), unsafe_allow_html=True)
    with m5:
        st.markdown(kpi_card("Mkt Share", f"{total_share:.1f}%", "#A855F7", "🔗"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_pricing_comparison(data, filters):
    """Plot pricing comparison across services."""
    st.subheader("💵 Pricing Timeline Comparison")
    
    # Filter and prepare data - handle NaT values
    pricing_valid = data['pricing_named'].dropna(subset=['effective_date'])
    start_date = pd.Timestamp(filters['date_range'][0])
    end_date = pd.Timestamp(filters['date_range'][1])
    
    filtered_pricing = pricing_valid[
        pricing_valid['name'].isin(filters['services']) &
        (pricing_valid['effective_date'] >= start_date) &
        (pricing_valid['effective_date'] <= end_date)
    ]
    
    if filtered_pricing.empty:
        st.warning("No pricing data available for selected filters")
        return
    
    # Ensure color mapping exists for all platforms
    for s in filters['services']:
        if s not in COMPANY_COLORS:
            COMPANY_COLORS[s] = "#888888"
        if s not in COMPANY_ICONS:
            COMPANY_ICONS[s] = "📱"
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for idx, service in enumerate(filters['services']):
        service_data = filtered_pricing[filtered_pricing['name'] == service].sort_values('effective_date')
        
        fig.add_trace(go.Scatter(
            x=service_data['effective_date'],
            y=service_data['price'],
            name=service,
            mode='lines+markers',
            line=dict(width=3, color=COMPANY_COLORS.get(service, colors[idx % len(colors)])),
            marker=dict(size=8),
            hovertemplate=f"{service}<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Subscription Price Evolution",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_price = filtered_pricing.groupby('name')['price'].last().mean()
        st.metric("Average Current Price", f"${avg_price:.2f}")
    
    with col2:
        max_price = filtered_pricing.groupby('name')['price'].last().max()
        st.metric("Highest Current Price", f"${max_price:.2f}")
    
    with col3:
        price_changes = filtered_pricing.groupby('name').apply(
            lambda x: ((x['price'].iloc[-1] - x['price'].iloc[0]) / x['price'].iloc[0] * 100) 
            if len(x) > 1 else 0
        ).mean()
        st.metric("Avg Price Change", f"{price_changes:.1f}%")
    
    with col4:
        # Safe column access for change_percentage
        if 'change_percentage' in filtered_pricing.columns:
            increase_count = len(filtered_pricing[filtered_pricing['change_percentage'] > 0])
        else:
            increase_count = 0
        st.metric("Price Increases", f"{increase_count}")

def plot_subscriber_dynamics(data, filters):
    """Plot subscriber growth and churn dynamics."""
    st.subheader("👥 Subscriber Dynamics")
    
    # Filter data - handle NaT values
    metrics_valid = data['metrics_named'].dropna(subset=['date'])
    start_date = pd.Timestamp(filters['date_range'][0])
    end_date = pd.Timestamp(filters['date_range'][1])
    
    filtered_metrics = metrics_valid[
        metrics_valid['name'].isin(filters['services']) &
        (metrics_valid['date'] >= start_date) &
        (metrics_valid['date'] <= end_date)
    ]
    
    if filtered_metrics.empty:
        st.warning("No metrics data available")
        return
    
    colors = px.colors.qualitative.Set3
    
    # Define metrics to plot
    # tuple: (column_name, title, y_label, divisor, is_growth)
    metrics_to_plot = [
        ('subscriber_count', 'Subscriber Growth', 'Subscribers (Millions)', 1e6),
        ('churn_rate', 'Churn Rate', 'Churn Rate (%)', 1.0),
        ('market_share', 'Market Share', 'Market Share (%)', 1.0),
        ('arpu', 'ARPU Trends', 'ARPU ($)', 1.0)
    ]
    
    for col, title, ylabel, divisor in metrics_to_plot:
        fig = go.Figure()
        
        for idx, service in enumerate(filters['services']):
            service_data = filtered_metrics[filtered_metrics['name'] == service].sort_values('date')
            color = COMPANY_COLORS.get(service, colors[idx % len(colors)])
            
            fig.add_trace(go.Scatter(
                x=service_data['date'],
                y=service_data[col] / divisor * (100 if col == 'churn_rate' else 1),
                name=service,
                mode='lines',
                line=dict(width=3 if col == 'subscriber_count' else 2, color=color),
                showlegend=True
            ))
            
        fig.update_layout(
            title=title,
            yaxis_title=ylabel,
            xaxis_title="Date",
            height=500, # Individual graph height
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---") # Add separation line

def plot_search_trends_analysis(data, filters):
    """Plot search trends analysis."""
    st.subheader("🔍 Cancellation Intent Signals")
    
    # Filter trends data - handle NaT values
    trends_valid = data['trends_named'].dropna(subset=['date'])
    start_date = pd.Timestamp(filters['date_range'][0])
    end_date = pd.Timestamp(filters['date_range'][1])
    
    filtered_trends = trends_valid[
        trends_valid['name'].isin(filters['services']) &
        (trends_valid['date'] >= start_date) &
        (trends_valid['date'] <= end_date)
    ]
    
    if filtered_trends.empty:
        st.warning("No search trends data available")
        return
    
    # Calculate weekly aggregates
    filtered_trends['week'] = filtered_trends['date'].dt.to_period('W').dt.start_time
    weekly_trends = filtered_trends.groupby(['week', 'name'])['search_volume'].mean().reset_index()
    
    # Create heatmap data - we need to handle each service separately to allow custom colors
    services = sorted(weekly_trends['name'].unique())
    
    # Create subplots - one row per service
    fig = make_subplots(
        rows=len(services), 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.15, # Increased spacing as requested
        subplot_titles=services
    )
    
    for idx, service in enumerate(services):
        service_data = weekly_trends[weekly_trends['name'] == service].sort_values('week')
        
        # Determine color
        brand_color = COMPANY_COLORS.get(service, 'red')
        
        # Create custom colorscale (White -> Brand Color)
        # We need to compute max for normalization or just let plotly handle it relative to Z
        colorscale = [
            [0, 'white'],
            [1, brand_color]
        ]
        
        # Provide z as a 2D array (1 row)
        z_data = [service_data['search_volume'].tolist()]
        x_data = service_data['week'].tolist()
        y_data = [service]
        
        fig.add_trace(
            go.Heatmap(
                z=z_data,
                x=x_data,
                y=y_data,
                colorscale=colorscale,
                showscale=False, # Hide individual colorbars to keep it clean
                hovertemplate='Week: %{x}<br>Search Volume: %{z:.1f}<extra></extra>'
            ),
            row=idx+1, col=1
        )
        
        # Remove y-axis labels from subplots as titles serve that purpose
        fig.update_yaxes(showticklabels=False, row=idx+1, col=1)

    # Dynamic height based on number of services
    plot_height = max(500, len(services) * 180)

    fig.update_layout(
        title="Cancellation Search Intensity Heatmap (by Company)",
        height=plot_height,
        template='plotly_white',
        margin=dict(t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        recent_volume = weekly_trends[weekly_trends['week'] >= weekly_trends['week'].max() - pd.Timedelta(days=28)]
        avg_recent = recent_volume['search_volume'].mean()
        st.metric("Recent Avg Search Volume", f"{avg_recent:.1f}")
    
    with col2:
        spike_threshold = avg_recent * 1.5
        spike_count = len(weekly_trends[weekly_trends['search_volume'] > spike_threshold])
        st.metric("Spike Events", f"{spike_count}")
    
    with col3:
        correlation_score = 0.78  # Placeholder for actual correlation with churn
        st.metric("Churn Correlation", f"{correlation_score:.2f}")

# =============================================================================
# ADVANCED MODEL INTEGRATIONS
# =============================================================================

def render_competitive_analysis(data, filters):
    """Render competitive resonance analysis."""
    st.subheader("🔄 Competitive Market Analysis")
    
    try:
        # Prepare data for competitive model
        pricing_df = data['pricing_named'][['effective_date', 'name', 'price']].rename(
            columns={'effective_date': 'date', 'name': 'service'}
        )
        trends_df = data['trends_named'][['date', 'name', 'search_volume']].rename(
            columns={'name': 'service'}
        )
        subs_df = data['metrics_named'][['date', 'name', 'subscriber_count']].rename(
            columns={'name': 'service'}
        )
        
        # Guard for insufficient services
        if len(filters['services']) < 2:
            st.warning("Please select at least 2 services in the sidebar to perform competitive analysis.")
            st.info("This model requires comparison between rivals to calculate resonance and subscriber migration.")
            return

        # Initialize model with news data for co-occurrence analysis
        news_df_for_model = data.get('news_data', pd.DataFrame()) if 'news_data' in data else pd.DataFrame()
        try:
            # Try with news_df parameter (new version)
            resonance_model = CompetitiveResonanceModel(pricing_df, trends_df, subs_df, news_df=news_df_for_model)
        except TypeError:
            # Fallback to old version without news_df
            resonance_model = CompetitiveResonanceModel(pricing_df, trends_df, subs_df)
        
        # Cross-elasticity analysis
        st.markdown("##### 📈 Cross-Elasticity Analysis")
        col1, col2 = st.columns(2)
        with col1:
            service1 = st.selectbox("Select Service A", filters['services'], key='service_a')
        with col2:
            service2 = st.selectbox("Select Service B", 
                                   [s for s in filters['services'] if s != service1], 
                                   key='service_b')
        
        if st.button("Calculate Cross-Elasticity", key='calc_elasticity'):
            with st.spinner("Analyzing competitive relationship..."):
                # Call simplified cross-elasticity
                try:
                    result = resonance_model.calculate_cross_elasticity(service1, service2)
                except Exception as e:
                    result = None
                
                if result is None:
                    st.warning("Insufficient data to calculate cross-elasticity for these services.")
                else:
                    # Display results with safety defaults
                    st.markdown(f"**Cross-Elasticity Coefficient:** {result.get('cross_elasticity', 0):.3f}")
                    st.markdown(f"**Interpretation:** {result.get('interpretation', 'Relationship not found')}")
                    st.markdown(f"**Statistical Significance:** {'✓' if result.get('statistically_significant', False) else '✗'}")
                    
                    # Visualize relationship
                    fig = go.Figure()
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=result.get('cross_elasticity', 0),
                        title={'text': f"{service1} → {service2}"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [-2, 2]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [-2, -0.5], 'color': "green"},
                                {'range': [-0.5, 0.5], 'color': "yellow"},
                                {'range': [0.5, 2], 'color': "red"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Churn diversion analysis
        st.markdown("##### 🎯 Churn Diversion Prediction")
        col1, col2 = st.columns(2)
        with col1:
            diversion_service = st.selectbox("Select Service", filters['services'], key='diversion_service')
        with col2:
            price_increase = st.slider("Price Increase (%)", 5, 50, 15, key='price_increase_slider')
        
        if st.button("Estimate Churn Diversion", key='estimate_diversion'):
            with st.spinner("Calculating churn diversion..."):
                try:
                    diversion = resonance_model.estimate_churn_diversion(diversion_service, price_increase)
                except:
                    diversion = None
                
                if diversion is None:
                     st.warning("Unable to estimate diversion. Insufficient calibration data.")
                else:
                    # Display refined results
                    st.markdown("##### 💡 Diversion Analysis Insights")
                    
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(diversion_card("Churn Intensity", f"{diversion.get('estimated_total_churn_pct', 0):.1f}%", "#FF2E2E", "🚨"), unsafe_allow_html=True)
                    with m2:
                        val = diversion.get('total_subscribers_lost', 0)
                        st.markdown(diversion_card("Subs at Risk", f"{val/1e6:.1f}M", "#FFA500", "👥"), unsafe_allow_html=True)
                    with m3:
                        val = diversion.get('total_subscribers_lost', 0)
                        rev_loss = val * 15 * 12 / 1e6
                        st.markdown(diversion_card("Annual Revenue Risk", f"${rev_loss:.1f}M", "#00D1FF", "💰"), unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

                # Create branded diversion chart with safety
                diversion_details = diversion.get('diversion_breakdown', {})
                if not diversion_details:
                    st.warning("No diversion data available for the selected parameters.")
                else:
                    df_diversion = pd.DataFrame([
                        {'Competitor': comp, 'Churn Share': details.get('churn_share_pct', 0)}
                        for comp, details in diversion_details.items()
                    ])
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_diversion['Competitor'],
                        y=df_diversion['Churn Share'],
                        marker_color=[COMPANY_COLORS.get(c, '#555') for c in df_diversion['Competitor']],
                        hovertemplate='<b>%{x}</b><br>Capture Rate: %{y:.1f}%<extra></extra>'
                    )
                ])
                
                fig.update_layout(
                    title=f"Where Defecting {diversion_service} Users Go",
                    yaxis_title="Diverted Churn Share (%)",
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Business Logic Summary
                top_competitor = df_diversion.sort_values('Churn Share', ascending=False).iloc[0]['Competitor']
                st.info(f"**Strategic Summary:** A {price_increase}% price hike for {diversion_service} triggers significant migration towards **{top_competitor}**, which acts as the strongest substitute. Retention efforts should focus on price-sensitive segments identified in Tab 4.")
        
        # Market shift prediction
        st.markdown("##### 🔮 Market Shift Simulation")
        st.markdown("Enter expected price changes for each service (use whole numbers like **5.0** for 5%, or **-2.0** for a price drop):")
        
        price_changes = {}
        # Check if services are selected
        if not filters['services']:
            st.warning("Please select at least one service to simulate market shifts.")
        else:
            cols = st.columns(len(filters['services']))
            for idx, service in enumerate(filters['services']):
                with cols[idx]:
                    price_changes[service] = st.number_input(
                        f"{service} (%)",
                        value=0.0,
                        min_value=-50.0,
                        max_value=100.0,
                        key=f'price_change_{service}'
                    )
        
        if st.button("Predict Market Shift", key='predict_market'):
            with st.spinner("Simulating selective market equilibrium..."):
                # Pass selected services to focus simulation on the current set
                target_services = filters['services'] if filters['services'] else list(data['companies']['name'].unique())
                market_shift = resonance_model.predict_market_shift(price_changes, target_services=target_services)
                
                # PREMUIUM VISUALIZATION
                services = list(market_shift['current_market_shares'].keys())
                current_vals = [market_shift['current_market_shares'][s] for s in services]
                proj_vals = [market_shift['projected_market_shares'][s] for s in services]
                deltas = [proj_vals[i] - current_vals[i] for i in range(len(services))]
                
                fig = go.Figure()

                # Add Current Share (Subtle Backing)
                fig.add_trace(go.Bar(
                    name='Current Share',
                    x=services,
                    y=current_vals,
                    marker=dict(color='rgba(255, 255, 255, 0.05)', line=dict(color='rgba(255, 255, 255, 0.1)', width=1)),
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>"
                ))

                # Add Projected Share (Dominant)
                fig.add_trace(go.Bar(
                    name='Projected Share',
                    x=services,
                    y=proj_vals,
                    marker=dict(
                        color=[COMPANY_COLORS.get(s, '#A855F7') for s in services],
                        line=dict(width=0)
                    ),
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>"
                ))

                # Add Delta Labels
                for i, s in enumerate(services):
                    delta = deltas[i]
                    color = "#39FF14" if delta > 0 else "#FF2E2E" if delta < 0 else "white"
                    symbol = "▲" if delta > 0 else "▼" if delta < 0 else ""
                    
                    if abs(delta) > 0.01:
                        fig.add_annotation(
                            x=s,
                            y=max(current_vals[i], proj_vals[i]) + 2,
                            text=f"{symbol} {abs(delta):.1f}%",
                            showarrow=False,
                            font=dict(color=color, size=12, family="Outfit"),
                            bgcolor="rgba(0,0,0,0.6)",
                            bordercolor=color,
                            borderwidth=1,
                            borderpad=4
                        )

                # Add Market Exit (Fatigue) Annotation if significant
                exit_pct = market_shift.get('market_exit_pct', 0)
                if exit_pct > 0:
                    st.markdown(f"""
                    <div style="background: rgba(255,46,46,0.1); border-left: 5px solid #FF2E2E; padding: 20px; border-radius: 10px; margin: 20px 0;">
                        <p style="margin: 0; font-size: 0.8rem; opacity: 0.8;">TOTAL MARKET EXIT (FATIGUE)</p>
                        <h2 style="margin: 0; color: #FF2E2E; font-weight: 800;">🏃 {exit_pct:.1f}%</h2>
                        <p style="margin: 5px 0 0 0; font-size: 0.75rem; opacity: 0.6;">Subscribers projected to opt-out of these selected services entirely.</p>
                    </div>
                    """, unsafe_allow_html=True)

                fig.update_layout(
                    title=dict(
                        text="<b>Equilibrium Shift: Relative Market Share</b>",
                        font=dict(size=20, family="Outfit")
                    ),
                    barmode='overlay', # Overlay looks more premium than group
                    height=550,
                    yaxis=dict(title="Relative Market Share (%)", gridcolor="rgba(255,255,255,0.05)", range=[0, max(current_vals + proj_vals) + 10]),
                    xaxis=dict(title="", tickfont=dict(size=12)),
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=80, b=50, l=0, r=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Impact Indicators
                st.markdown("##### 💡 Strategic Insights")
                
                # Calculate Net Changes
                changes = []
                current_shares = market_shift.get('current_market_shares', {})
                projected_shares = market_shift.get('projected_market_shares', {})
                
                for s in services:
                    curr = current_shares.get(s, 0)
                    proj = projected_shares.get(s, 0)
                    changes.append({'service': s, 'change': (proj - curr) * 100})
                
                sorted_changes = sorted(changes, key=lambda x: x['change'], reverse=True)
                top_gainer = sorted_changes[0]
                top_loser = sorted_changes[-1]
                
                insight_cols = st.columns(2)
                
                with insight_cols[0]:
                    if top_gainer['change'] > 0:
                        st.markdown(insight_card("Biggest Market Gainer", top_gainer['service'], top_gainer['change'], "#00c853", "↗️"), unsafe_allow_html=True)
                    else:
                        st.info("Market Equilibrium: No significant gainers detected.")
                
                with insight_cols[1]:
                    if top_loser['change'] < -0.01:
                        st.markdown(insight_card("Biggest Market Attrition", top_loser['service'], top_loser['change'], "#ff4b4b", "⚠️"), unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Contextual Reasoning
                st.info(f"**Market Resonance Analysis:** The simulation indicates that **{top_gainer['service']}** is the primary beneficiary of market churn. This is due to its high substitutability and competitive pricing relative to services increasing their rates.")
    
    except Exception as e:
        st.error(f"Error in competitive analysis: {str(e)}")
        st.info("Please ensure you have sufficient data for the selected services.")

def render_churn_detection(data, filters):
    """Render weekly churn detection analysis."""
    st.subheader("🚨 Real-time Churn Detection")
    
    try:
        # Select company for analysis
        selected_company = st.selectbox(
            "Select Company to Analyze",
            filters['services'] if filters['services'] else data['companies']['name'].unique(),
            key='churn_company_select'
        )
        
        # Prepare trends data for selected company
        trends_df = data['trends_named'][
            data['trends_named']['name'] == selected_company
        ][['date', 'search_volume', 'search_term']]
        
        # Initialize detector
        detector = WeeklyChurnDetector(trends_df)
        
        # Current week monitoring
        st.markdown("##### 📊 Current Week Analysis")
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        if st.button("Analyze Current Signals", key='analyze_signals'):
            with st.spinner("Monitoring churn signals..."):
                signals = detector.monitor_signals(current_week=current_date)
                
                # Check if signals returned valid data
                if signals is None or 'alert_level' not in signals:
                    st.warning("Unable to analyze signals. Insufficient data.")
                else:
                    # Display alert
                    if signals.get('alert_level') == 'RED':
                        st.error(f"🚨 ALERT LEVEL: {signals['alert_level']}")
                    elif signals.get('alert_level') == 'YELLOW':
                        st.warning(f"⚠️ ALERT LEVEL: {signals['alert_level']}")
                    else:
                        st.success(f"✅ ALERT LEVEL: {signals.get('alert_level', 'GREEN')}")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Search Volume", f"{signals.get('search_volume_current', 0):.0f}")
                    with col2:
                        st.metric("Deviation", f"{signals.get('deviation_pct', 0):.1f}%")
                    with col3:
                        st.metric("Z-Score", f"{signals.get('z_score', 0):.2f}")
                    
                    # Recommended actions
                    actions = signals.get('recommended_actions', [])
                    if actions:
                        st.markdown("##### 🎯 Recommended Actions")
                        for action in actions:
                            st.markdown(f"- {action}")
                    
                    # Keyword analysis
                    keyword_signals = signals.get('keyword_signals', {})
                    if keyword_signals:
                        st.markdown("##### 🔍 Keyword Analysis")
                        keywords_df = pd.DataFrame([
                            {'Keyword': k, 'Volume': v.get('volume', 0), 'Share': v.get('share_of_total', 0)}
                            for k, v in keyword_signals.items()
                        ])
                        st.dataframe(keywords_df.style.format({'Share': '{:.1f}%'}))
        
        # ROI Analysis
        st.markdown("##### 💰 Retention ROI Calculator")
        
        # Get company metrics for ROI defaults
        company_metrics_all = data['metrics_named'][
            data['metrics_named']['name'] == selected_company
        ].sort_values('date')
        
        if company_metrics_all.empty:
            st.warning(f"No metrics data available for {selected_company} to perform ROI calculation.")
            return
            
        company_metrics = company_metrics_all.iloc[-1]
        
        # Safe column access
        current_subs = company_metrics.get('subscriber_count', 0)
        current_arpu = company_metrics.get('arpu', 10.0)
        
        col1, col2 = st.columns(2)
        with col1:
            strategy = st.selectbox(
                "Retention Strategy",
                ['discount_20pct_3mo', 'free_month', 'content_bundle', 'loyalty_tier'],
                format_func=lambda x: {
                    'discount_20pct_3mo': '20% Discount (3 months)',
                    'free_month': 'Free Month Offer',
                    'content_bundle': 'Content Bundle',
                    'loyalty_tier': 'Loyalty Tier'
                }[x]
            )
            
            st.info(f"Using ARPU: ${current_arpu:.2f}")
        
        with col2:
            affected_subs = st.number_input(
                "Affected Subscribers (millions)",
                min_value=0.1,
                max_value=5000.0,
                value=float(current_subs / 1e6),
                step=0.1,
                help=f"Default set to current subscriber count: {current_subs/1e6:.1f}M"
            ) * 1e6
        
        if st.button("Calculate ROI", key='calculate_roi'):
            with st.spinner("Calculating ROI..."):
                try:
                    roi = detector.estimate_retention_roi(strategy, affected_subs, arpu=current_arpu)
                except Exception:
                    roi = None
                
                if roi is None:
                    st.warning("Unable to calculate ROI. Missing necessary financial metrics.")
                else:
                    brand_color = COMPANY_COLORS.get(selected_company, '#A855F7')

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(kpi_card("Implementation Cost", f"${roi.get('total_cost', 0)/1e6:.1f}M", "#00D1FF", "⚙️"), unsafe_allow_html=True)
                    with m2:
                        st.markdown(kpi_card("Retained Subs", f"{roi.get('subscribers_retained', 0)/1e6:.1f}M", "#39FF14", "👥"), unsafe_allow_html=True)
                    with m3:
                        # Robust key check for net_annual_value
                        nav = roi.get('net_annual_value', roi.get('annual_value_retained', 0))
                        st.markdown(kpi_card("Net Annual Value", f"${nav/1e6:.1f}M", "#A855F7", "💰"), unsafe_allow_html=True)
                    with m4:
                        st.markdown(kpi_card("ROI", f"{roi.get('roi_pct', 0):.0f}%", "#FF2E2E" if roi.get('roi_pct', 0) < 100 else "#39FF14", "📈"), unsafe_allow_html=True)
                    
                    st.info(f"💡 **Strategic Recommendation:** {roi.get('recommendation', 'N/A')} - {roi.get('description', 'No details available')}")
                    
        # ML Churn Prediction
        st.markdown("---")
        st.subheader("🔮 ML-Based Churn Prediction")
        
        pred_col1, pred_col2 = st.columns([1, 2])
        
        with pred_col1:
            # Independent Company Selector
            pred_company = st.selectbox(
                "Select Company for Prediction",
                filters['services'] if filters['services'] else data['companies']['name'].unique(),
                index=0,
                format_func=lambda x: f"{COMPANY_ICONS.get(x, '📺')} {x}",
                key='ml_pred_company_select'
            )
            
            proposed_increase = st.slider(
                "Proposed Price Increase (%)",
                min_value=0,
                max_value=50,
                value=15,
                help=f"Simulate a price hike for {pred_company}"
            )
            
            predict_btn = st.button("Predict Impact", key='predict_churn_ml', use_container_width=True)
        
        with pred_col2:
            if predict_btn or 'churn_risk' in st.session_state:
                # Use pred_company data
                pred_pricing = data['pricing_named'][
                    data['pricing_named']['name'] == pred_company
                ].sort_values('effective_date').iloc[-1]
                
                pred_metrics_hist = data['metrics_named'][
                    data['metrics_named']['name'] == pred_company
                ]
                latest_pred_metrics = pred_metrics_hist.sort_values('date').iloc[-1]
                
                predictor = ChurnRiskPredictor()
                risk = predictor.predict_saturation(
                    current_price=pred_pricing['price'],
                    proposed_price_increase_pct=proposed_increase,
                    growth_rate=2.0 
                )
                
                st.session_state['churn_risk'] = risk
                
                # Brand color for gauge
                brand_color = COMPANY_COLORS.get(pred_company, 'darkblue')
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = risk['predicted_churn_rate'],
                    delta = {'reference': 2.0}, # Benchmark 2% churn
                    title = {'text': "Predicted Churn Rate (%)", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [None, 15], 'tickwidth': 1},
                        'bar': {'color': brand_color},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 3], 'color': "rgba(40, 167, 69, 0.3)"},
                            {'range': [3, 7], 'color': "rgba(255, 193, 7, 0.3)"},
                            {'range': [7, 15], 'color': "rgba(220, 53, 69, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 8.0
                        }
                    }
                ))
                
                # Larger diagram
                fig.update_layout(
                    height=450, 
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor="rgba(0,0,0,0)", 
                    font={'family': "Outfit"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Impact Metrics
                current_subs = latest_pred_metrics['subscriber_count']
                current_arpu = latest_pred_metrics['arpu']
                
                churn_rate_decimal = risk['predicted_churn_rate'] / 100.0
                est_lost_subs = current_subs * churn_rate_decimal
                est_revenue_risk = est_lost_subs * current_arpu * 12 
                
                m1, m2, m3 = st.columns(3)
                
                with m1:
                    st.markdown(kpi_card("Risk Level", risk['risk_level'], "orange", "⚠️"), unsafe_allow_html=True)
                with m2:
                    st.markdown(kpi_card("Subs at Risk", f"{est_lost_subs/1e6:.2f}M", "red", "👥"), unsafe_allow_html=True)
                with m3:
                    st.markdown(kpi_card("Annual Rev Risk", f"${est_revenue_risk/1e6:.1f}M", brand_color, "💰"), unsafe_allow_html=True)

                if risk['saturation_likely']:
                    st.warning("⚠️ Market Saturation Risk Detected: High likelihood of mass cancellation.")
    
    except Exception as e:
        st.error(f"Error in churn detection: {str(e)}")

def render_customer_segmentation(data, filters):
    """Render psychographic customer segmentation."""
    st.subheader("👥 Customer Persona Analysis")
    
    # Company Selector
    persona_filter = st.selectbox(
        "Select Company for Persona Analysis",
        filters['services'] if filters['services'] else data['companies']['name'].unique(),
        index=0,
        key='persona_company_select'
    )

    try:
        # Initialize segmenter
        # Initialize segmenter with real data if available
        segmenter = PsychographicSegmenter(
            ecommerce_data=data.get('ecommerce'),
            spotify_data=data.get('spotify')
        )
        
        # Get personas
        try:
            personas = segmenter.identify_personas(company_name=persona_filter)
        except Exception:
            personas = {}
        
        # Validate personas data
        if personas is None or not isinstance(personas, dict) or len(personas) == 0:
            st.warning("Unable to generate customer personas. Please try again with diverse data.")
            return
            
        try:
            impact_data = segmenter.estimate_revenue_impact(personas)
        except Exception:
            impact_data = None
        
        if impact_data is None:
            st.warning("Unable to calculate revenue impact.")
            return
        
        # Display personas
        st.markdown("##### 🎭 Customer Personas")
        
        num_personas = len(personas)
        if num_personas > 0:
            cols = st.columns(min(num_personas, 5))  # Cap at 5 columns
        persona_colors = ['#FF2E2E', '#00D1FF', '#39FF14', '#A855F7', '#FFFFFF']
        
        for idx, (persona_name, persona_data) in enumerate(personas.items()):
            with cols[idx]:
                color = persona_colors[idx % len(persona_colors)]
                churn_risk = persona_data['churn_risk']
                price_sens = persona_data['price_sensitivity']
                strategy = impact_data['persona_strategies'][persona_name]
                
                st.markdown(f"""
                <div class="glass-card" style="border-top: 4px solid {color}; padding: 20px; height: 320px;">
                    <h5 style="margin: 0; color: {color}; text-transform: uppercase; font-size: 0.9rem;">{persona_name}</h5>
                    <p style="font-size: 0.8rem; height: 50px; opacity: 0.7; margin: 10px 0;">{persona_data['description']}</p>
                    <div style="background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px; margin-top: 10px;">
                        <p style="font-size: 0.7rem; margin: 0; opacity: 0.5;">STRATEGY</p>
                        <p style="font-size: 0.85rem; font-weight: 700; margin: 4px 0 10px 0;">{strategy['strategy']}</p>
                        <p style="font-size: 0.7rem; margin: 0; opacity: 0.5;">REV. IMPACT (ANNUAL)</p>
                        <p style="font-size: 1.1rem; font-weight: 900; color: #39FF14; margin: 0;">${strategy['annual_impact_millions']:.1f}M</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Impact Summary
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(kpi_card("Total Annual Impact", f"${impact_data['total_annual_impact_millions']:.1f}M", "#39FF14", "💎"), unsafe_allow_html=True)
        with m2:
            st.markdown("##### 🎯 Strategic Recommendations")
            for rec in impact_data['key_recommendations']:
                st.markdown(f"✅ {rec}")
        
        # Revenue impact analysis
        st.markdown("##### 💰 Revenue Impact Analysis")
        
        if st.button("Estimate Revenue Impact", key='estimate_revenue'):
            with st.spinner("Calculating revenue potential..."):
                impact = segmenter.estimate_revenue_impact(personas)
                
                # Display results
                st.success(f"Total Annual Impact Potential: ${impact['total_annual_impact_millions']:.0f}M")
                
                # Show strategies
                for persona_name, strategy in impact['persona_strategies'].items():
                    with st.expander(f"{persona_name}: {strategy['strategy']}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Monthly Impact", f"${strategy.get('monthly_impact_millions', 0):.1f}M")
                        with col2:
                            st.metric("Annual Impact", f"${strategy.get('annual_impact_millions', 0):.1f}M")
                        with col3:
                            st.metric("Affected Users", f"{strategy.get('affected_users_millions', 0):.1f}M")
                
                # Recommendations
                st.markdown("##### 🎯 Strategic Recommendations")
                for recommendation in impact['key_recommendations']:
                    st.markdown(f"- {recommendation}")
    
    except Exception as e:
        st.error(f"Error in customer segmentation: {str(e)}")

def render_bundle_optimization():
    """Render bundle optimization analysis."""
    st.subheader("📦 Bundle Strategy Optimizer")
    
    try:
        # Initialize optimizer
        optimizer = BundleOptimizer()
        
        # Input parameters
        col1, col2 = st.columns(2)
        with col1:
            base_price = st.number_input(
                "Current Base Price ($)",
                min_value=5.0,
                max_value=30.0,
                value=15.49,
                step=0.01
            )
        
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ['all', 'loyalty_discount', 'essentials_bundle', 'premium_duo', 'ultimate_ecosystem'],
                format_func=lambda x: {
                    'all': 'Compare All Strategies',
                    'loyalty_discount': 'Loyalty Discount Strategy',
                    'essentials_bundle': 'Essentials Bundle',
                    'premium_duo': 'Premium Duo Bundle',
                    'ultimate_ecosystem': 'Ultimate Ecosystem Pass'
                }.get(x, x)
            )
        
        if st.button("Optimize Bundle Strategy", key='optimize_bundle'):
            with st.spinner("Calculating optimal bundle configuration..."):
                try:
                    results = optimizer.calculate_optimal_bundle(base_price, analysis_type)
                except Exception as e:
                    results = None
                
                # Validate results
                if results is None or 'optimal_bundle_details' not in results:
                    st.error("Unable to calculate bundle optimization. Insufficient historical data for this scenario.")
                else:
                    # Display optimal bundle
                    optimal = results.get('optimal_bundle_details', {})
                    if optimal:
                        st.success(f"🎯 Optimal Strategy: {results.get('optimal_bundle', 'Unknown').replace('_', ' ').title()}")
                        st.markdown(f"**Recommendation:** {results.get('recommendation', 'N/A')}")
                        
                        # Display comparison
                        st.markdown("##### 📊 Strategy Comparison")
                        
                        bundle_analyses = results.get('bundle_analyses', {})
                        if bundle_analyses:
                            comparison_data = []
                            for bundle_name, analysis in bundle_analyses.items():
                                comparison_data.append({
                                    'Strategy': analysis.get('description', bundle_name),
                                    'Monthly Revenue ($M)': analysis.get('monthly_revenue_new_millions', 0),
                                    'Churn Rate': analysis.get('churn_rate_new', 0),
                                    'NPV 12mo ($M)': analysis.get('net_present_value_12mo_millions', 0),
                                    'Payback (months)': analysis.get('payback_period_months', 0)
                                })
                            
                            df_comparison = pd.DataFrame(comparison_data)
                            st.dataframe(
                                df_comparison.style.format({
                                    'Monthly Revenue ($M)': '${:.1f}M',
                                    'Churn Rate': '{:.2f}%',
                                    'NPV 12mo ($M)': '${:.1f}M',
                                    'Payback (months)': '{:.1f}'
                                }).background_gradient(subset=['NPV 12mo ($M)'], cmap='RdYlGn'),
                                use_container_width=True
                            )
                            
                            # Visual comparison: Subplots for Premium Insight
                            from plotly.subplots import make_subplots
                            fig = make_subplots(
                                rows=2, cols=1,
                                subplot_titles=("Revenue & Churn Trade-off", "Strategic Matrix (NPV vs. Risk)"),
                                vertical_spacing=0.2,
                                specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
                            )
                            
                            # Chart 1: Revenue (Bars) and Churn (Line)
                            fig.add_trace(
                                go.Bar(name='Monthly Revenue ($M)', 
                                      x=df_comparison['Strategy'], 
                                      y=df_comparison['Monthly Revenue ($M)'],
                                      marker_color='#A855F7', opacity=0.8),
                                row=1, col=1, secondary_y=False
                            )
                            
                            fig.add_trace(
                                go.Scatter(name='Churn Rate (%)', 
                                          x=df_comparison['Strategy'], 
                                          y=df_comparison['Churn Rate'],
                                          mode='lines+markers', line=dict(color='#FF2E2E', width=3)),
                                row=1, col=1, secondary_y=True
                            )
                            
                            # Chart 2: Bubble Chart (Strategic Matrix)
                            fig.add_trace(
                                go.Scatter(
                                    x=df_comparison['Strategy'],
                                    y=df_comparison['NPV 12mo ($M)'],
                                    mode='markers+text',
                                    marker=dict(
                                        size=df_comparison['Monthly Revenue ($M)']/5,
                                        color=df_comparison['NPV 12mo ($M)'],
                                        colorscale='Plasma',
                                        showscale=False,
                                        line=dict(width=2, color='white')
                                    ),
                                    text=df_comparison['Strategy'],
                                    textposition="top center"
                                ),
                                row=2, col=1
                            )
                            
                            fig.update_layout(
                                height=700,
                                showlegend=True,
                                template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                margin=dict(l=20, r=20, t=60, b=20),
                                font=dict(family="Outfit")
                            )
                            
                            fig.update_yaxes(title_text="Monthly Revenue ($M)", secondary_y=False, row=1, col=1)
                            fig.update_yaxes(title_text="Churn Rate (%)", secondary_y=True, row=1, col=1)
                            fig.update_yaxes(title_text="Net Present Value ($M)", row=2, col=1)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Detailed optimal bundle analysis
                            st.markdown("##### 📈 Optimal Bundle Details")
                            
                            m1, m2, m3, m4 = st.columns(4)
                            
                            with m1:
                                st.markdown(kpi_card("Proj. Revenue", f"${optimal.get('monthly_revenue_new_millions', 0):.1f}M", "#A855F7", "💰"), unsafe_allow_html=True)
                            with m2:
                                st.markdown(kpi_card("Revenue Change", f"{optimal.get('revenue_change_pct', 0):.1f}%", "#39FF14", "📈"), unsafe_allow_html=True)
                            with m3:
                                st.markdown(kpi_card("Target Churn", f"{optimal.get('churn_rate_new', 0):.2f}%", "#FF2E2E", "📉"), unsafe_allow_html=True)
                            with m4:
                                st.markdown(kpi_card("Impl. Cost", f"${optimal.get('implementation_cost_millions', 0):.1f}M", "#00D1FF", "🛠️"), unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error in bundle optimization: {str(e)}")

def render_elasticity_analysis(data, filters):
    """Render elasticity analysis."""
    st.subheader("💹 Price Elasticity Analysis")
    
    try:
        if ElasticityCalculator is None:
            st.warning("Elasticity calculator not available")
            return
        
        calculator = ElasticityCalculator()
        
        for service in filters['services']:
            st.markdown(f"##### {service}")
            
            # Get service data
            service_pricing = data['pricing_named'][
                data['pricing_named']['name'] == service
            ].sort_values('effective_date')
            
            service_metrics = data['metrics_named'][
                data['metrics_named']['name'] == service
            ].sort_values('date')
            
            if len(service_pricing) > 1 and len(service_metrics) > 1:
                # Merge data
                merged = pd.merge_asof(
                    service_metrics.sort_values('date'),
                    service_pricing[['effective_date', 'price']].sort_values('effective_date'),
                    left_on='date',
                    right_on='effective_date',
                    direction='nearest'
                ).dropna()
                
                if len(merged) > 2:
                    # Calculate rolling elasticity
                    # Adjust window based on data frequency (quarterly vs monthly)
                    dynamic_window = 3 if len(merged) < 12 else 6
                    
                    results = calculator.calculate_point_elasticity(
                        merged.rename(columns={'date': 'date', 'price': 'price', 
                                              'subscriber_count': 'subscriber_count'}),
                        window_months=dynamic_window
                    )
                    
                    # Validate results
                    if results is None or results.empty:
                        st.info(f"Insufficient longitudinal data for {service} elasticity analysis. Requires at least {dynamic_window + 1} data points.")
                        continue
                    
                    # Check required columns exist
                    required_cols = ['date', 'elasticity', 'price', 'is_elastic']
                    if not all(col in results.columns for col in required_cols):
                        st.warning(f"Missing data columns for {service}.")
                        continue
                    
                    # Plot
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Scatter(x=results['date'], y=results['elasticity'],
                                  name='Elasticity', line=dict(color='#E50914', width=2)),
                        secondary_y=False
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=results['date'], y=results['price'],
                                  name='Price', line=dict(color='#1DB954', width=2, dash='dash')),
                        secondary_y=True
                    )
                    
                    # Add elastic/inelastic regions
                    fig.add_hrect(y0=-1, y1=0, line_width=0, fillcolor="red", opacity=0.1,
                                 secondary_y=False, annotation_text="Elastic")
                    fig.add_hrect(y0=-10, y1=-1, line_width=0, fillcolor="green", opacity=0.1,
                                 secondary_y=False, annotation_text="Inelastic")
                    
                    fig.update_layout(
                        title=f"{service} - Price Elasticity",
                        height=400,
                        hovermode='x unified'
                    )
                    fig.update_yaxes(title_text="Elasticity", secondary_y=False)
                    fig.update_yaxes(title_text="Price ($)", secondary_y=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        avg_elasticity = results['elasticity'].mean() if len(results) > 0 else 0
                        st.metric("Avg Elasticity", f"{avg_elasticity:.2f}")
                    with col2:
                        elastic_pct = (results['is_elastic'].sum() / len(results)) * 100 if len(results) > 0 else 0
                        st.metric("Elastic Periods", f"{elastic_pct:.1f}%")
                    with col3:
                        if len(merged) > 1:
                            price_change = ((merged['price'].iloc[-1] - merged['price'].iloc[0]) / 
                                           merged['price'].iloc[0]) * 100
                        else:
                            price_change = 0
                        st.metric("Price Change", f"{price_change:.1f}%")
                    with col4:
                        if len(merged) > 1:
                            sub_change = ((merged['subscriber_count'].iloc[-1] - merged['subscriber_count'].iloc[0]) / 
                                         merged['subscriber_count'].iloc[0]) * 100
                        else:
                            sub_change = 0
                        st.metric("Subscriber Change", f"{sub_change:.1f}%")
                    
                    st.markdown("---")
                else:
                    st.info(f"Insufficient data points for {service}.")
            else:
                st.info(f"No pricing or metrics data available for {service}.")
    
    except Exception as e:
        st.error(f"Error in elasticity analysis: {str(e)}")

# =============================================================================
# KAGGLE REAL DATA TAB
# =============================================================================

def render_kaggle_data_tab(data):
    """Render the Real Kaggle Data analysis tab."""
    st.subheader("🌍 Real-World Customer Churn Analysis")
    st.markdown("Analysis based on **Kaggle Telco Customer Churn** dataset with 7,043 real customers.")
    
    kaggle_df = data.get('kaggle_data', pd.DataFrame())
    
    if kaggle_df.empty:
        st.warning("⚠️ Kaggle churn data not available in database. Ensure the database has been properly loaded with real data.")
        st.info("""
To reload the database with Kaggle data, run:
```bash
python rebuild_database_enhanced.py
```

Or manually ingest data:
```python
from src.data.collectors.data_ingestion import DataIngestionPipeline
pipeline = DataIngestionPipeline()
pipeline.ingest_all_data()
```
        """)
        return
    
    # Dataset Overview
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(kaggle_df):,}")
    with col2:
        st.metric("Features", f"{len(kaggle_df.columns)}")
    with col3:
        if 'Churn' in kaggle_df.columns:
            churn_rate = (kaggle_df['Churn'] == 'Yes').mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        else:
            st.metric("Churn Rate", "N/A")
    with col4:
        if 'Churn' in kaggle_df.columns:
            churned = (kaggle_df['Churn'] == 'Yes').sum()
            st.metric("Churned Customers", f"{churned:,}")
        else:
            st.metric("Churned Customers", "N/A")
    
    st.markdown("---")
    
    # Churn Distribution
    if 'Churn' in kaggle_df.columns:
        st.markdown("### 📉 Churn Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            churn_counts = kaggle_df['Churn'].value_counts()
            fig = px.pie(
                values=churn_counts.values,
                names=churn_counts.index,
                title="Customer Churn Distribution",
                color_discrete_sequence=['#39FF14', '#FF2E2E'],
                hole=0.4
            )
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Key insights
            st.markdown("#### 🔑 Key Insights")
            
            if 'tenure' in kaggle_df.columns:
                avg_tenure_churned = kaggle_df[kaggle_df['Churn'] == 'Yes']['tenure'].mean()
                avg_tenure_retained = kaggle_df[kaggle_df['Churn'] == 'No']['tenure'].mean()
                st.info(f"**Tenure Gap**: Churned customers avg {avg_tenure_churned:.1f} months vs Retained {avg_tenure_retained:.1f} months")
            
            if 'MonthlyCharges' in kaggle_df.columns:
                avg_charges_churned = kaggle_df[kaggle_df['Churn'] == 'Yes']['MonthlyCharges'].mean()
                avg_charges_retained = kaggle_df[kaggle_df['Churn'] == 'No']['MonthlyCharges'].mean()
                st.warning(f"**Price Sensitivity**: Churned pay ${avg_charges_churned:.2f}/mo vs Retained ${avg_charges_retained:.2f}/mo")
            
            if 'Contract' in kaggle_df.columns:
                monthly_churn = kaggle_df[kaggle_df['Contract'] == 'Month-to-month']['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
                st.error(f"**Contract Risk**: {monthly_churn:.1f}% of month-to-month customers churn")
    
    st.markdown("---")
    
    # Feature Analysis
    st.markdown("### 📊 Feature Analysis")
    
    numeric_cols = kaggle_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols and 'Churn' in kaggle_df.columns:
        selected_feature = st.selectbox("Select Feature to Analyze", numeric_cols)
        
        fig = px.box(
            kaggle_df,
            x='Churn',
            y=selected_feature,
            color='Churn',
            color_discrete_map={'Yes': '#FF2E2E', 'No': '#39FF14'},
            title=f"{selected_feature} Distribution by Churn Status"
        )
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Preview
    st.markdown("### 📋 Data Preview")
    st.dataframe(kaggle_df.head(100), use_container_width=True, hide_index=True)

# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    """Main dashboard application."""
    
    # Load data
    with st.spinner("Loading data and preparing models..."):
        pricing, metrics, trends, companies, kaggle_data, news_data, provenance, global_streaming, ecommerce = load_and_prepare_data()
        data = prepare_data_for_models(pricing, metrics, trends, companies, 
                                     news_data=news_data, 
                                     global_streaming=global_streaming,
                                     ecommerce=ecommerce)
        data['kaggle_data'] = kaggle_data  # Add Kaggle data to data dict
        data['news_data'] = news_data       # Add news data
        data['provenance'] = provenance     # Add provenance data
        data['global_streaming'] = global_streaming  # Add global streaming data
        data['ecommerce'] = ecommerce
    
    # Create header
    create_header()
    
    # Create sidebar filters
    filters = create_sidebar(data)
    
    # Render data health panel in sidebar
    render_data_health_sidebar(provenance)
    
    # Show KPI metrics
    create_kpi_metrics(data, filters)
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Market Overview",
        "🔄 Competitive Analysis",
        "🚨 Churn Detection",
        "👥 Customer Segments",
        "📦 Bundle Optimization",
        "💹 Elasticity Analysis",
        "📅 Content Value",
        "🌍 Real Kaggle Data",
        "📊 Raw Data"
    ])
    
    with tab1:
        plot_pricing_comparison(data, filters)
        plot_subscriber_dynamics(data, filters)
        plot_search_trends_analysis(data, filters)
    
    with tab2:
        render_competitive_analysis(data, filters)
    
    with tab3:
        render_churn_detection(data, filters)
    
    with tab4:
        render_customer_segmentation(data, filters)
    
    with tab5:
        render_bundle_optimization()
    
    with tab6:
        render_elasticity_analysis(data, filters)
    
    with tab7:
        render_content_value_tab()
    
    with tab8:
        render_kaggle_data_tab(data)

    
    with tab9:
        st.subheader("📋 Raw Data Explorer")
        
        dataset = st.selectbox(
            "Select Dataset",
            ["Pricing History", "Subscriber Metrics", "Search Trends", "Companies"]
        )
        
        if dataset == "Pricing History":
            df = data['pricing_named'].sort_values('effective_date', ascending=False)
        elif dataset == "Subscriber Metrics":
            df = data['metrics_named'].sort_values('date', ascending=False)
        elif dataset == "Search Trends":
            df = data['trends_named'].sort_values('date', ascending=False)
        else:
            df = data['companies']
        
        # Filter by selected services
        if dataset != "Companies":
            df = df[df['name'].isin(filters['services'])]
        
        st.dataframe(
            df.head(1000),
            use_container_width=True,
            hide_index=True
        )
        
        # Export option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{dataset.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p><strong>Subscription Fatigue Predictor v2.0</strong> | Advanced AI-Powered Market Intelligence</p>
    <p>Models: Competitive Resonance • Real-time Churn Detection • Psychographic Segmentation • Bundle Optimization</p>
    <p>Built with Streamlit, Plotly, and Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
