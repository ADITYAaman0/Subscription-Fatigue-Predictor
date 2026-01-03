"""
Configuration management for the Subscription Fatigue Predictor project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# Database
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///data/subscription_fatigue.db')
DATABASE_PATH = DATA_DIR / 'subscription_fatigue.db'

# API Keys
KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME', 'your_username')
KAGGLE_KEY = os.getenv('KAGGLE_KEY', 'your_key')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'your_api_key')

# Data Collection Settings
SERVICES = ['Netflix', 'Spotify', 'Disney Plus', 'HBO Max']
GOOGLE_TRENDS_KEYWORDS = [
    'Cancel Netflix',
    'Cancel Spotify',
    'Cancel Disney Plus',
    'streaming service expensive',
    'Netflix price increase'
]

# Analysis Parameters
ELASTICITY_WINDOW_MONTHS = 6
CHANGE_POINT_PENALTY = 10
FORECASTING_HORIZON_DAYS = 90
MAX_LAG_WEEKS = 8

# Model Parameters
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': 42
}

TRANSFORMER_PARAMS = {
    'd_model': 128,
    'nhead': 8,
    'num_layers': 6,
    'dropout': 0.1
}

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = PROJECT_ROOT / 'logs' / 'app.log'
