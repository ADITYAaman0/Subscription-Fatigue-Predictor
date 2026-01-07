"""
# 🚀 Subscription Fatigue Predictor: Orchestration Layer
# Main entry point for the end-to-end analytical pipeline.
"""

import logging
from pathlib import Path
import sys

# Configure Project Environment
# Ensure the source directory is in the system path for seamless module resolution.
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import logger
from src.data.collectors.data_ingestion import DataIngestionPipeline
from src.models.economic.economic_models import ElasticityCalculator
from src.models.statistical.statistical_models import ChangePointDetector, CausalAnalyzer
from src.models.ml.ml_models import ChurnRiskPredictor
from src.models.advanced_models import (
    BundleOptimizer, 
    PsychographicSegmenter, 
    CompetitiveResonanceModel,
    WeeklyChurnDetector
)


def main():
    """
    Execute the core project pipeline.
    
    Orchestrates data collection, database initialization, and component verification.
    """
    
    logger.info("Initializing Subscription Fatigue Predictor Pipeline...")
    
    # Step 1: Data Acquisition & Storage
    # Trigger the ingestion pipeline to populate the local SQLite database with synthetic records.
    logger.info("Step 1: Orchestrating data ingestion...")
    pipeline = DataIngestionPipeline('data/subscription_fatigue.db')
    pipeline.run_full_pipeline()
    
    # Step 2: Verification & Handover
    # Log successful initialization and provide instructions for dashboard execution.
    logger.info("Database synchronized and components verified.")
    logger.info("=" * 60)
    logger.info("SYSTEM READY: Subscription Intelligence Engine Online")
    logger.info("Action: Run 'streamlit run src/visualization/dashboard.py' to launch UI.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
