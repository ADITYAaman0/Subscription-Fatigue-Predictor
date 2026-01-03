"""
Main entry point and orchestration for the Subscription Fatigue Predictor.
"""

import logging
from pathlib import Path
import sys

# Setup path
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
    """Main orchestration function."""
    
    logger.info("Starting Subscription Fatigue Predictor")
    
    # Step 1: Data Collection
    logger.info("Step 1: Collecting data...")
    pipeline = DataIngestionPipeline('data/subscription_fatigue.db')
    pipeline.run_full_pipeline()
    
    logger.info("Data collection complete")
    logger.info("=" * 60)
    logger.info("All components initialized successfully!")
    logger.info("Run: streamlit run src/visualization/dashboard.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
