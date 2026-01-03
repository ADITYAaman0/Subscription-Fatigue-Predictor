"""
Data collection pipeline for subscription services data.
"""

import logging
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Orchestrates all data collection and storage.
    """
    
    def __init__(self, db_path='data/subscription_fatigue.db'):
        self.db_path = db_path
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.conn = None
    
    def create_sample_pricing_data(self):
        """Generate sample pricing data."""
        logger.info("Generating sample pricing data...")
        
        dates = pd.date_range('2020-01-01', '2025-12-01', freq='MS')
        data = []
        
        for service_id, service in [(1, 'Netflix'), (2, 'Spotify'), (3, 'Disney Plus')]:
            base_price = {1: 10.99, 2: 9.99, 3: 7.99}[service_id]
            
            for i, date in enumerate(dates):
                price = base_price + (i // 12) * 1.50
                
                data.append({
                    'company_id': service_id,
                    'effective_date': date.strftime('%Y-%m-%d'),
                    'price': price,
                    'currency': 'USD'
                })
        
        return pd.DataFrame(data)
    
    def create_sample_metrics_data(self):
        """Generate sample subscriber metrics."""
        logger.info("Generating sample subscriber metrics...")
        
        dates = pd.date_range('2020-01-01', '2025-12-01', freq='MS')
        data = []
        
        np = __import__('numpy')
        
        for service_id in [1, 2, 3]:
            base_subs = {1: 200_000_000, 2: 150_000_000, 3: 100_000_000}[service_id]
            
            for i, date in enumerate(dates):
                growth = np.random.uniform(-0.02, 0.05)
                subs = int(base_subs * (1 + growth * i))
                
                data.append({
                    'company_id': service_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'subscriber_count': subs,
                    'arpu': np.random.uniform(10, 15),
                    'churn_rate': np.random.uniform(2, 8)
                })
        
        return pd.DataFrame(data)
    
    def create_sample_trends_data(self):
        """Generate sample search trends."""
        logger.info("Generating sample search trends...")
        
        dates = pd.date_range('2020-01-01', '2025-12-01', freq='W')
        data = []
        
        np = __import__('numpy')
        
        for service_id, keyword in [(1, 'Cancel Netflix'), (2, 'Cancel Spotify')]:
            for date in dates:
                base_volume = np.random.randint(30, 50)
                spike = 40 if np.random.random() < 0.05 else 0
                
                data.append({
                    'company_id': service_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'search_term': keyword,
                    'search_volume': base_volume + spike,
                    'region': 'US'
                })
        
        return pd.DataFrame(data)
    
    def store_to_database(self, table_name, df):
        """Store DataFrame in SQLite database."""
        if self.conn is None:
            logger.warning("Database connection not available")
            return
        
        try:
            df.to_sql(table_name, self.conn, if_exists='append', index=False)
            logger.info(f"Stored {len(df)} records in {table_name}")
        except Exception as e:
            logger.error(f"Failed to store in {table_name}: {e}")
    
    def run_full_pipeline(self):
        """Execute complete data collection."""
        logger.info("=" * 50)
        logger.info("STARTING DATA COLLECTION PIPELINE")
        logger.info("=" * 50)
        
        try:
            # Generate sample data
            pricing_df = self.create_sample_pricing_data()
            self.store_to_database('pricing_history', pricing_df)
            
            metrics_df = self.create_sample_metrics_data()
            self.store_to_database('subscriber_metrics', metrics_df)
            
            trends_df = self.create_sample_trends_data()
            self.store_to_database('search_trends', trends_df)
            
            if self.conn:
                self.conn.commit()
            
            logger.info("=" * 50)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
        finally:
            if self.conn:
                self.conn.close()
