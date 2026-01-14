import os
import json
import hashlib
import logging
import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from src.utils.config import DATABASE_PATH, PROJECT_ROOT
from src.data.collectors.kaggle_collector import KaggleDataCollector
from src.data.collectors.news_scraper import NewsWebScraper

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Orchestrates all data collection and storage.
    Extended with advanced analytics sample data generation.
    """
    
    def __init__(self, db_path=None):
        self.db_path = db_path if db_path else str(DATABASE_PATH)
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self.conn = None
    
    def initialize_database(self):
        """Initialize core database schema and extensions."""
        if self.conn is None:
            return
        
        try:
            # Import core schema from schema.py (formerly schema.sql)
            from src.data.database.schema import SCHEMA
            self.conn.executescript(SCHEMA)
            logger.info("Core schema initialized successfully")
            
            # Also initialize extensions
            self.initialize_schema()
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def initialize_schema(self):
        """Initialize database schema including new analytics tables."""
        if self.conn is None:
            return
        
        try:
            # Now that we renamed files, this should work
            from src.data.database.schema_extensions import SCHEMA_EXTENSIONS
            self.conn.executescript(SCHEMA_EXTENSIONS)
            logger.info("Schema extensions applied successfully")
        except Exception as e:
            logger.warning(f"Could not apply schema extensions: {e}")
    
    def create_sample_companies_data(self):
        """Generate sample companies data."""
        logger.info("Generating sample companies data...")
        companies = [
            {'company_id': 1, 'name': 'Netflix', 'sector': 'Streaming', 'country': 'US', 'stock_symbol': 'NFLX'},
            {'company_id': 2, 'name': 'Spotify', 'sector': 'Music', 'country': 'US', 'stock_symbol': 'SPOT'},
            {'company_id': 3, 'name': 'Disney Plus', 'sector': 'Streaming', 'country': 'US', 'stock_symbol': 'DIS'},
            {'company_id': 4, 'name': 'HBO Max', 'sector': 'Streaming', 'country': 'US', 'stock_symbol': 'WBD'},
            {'company_id': 5, 'name': 'Amazon Prime', 'sector': 'Streaming', 'country': 'US', 'stock_symbol': 'AMZN'}
        ]
        return pd.DataFrame(companies)

    def create_sample_pricing_data(self):
        """Generate sample pricing data."""
        logger.info("Generating sample pricing data...")
        
        dates = pd.date_range('2020-01-01', '2025-12-01', freq='MS')
        data = []
        
        for service_id, service in [(1, 'Netflix'), (2, 'Spotify'), (3, 'Disney Plus'), (4, 'HBO Max'), (5, 'Amazon Prime')]:
            base_price = {1: 10.99, 2: 9.99, 3: 7.99, 4: 14.99, 5: 12.99}[service_id]
            prev_price = base_price
            
            for i, date in enumerate(dates):
                # Apply price increase every 12 months
                price = base_price + (i // 12) * 1.50
                change_pct = ((price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                
                data.append({
                    'company_id': service_id,
                    'effective_date': date.strftime('%Y-%m-%d'),
                    'price': price,
                    'previous_price': prev_price,
                    'change_percentage': change_pct,
                    'currency': 'USD'
                })
                prev_price = price
        
        return pd.DataFrame(data)
    
    def create_sample_metrics_data(self):
        """Generate sample subscriber metrics."""
        logger.info("Generating sample subscriber metrics...")
        
        dates = pd.date_range('2020-01-01', '2025-12-01', freq='MS')
        data = []
        
        for service_id in [1, 2, 3, 4, 5]:
            base_subs = {1: 200_000_000, 2: 150_000_000, 3: 100_000_000, 4: 80_000_000, 5: 200_000_000}[service_id]
            
            for i, date in enumerate(dates):
                growth = np.random.uniform(-0.02, 0.05)
                subs = int(base_subs * (1 + growth * i))
                
                data.append({
                    'company_id': service_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'subscriber_count': subs,
                    'premium_subscribers': int(subs * 0.7),
                    'market_share': np.random.uniform(10, 30),
                    'arpu': np.random.uniform(10, 15),
                    'churn_rate': np.random.uniform(2, 8)
                })
        
        return pd.DataFrame(data)
    
    def create_sample_trends_data(self):
        """Generate sample search trends."""
        logger.info("Generating sample search trends...")
        
        dates = pd.date_range('2020-01-01', '2025-12-01', freq='W')
        data = []
        
        for service_id, keyword in [(1, 'Cancel Netflix'), (2, 'Cancel Spotify'), (3, 'Cancel Disney Plus'), (4, 'Cancel HBO Max'), (5, 'Cancel Amazon Prime')]:
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
    
    def create_sample_content_calendar(self):
        """Generate sample content calendar data for analytics."""
        logger.info("Generating sample content calendar...")
        
        companies = {1: 'Netflix', 2: 'Spotify', 3: 'Disney Plus', 4: 'HBO Max', 5: 'Amazon Prime'}
        content_types = {
            1: ['movie', 'series', 'documentary'],
            2: ['album', 'podcast', 'playlist'],
            3: ['movie', 'series', 'kids'],
            4: ['movie', 'series', 'premium'],
            5: ['movie', 'series', 'original']
        }
        tiers = ['blockbuster', 'premium', 'standard', 'standard', 'filler']
        genres = ['action', 'drama', 'comedy', 'documentary', 'kids', 'reality']
        
        data = []
        base_date = datetime.now().date()
        
        for company_id in companies:
            n_releases = np.random.randint(20, 40)
            
            for i in range(n_releases):
                release_date = base_date + timedelta(days=np.random.randint(1, 180))
                tier = np.random.choice(tiers, p=[0.1, 0.2, 0.4, 0.2, 0.1])
                
                quality = {
                    'blockbuster': np.random.uniform(8, 10),
                    'premium': np.random.uniform(6, 8),
                    'standard': np.random.uniform(4, 7),
                    'filler': np.random.uniform(2, 5)
                }[tier]
                
                data.append({
                    'company_id': company_id,
                    'release_date': release_date.strftime('%Y-%m-%d'),
                    'title': f"{companies[company_id]} Original #{i+1}",
                    'content_type': np.random.choice(content_types[company_id]),
                    'tier': tier,
                    'quality_score': round(quality, 1),
                    'expected_engagement': round(np.random.uniform(0.5, 2.0), 2),
                    'genre': np.random.choice(genres),
                    'is_exclusive': int(np.random.choice([True, False], p=[0.7, 0.3]))
                })
        
        return pd.DataFrame(data)
    

    

    
    def store_to_database(self, table_name, df, source_type: str = 'synthetic', 
                          source_identifier: str = None, justification: str = None):
        """Store DataFrame in SQLite database with provenance tracking."""
        if self.conn is None:
            logger.warning("Database connection not available")
            return
        
        try:
            df.to_sql(table_name, self.conn, if_exists='replace', index=False)
            self.conn.commit()
            logger.info(f"Stored {len(df)} records in {table_name}")
            
            # Log provenance
            self.log_provenance(
                table_name=table_name,
                source_type=source_type,
                source_identifier=source_identifier,
                record_count=len(df),
                justification=justification
            )
        except Exception as e:
            logger.error(f"Failed to store in {table_name}: {e}")
    
    def log_provenance(self, table_name: str, source_type: str, 
                       source_identifier: str = None, record_count: int = 0,
                       justification: str = None, metadata: Dict[str, Any] = None):
        """
        Log data provenance for tracking real vs synthetic data usage.
        
        Args:
            table_name: Name of the table being populated
            source_type: 'kaggle', 'web_scrape', or 'synthetic'
            source_identifier: Dataset slug or URL
            record_count: Number of records ingested
            justification: Required explanation when source_type='synthetic'
            metadata: Additional JSON metadata
        """
        if self.conn is None:
            return
        
        try:
            # Calculate checksum of data
            checksum = hashlib.md5(f"{table_name}_{record_count}_{datetime.now()}".encode()).hexdigest()[:16]
            
            self.cursor.execute("""
                INSERT INTO data_provenance 
                (table_name, source_type, source_identifier, record_count, 
                 ingestion_timestamp, version, checksum, justification, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                table_name,
                source_type,
                source_identifier,
                record_count,
                datetime.now().isoformat(),
                '1.0',
                checksum,
                justification,
                json.dumps(metadata) if metadata else None
            ))
            self.conn.commit()
            logger.debug(f"Logged provenance for {table_name}: {source_type}")
        except Exception as e:
            logger.warning(f"Could not log provenance: {e}")
    
    def log_data_quality(self, table_name: str, check_type: str, 
                         passed: bool, failed_records: int = 0, details: str = None):
        """Log data quality check results."""
        if self.conn is None:
            return
        
        try:
            self.cursor.execute("""
                INSERT INTO data_quality_log 
                (table_name, check_type, check_passed, failed_records, details)
                VALUES (?, ?, ?, ?, ?)
            """, (table_name, check_type, passed, failed_records, details))
            self.conn.commit()
        except Exception as e:
            logger.warning(f"Could not log data quality: {e}")
    
    def ingest_news_data(self, days_back: int = 7) -> bool:
        """
        Ingest news data from web scraping (Google News RSS).
        
        Args:
            days_back: Number of days of news to fetch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Fetching news data for last {days_back} days...")
            scraper = NewsWebScraper(rate_limit_seconds=2.0)
            
            news_df = scraper.get_news_with_sentiment(days_back=days_back)
            
            if news_df.empty:
                logger.warning("No news articles fetched")
                return False
            
            # Store to database with provenance
            self.store_to_database(
                'news_articles', 
                news_df,
                source_type='web_scrape',
                source_identifier='google_news_rss'
            )
            
            # Log quality check
            null_count = news_df['title'].isnull().sum()
            self.log_data_quality(
                'news_articles', 
                'null_check', 
                passed=(null_count == 0),
                failed_records=null_count,
                details='Checked for null titles'
            )
            
            logger.info(f"Successfully ingested {len(news_df)} news articles")
            return True
            
        except Exception as e:
            logger.warning(f"News ingestion failed: {e}")
            return False
    
    def ingest_real_world_data(self) -> Dict[str, bool]:
        """
        Fetches and stores ALL real-world data from Kaggle with provenance tracking.
        Dynamically iterates over all datasets in KaggleDataCollector.DATASETS.
        Recursively searches for CSVs and maps core datasets to standard tables.
        
        Returns:
            Dictionary mapping dataset names to success status
        """
        results = {}
        
        # Mapping specific datasets to core tables expected by the application
        CORE_MAPPINGS = {
            'streaming_prices': 'real_pricing_history',
            'telco_churn': 'real_world_churn_data',
            'netflix_subscribers': 'real_netflix_subscribers',
            'global_streaming': 'real_global_streaming'
        }
        
        try:
            collector = KaggleDataCollector()
            
            # Iterate over ALL datasets in the collector
            for dataset_name, dataset_slug in collector.DATASETS.items():
                try:
                    logger.info(f"Attempting to download: {dataset_name} ({dataset_slug})")
                    
                    # Download the dataset
                    dataset_path = collector.download_dataset(dataset_slug)
                    
                    if dataset_path and os.path.exists(dataset_path):
                        # Find all CSV files recursively
                        csv_files = []
                        for root, dirs, files in os.walk(dataset_path):
                            for file in files:
                                if file.lower().endswith('.csv'):
                                    csv_files.append(os.path.join(root, file))
                        
                        if csv_files:
                            # Load and store each CSV file
                            for csv_path in csv_files:
                                try:
                                    logger.info(f"Processing {csv_path}...")
                                    df = pd.read_csv(csv_path)
                                    
                                    # 1. Generic Ingestion: Create table name from dataset and file
                                    clean_filename = os.path.basename(csv_path).replace('.csv', '').replace('-', '_').replace(' ', '_')
                                    table_name_generic = f"kaggle_{dataset_name}_{clean_filename}"
                                    table_name_generic = table_name_generic[:60]  # SQLite limit
                                    
                                    self.store_to_database(
                                        table_name_generic, df,
                                        source_type='kaggle',
                                        source_identifier=f"{dataset_slug}/{clean_filename}",
                                        justification=None
                                    )
                                    
                                    # 2. Specific Mapping: Check if this is a core dataset
                                    if dataset_name in CORE_MAPPINGS:
                                        target_table = CORE_MAPPINGS[dataset_name]
                                        logger.info(f"Mapping {dataset_name} to core table {target_table}")
                                        self.store_to_database(
                                            target_table, df,
                                            source_type='kaggle',
                                            source_identifier=dataset_slug,
                                            justification="Core real-world dataset mapping"
                                        )
                                    
                                    results[f"{dataset_name}_{clean_filename}"] = True
                                    logger.info(f"✓ Ingested {dataset_name}/{clean_filename}: {len(df)} records")
                                    
                                except Exception as e:
                                    logger.warning(f"Failed to read CSV {csv_path}: {e}")
                                    results[f"{dataset_name}_{os.path.basename(csv_path)}"] = False
                        else:
                            logger.warning(f"No CSV files found in {dataset_path} (recursively)")
                            results[dataset_name] = False
                    else:
                        logger.warning(f"Download failed for {dataset_name}")
                        results[dataset_name] = False
                        
                except Exception as e:
                    logger.warning(f"Dataset {dataset_name} ingestion failed: {e}")
                    results[dataset_name] = False
            
            success_count = sum(1 for v in results.values() if v)
            total_count = len(results)
            logger.info(f"Successfully ingested {success_count} files from Kaggle datasets")
            
        except Exception as e:
            logger.warning(f"Kaggle collector initialization failed: {e}")
        
        return results
    
    def ingest_newsapi_data(self, days_back: int = 7) -> bool:
        """
        Ingest news data from NewsAPI (if API key is available).
        Falls back to web scraping if NewsAPI fails.
        
        Args:
            days_back: Number of days of news to fetch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try NewsAPI first (better quality)
            from src.data.collectors.newsapi_collector import NewsAPICollector
            
            logger.info(f"Fetching news via NewsAPI for last {days_back} days...")
            collector = NewsAPICollector()
            news_df = collector.get_news_with_sentiment(days_back=days_back)
            
            if not news_df.empty:
                self.store_to_database(
                    'news_articles', news_df,
                    source_type='newsapi',
                    source_identifier='newsapi.org'
                )
                logger.info(f"✓ Ingested {len(news_df)} articles from NewsAPI")
                return True
                
        except Exception as e:
            logger.warning(f"NewsAPI failed: {e}. Falling back to web scraping...")
        
        # Fallback to web scraping
        return self.ingest_news_data(days_back)


    def ingest_google_trends_data(self) -> bool:
        """
        Ingest real search trend data from Google Trends.
        Targeting 'cancel [service]' keywords for churn signals.
        """
        try:
            from src.data.collectors.google_trends_collector import GoogleTrendsCollector
            
            logger.info("Initializing Google Trends Collector...")
            collector = GoogleTrendsCollector()
            
            trends_df = collector.get_weekly_interest()
            
            if not trends_df.empty:
                self.store_to_database(
                    'search_trends', trends_df,
                    source_type='google_trends',
                    source_identifier='pytrends_api',
                    justification=None
                )
                logger.info(f"✓ Ingested {len(trends_df)} search trend records")
                return True
            else:
                logger.warning("Google Trends returned no data (possible rate limit or connection issue)")
                return False
                
        except Exception as e:
            logger.warning(f"Google Trends ingestion failed: {e}")
            return False

    def run_full_pipeline(self, skip_synthetic_if_real: bool = True):
        """
        Execute complete data collection pipeline.
        Prioritizes real data from Kaggle/web scraping, falls back to synthetic with justification.
        
        Args:
            skip_synthetic_if_real: If True, skip synthetic data generation when real data is available
        """
        logger.info("=" * 50)
        logger.info("STARTING DATA COLLECTION PIPELINE (Real Data Priority)")
        logger.info("=" * 50)
        
        try:
            # 1. Initialize Database & Schema
            self.initialize_database()
            
            # 2. PRIORITY: Real Data Ingestion from Kaggle
            logger.info("\n--- Phase 1: Real Data Ingestion (Kaggle) ---")
            kaggle_results = self.ingest_real_world_data()
            
            # 3. PRIORITY: News Data from NewsAPI (with web scraping fallback)
            logger.info("\n--- Phase 2: News Data Ingestion (NewsAPI + Scraping) ---")
            news_success = self.ingest_newsapi_data(days_back=7)
            
            # 3.5. NEW: Google Trends Data (Real Search Volume)
            logger.info("\n--- Phase 2.5: Google Trends Data Ingestion ---")
            trends_success = self.ingest_google_trends_data()
            
            # 4. Companies data (always synthetic - reference data)
            logger.info("\n--- Phase 3: Reference Data ---")
            companies_df = self.create_sample_companies_data()
            self.store_to_database(
                'companies', companies_df,
                source_type='synthetic',
                source_identifier='internal_reference',
                justification='Reference data for company mappings - no external source available'
            )
            
            # 5. FALLBACK: Synthetic data only if real data unavailable
            logger.info("\n--- Phase 4: Synthetic Fallbacks (if needed) ---")
            
            if not kaggle_results.get('streaming_prices', False) or not skip_synthetic_if_real:
                pricing_df = self.create_sample_pricing_data()
                self.store_to_database(
                    'pricing_history', pricing_df,
                    source_type='synthetic',
                    source_identifier='generated',
                    justification='Real Kaggle streaming prices unavailable - API auth failed or dataset not accessible'
                )
            else:
                logger.info("Skipping synthetic pricing - real data available")
            
            if not kaggle_results.get('netflix_subscribers', False) or not skip_synthetic_if_real:
                metrics_df = self.create_sample_metrics_data()
                self.store_to_database(
                    'subscriber_metrics', metrics_df,
                    source_type='synthetic',
                    source_identifier='generated',
                    justification='Real Kaggle subscriber metrics unavailable'
                )
            else:
                logger.info("Skipping synthetic metrics - real data available")
            
            # Use Google Trends if available, otherwise fallback to synthetic
            if not trends_success or not skip_synthetic_if_real:
                trends_df = self.create_sample_trends_data()
                self.store_to_database(
                    'search_trends', trends_df,
                    source_type='synthetic',
                    source_identifier='generated',
                    justification='Google Trends data unavailable - using synthetic search trends'
                )
            else:
                logger.info("Skipping synthetic trends - Google Trends data available")
            
            # Content calendar (always synthetic for now)
            content_df = self.create_sample_content_calendar()
            self.store_to_database(
                'content_calendar', content_df,
                source_type='synthetic',
                source_identifier='generated',
                justification='No public dataset for content calendars - using projected release data'
            )
            
            if self.conn:
                self.conn.commit()
            
            # Summary
            logger.info("\n" + "=" * 50)
            logger.info("PIPELINE COMPLETE - Data Sources Summary:")
            logger.info(f"  Kaggle datasets: {sum(kaggle_results.values())}/4 ingested")
            logger.info(f"  News articles: {'SUCCESS' if news_success else 'FALLBACK TO SYNTHETIC'}")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if self.conn:
                self.conn.close()
    
    def get_data_provenance_summary(self) -> pd.DataFrame:
        """Get summary of data provenance for dashboard display."""
        if self.conn is None:
            return pd.DataFrame()
        
        try:
            query = """
                SELECT 
                    table_name,
                    source_type,
                    source_identifier,
                    record_count,
                    ingestion_timestamp,
                    justification
                FROM data_provenance
                ORDER BY ingestion_timestamp DESC
            """
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            logger.warning(f"Could not fetch provenance summary: {e}")
            return pd.DataFrame()
if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    pipeline.run_full_pipeline()
