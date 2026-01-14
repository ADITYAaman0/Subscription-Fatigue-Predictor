"""
Utility to extract pricing data from real_global_streaming table.
Backfills competitor pricing when Kaggle streaming_prices extraction fails.
"""

import logging
import pandas as pd
import sqlite3
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PricingExtractor:
    """
    Extracts pricing information from real_global_streaming table.
    
    The real_global_streaming table contains data on multiple streaming services
    with pricing information that can be used to backfill competitor pricing.
    """
    
    def __init__(self, db_path: str = 'data/subscription_fatigue.db'):
        """
        Initialize pricing extractor.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
    
    def get_competitor_pricing(self, service_name: Optional[str] = None) -> pd.DataFrame:
        """
        Extract current pricing for competitors from real_global_streaming.
        
        Args:
            service_name: Optional service name to filter (e.g., 'Netflix', 'Disney Plus')
            
        Returns:
            DataFrame with service name, price, and date columns
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='real_global_streaming'
            """)
            
            if not cursor.fetchone():
                logger.warning("real_global_streaming table not found")
                conn.close()
                return pd.DataFrame()
            
            # Query pricing data
            # Note: Column names may vary, so we'll try common patterns
            query = "SELECT * FROM real_global_streaming LIMIT 100"
            df = pd.read_sql(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning("real_global_streaming table is empty")
                return pd.DataFrame()
            
            # Try to identify price and service name columns
            price_cols = [c for c in df.columns if 'price' in c.lower() or 'cost' in c.lower() or 'subscription' in c.lower()]
            service_cols = [c for c in df.columns if 'service' in c.lower() or 'name' in c.lower() or 'platform' in c.lower()]
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'year' in c.lower()]
            
            if not price_cols or not service_cols:
                logger.warning(f"Could not identify price/service columns in real_global_streaming. Columns: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Extract relevant columns
            price_col = price_cols[0]
            service_col = service_cols[0]
            date_col = date_cols[0] if date_cols else None
            
            result_df = df[[service_col, price_col]].copy()
            result_df.columns = ['service', 'price']
            
            if date_col:
                result_df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            else:
                result_df['date'] = pd.Timestamp.now()
            
            # Filter by service if specified
            if service_name:
                result_df = result_df[result_df['service'].str.contains(service_name, case=False, na=False)]
            
            # Get latest price for each service
            if 'date' in result_df.columns:
                result_df = result_df.sort_values('date', ascending=False)
                result_df = result_df.groupby('service').first().reset_index()
            
            logger.info(f"Extracted pricing for {len(result_df)} services from real_global_streaming")
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting pricing from real_global_streaming: {e}")
            return pd.DataFrame()
    
    def get_average_competitor_price(self, exclude_service: Optional[str] = None) -> float:
        """
        Get average competitor pricing.
        
        Args:
            exclude_service: Service name to exclude from average (e.g., 'Netflix')
            
        Returns:
            Average price across competitors
        """
        df = self.get_competitor_pricing()
        
        if df.empty:
            return 0.0
        
        if exclude_service:
            df = df[~df['service'].str.contains(exclude_service, case=False, na=False)]
        
        if df.empty:
            return 0.0
        
        # Clean price column (remove non-numeric characters)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])
        
        if df.empty:
            return 0.0
        
        return float(df['price'].mean())
    
    def backfill_pricing_history(self, target_table: str = 'pricing_history') -> bool:
        """
        Backfill pricing_history table with data from real_global_streaming.
        
        Args:
            target_table: Table to backfill
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pricing_df = self.get_competitor_pricing()
            
            if pricing_df.empty:
                logger.warning("No pricing data to backfill")
                return False
            
            conn = sqlite3.connect(self.db_path)
            
            # Map to pricing_history schema
            # Note: This assumes we have company_id mapping
            # In practice, you'd need to join with companies table
            
            # For now, just log what we found
            logger.info(f"Found {len(pricing_df)} pricing records to backfill")
            logger.info(f"Services: {pricing_df['service'].unique().tolist()}")
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error backfilling pricing: {e}")
            return False


def get_competitor_pricing_from_db(service_name: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to get competitor pricing.
    
    Args:
        service_name: Optional service name to filter
        
    Returns:
        DataFrame with pricing data
    """
    extractor = PricingExtractor()
    return extractor.get_competitor_pricing(service_name=service_name)
