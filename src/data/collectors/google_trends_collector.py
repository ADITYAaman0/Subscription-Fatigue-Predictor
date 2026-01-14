import pandas as pd
import logging
import time
from pytrends.request import TrendReq
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class GoogleTrendsCollector:
    """
    Collector for Google Trends data to track 'intent to churn' signals.
    Uses pytrends to fetch interest over time for relevant keywords.
    """
    
    KEYWORDS = {
        'Netflix': ['cancel netflix', 'netflix price', 'delete netflix account'],
        'Disney Plus': ['cancel disney plus', 'disney plus price increase', 'unsubscribe disney'],
        'Spotify': ['cancel spotify', 'spotify price', 'switch from spotify'],
        'HBO Max': ['cancel hbo max', 'hbo max price', 'cancel max'],
        'Amazon Prime': ['cancel amazon prime', 'prime video cost']
    }
    
    def __init__(self):
        """Initialize pytrends interface."""
        # Connect to Google
        try:
            self.pytrends = TrendReq(hl='en-US', tz=360)
        except Exception as e:
            logger.error(f"Failed to initialize pytrends: {e}")
            self.pytrends = None
            
    def get_weekly_interest(self, companies=None):
        """
        Fetch weekly interest data for cancellation terms.
        
        Returns:
            pd.DataFrame: Normalized trends data [date, company, search_term, search_volume]
        """
        if not self.pytrends:
            return pd.DataFrame()
            
        all_trends = []
        
        target_companies = companies or self.KEYWORDS.keys()
        
        for company in target_companies:
            terms = self.KEYWORDS.get(company, [])
            for term in terms:
                try:
                    logger.info(f"Fetching Google Trends for: '{term}'")
                    self.pytrends.build_payload([term], cat=0, timeframe='today 12-m')
                    
                    # Interest Over Time
                    data = self.pytrends.interest_over_time()
                    
                    if not data.empty and term in data.columns:
                        df = data.reset_index()[['date', term]].copy()
                        df.columns = ['date', 'search_volume']
                        df['company'] = company
                        df['search_term'] = term
                        all_trends.append(df)
                        
                    # Sleep to avoid rate limits (Google is strict)
                    time.sleep(random.uniform(2, 5))
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch trends for '{term}': {e}")
                    if "429" in str(e):
                        logger.error("Google Trends rate limit reached. Stopping.")
                        break
        
        if all_trends:
            final_df = pd.concat(all_trends, ignore_index=True)
            return final_df.sort_values(['company', 'date'])
        
        return pd.DataFrame()
