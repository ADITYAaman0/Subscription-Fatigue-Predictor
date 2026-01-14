"""
NewsAPI collector for fetching subscription and streaming news.
Uses the NewsAPI.org service which provides access to news articles.
"""

import os
import logging
import hashlib
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time

from src.utils.config import NEWS_API_KEY

logger = logging.getLogger(__name__)


class NewsAPICollector:
    """
    Fetches news articles from NewsAPI.org related to streaming and subscriptions.
    
    Features:
    - Rate limiting to stay within API quota
    - Retry logic with exponential backoff
    - Article deduplication by URL
    - Multiple relevant keywords for comprehensive coverage
    """
    
    BASE_URL = "https://newsapi.org/v2"
    
    # Keywords for subscription/streaming news
    DEFAULT_KEYWORDS = [
        "Netflix subscription",
        "Netflix price increase",
        "Disney Plus subscription",
        "Spotify subscription",
        "HBO Max price",
        "streaming service cancel",
        "subscription fatigue",
        "streaming churn",
        "streaming service expensive",
        "cancel streaming subscription"
    ]
    
    def __init__(self, api_key: str = None):
        """
        Initialize the NewsAPI collector.
        
        Args:
            api_key: NewsAPI key (defaults to env variable)
        """
        self.api_key = api_key or NEWS_API_KEY
        if not self.api_key or self.api_key == 'your_api_key':
            raise ValueError("Valid NEWS_API_KEY required. Set in .env file.")
        
        self._requests_today = 0
        self._last_request_date = datetime.now().date()
    
    def _check_rate_limit(self):
        """Reset counter if new day, check if quota exceeded."""
        today = datetime.now().date()
        if today != self._last_request_date:
            self._requests_today = 0
            self._last_request_date = today
        
        # Free tier: 100 requests/day
        if self._requests_today >= 95:  # Leave some buffer
            logger.warning("Approaching NewsAPI daily limit (100 requests)")
            return False
        return True
    
    def _make_request(self, endpoint: str, params: dict, max_retries: int = 3) -> Optional[dict]:
        """
        Make API request with retry logic.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            max_retries: Maximum retry attempts
            
        Returns:
            JSON response or None
        """
        if not self._check_rate_limit():
            logger.warning("Rate limit approaching, skipping request")
            return None
        
        url = f"{self.BASE_URL}/{endpoint}"
        params['apiKey'] = self.api_key
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=10)
                self._requests_today += 1
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    logger.warning("Rate limited, waiting before retry...")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    return None
                    
            except requests.RequestException as e:
                wait_time = 2 ** attempt
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}. Retrying in {wait_time}s")
                time.sleep(wait_time)
        
        return None
    
    def search_articles(self, 
                        query: str, 
                        from_date: Optional[str] = None,
                        to_date: Optional[str] = None,
                        page_size: int = 100) -> List[Dict]:
        """
        Search for news articles.
        
        Args:
            query: Search query
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            page_size: Number of results per page
            
        Returns:
            List of article dictionaries
        """
        if from_date is None:
            # NewsAPI free tier: 30 days historical
            from_date = (datetime.now() - timedelta(days=29)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from': from_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': min(page_size, 100)
        }
        
        if to_date:
            params['to'] = to_date
        
        logger.info(f"Searching NewsAPI for: '{query}'")
        response = self._make_request('everything', params)
        
        if not response or response.get('status') != 'ok':
            return []
        
        return response.get('articles', [])
    
    def get_top_headlines(self, category: str = 'technology') -> List[Dict]:
        """
        Get top headlines in a category.
        
        Args:
            category: News category
            
        Returns:
            List of article dictionaries
        """
        params = {
            'category': category,
            'country': 'us',
            'pageSize': 50
        }
        
        response = self._make_request('top-headlines', params)
        
        if not response or response.get('status') != 'ok':
            return []
        
        return response.get('articles', [])
    
    def get_subscription_news(self, 
                               keywords: Optional[List[str]] = None,
                               days_back: int = 7) -> pd.DataFrame:
        """
        Aggregate news from all subscription-related keywords.
        
        Args:
            keywords: Custom keywords (uses defaults if None)
            days_back: Number of days to look back
            
        Returns:
            DataFrame with deduplicated articles
        """
        if keywords is None:
            keywords = self.DEFAULT_KEYWORDS
        
        from_date = (datetime.now() - timedelta(days=min(days_back, 29))).strftime('%Y-%m-%d')
        
        all_articles = []
        seen_urls = set()
        
        for keyword in keywords:
            articles = self.search_articles(keyword, from_date=from_date)
            
            for article in articles:
                url = article.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    
                    all_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': url,
                        'url_hash': hashlib.md5(url.encode()).hexdigest(),
                        'source_name': article.get('source', {}).get('name', 'Unknown'),
                        'published_at': article.get('publishedAt'),
                        'keyword_matched': keyword,
                        'author': article.get('author', ''),
                        'content': article.get('content', '')
                    })
            
            # Small delay between requests
            time.sleep(0.5)
        
        if not all_articles:
            logger.warning("No articles found from NewsAPI")
            return pd.DataFrame(columns=[
                'title', 'description', 'url', 'url_hash', 'source_name',
                'published_at', 'keyword_matched', 'author', 'content'
            ])
        
        df = pd.DataFrame(all_articles)
        df['published_at'] = pd.to_datetime(df['published_at'])
        df = df.sort_values('published_at', ascending=False)
        
        logger.info(f"NewsAPI: Found {len(df)} unique articles from {len(keywords)} keywords")
        return df
    
    def analyze_sentiment_simple(self, text: str) -> float:
        """Simple rule-based sentiment (-1 to 1)."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        positive = ['growth', 'increase', 'success', 'popular', 'gain', 'profit',
                    'record', 'surge', 'win', 'best', 'great', 'love']
        negative = ['cancel', 'churn', 'loss', 'decline', 'drop', 'expensive',
                    'fatigue', 'frustration', 'hate', 'worst', 'fail', 'problem']
        
        pos_count = sum(1 for w in positive if w in text_lower)
        neg_count = sum(1 for w in negative if w in text_lower)
        
        total = pos_count + neg_count
        return (pos_count - neg_count) / total if total > 0 else 0.0
    
    def get_news_with_sentiment(self, days_back: int = 7) -> pd.DataFrame:
        """Get news with sentiment scores."""
        df = self.get_subscription_news(days_back=days_back)
        
        if df.empty:
            df['sentiment_score'] = []
            return df
        
        df['sentiment_score'] = df.apply(
            lambda row: self.analyze_sentiment_simple(
                f"{row['title']} {row['description']}"
            ),
            axis=1
        )
        
        return df


def fetch_newsapi_articles(days_back: int = 7) -> pd.DataFrame:
    """Convenience function to fetch news articles."""
    try:
        collector = NewsAPICollector()
        return collector.get_news_with_sentiment(days_back=days_back)
    except ValueError as e:
        logger.warning(f"NewsAPI not available: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing NewsAPICollector...")
    try:
        collector = NewsAPICollector()
        df = collector.get_news_with_sentiment(days_back=7)
        print(f"\nFound {len(df)} articles")
        if not df.empty:
            print(df[['title', 'source_name', 'sentiment_score']].head())
    except ValueError as e:
        print(f"Error: {e}")
