"""
News web scraper for subscription/streaming services.
Scrapes news from Google News RSS feeds as an alternative to NewsAPI.
"""

import os
import re
import logging
import hashlib
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import time

logger = logging.getLogger(__name__)


class NewsWebScraper:
    """
    Scrapes subscription/streaming news from Google News RSS feeds.
    Uses RSS feeds which are publicly accessible and don't require API keys.
    
    Features:
    - Rate limiting to avoid being blocked
    - Retry logic with exponential backoff
    - Response caching
    - Article deduplication by URL hash
    """
    
    GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
    
    # Keywords for subscription fatigue / streaming news
    DEFAULT_KEYWORDS = [
        "Netflix subscription",
        "Netflix price increase",
        "Disney Plus subscription",
        "HBO Max price",
        "streaming service cancel",
        "subscription fatigue",
        "Spotify price increase",
        "streaming churn",
        "cancel streaming subscription",
        "streaming service expensive"
    ]
    
    # User agent to avoid being blocked
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/rss+xml, application/xml, text/xml, */*',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    def __init__(self, cache_dir: Optional[str] = None, rate_limit_seconds: float = 2.0):
        """
        Initialize the news scraper.
        
        Args:
            cache_dir: Directory to cache scraped articles (optional)
            rate_limit_seconds: Minimum seconds between requests
        """
        self.cache_dir = cache_dir
        self.rate_limit_seconds = rate_limit_seconds
        self._last_request_time = 0
        self._cache = {}
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self._last_request_time = time.time()
    
    def _make_request(self, url: str, max_retries: int = 3) -> Optional[str]:
        """
        Make HTTP request with retry logic.
        
        Args:
            url: URL to fetch
            max_retries: Maximum number of retries
            
        Returns:
            Response text or None if failed
        """
        self._rate_limit()
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.HEADERS, timeout=10)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                             f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
        
        logger.error(f"Failed to fetch {url} after {max_retries} attempts")
        return None
    
    def _parse_rss_feed(self, xml_content: str) -> List[Dict]:
        """
        Parse Google News RSS feed XML.
        
        Args:
            xml_content: Raw XML content
            
        Returns:
            List of article dictionaries
        """
        articles = []
        
        try:
            root = ET.fromstring(xml_content)
            channel = root.find('channel')
            
            if channel is None:
                return articles
            
            for item in channel.findall('item'):
                title_elem = item.find('title')
                link_elem = item.find('link')
                pub_date_elem = item.find('pubDate')
                description_elem = item.find('description')
                source_elem = item.find('source')
                
                article = {
                    'title': title_elem.text if title_elem is not None else '',
                    'url': link_elem.text if link_elem is not None else '',
                    'published_at': self._parse_date(pub_date_elem.text) if pub_date_elem is not None else None,
                    'description': self._clean_html(description_elem.text) if description_elem is not None else '',
                    'source_name': source_elem.text if source_elem is not None else 'Unknown'
                }
                
                # Generate unique hash for deduplication
                article['url_hash'] = hashlib.md5(article['url'].encode()).hexdigest()
                
                articles.append(article)
                
        except ET.ParseError as e:
            logger.error(f"Failed to parse RSS feed: {e}")
        
        return articles
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse RFC 822 date format used in RSS feeds."""
        try:
            # Example: "Mon, 13 Jan 2026 10:30:00 GMT"
            return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
        except (ValueError, TypeError):
            try:
                # Try alternative format
                return datetime.strptime(date_str[:25], "%a, %d %b %Y %H:%M:%S")
            except (ValueError, TypeError):
                return None
    
    def _clean_html(self, html_content: str) -> str:
        """Remove HTML tags from content."""
        if not html_content:
            return ''
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    
    def search_news(self, keyword: str, days_back: int = 7) -> List[Dict]:
        """
        Search for news articles matching a keyword.
        
        Args:
            keyword: Search keyword
            days_back: Number of days to look back
            
        Returns:
            List of article dictionaries
        """
        encoded_keyword = quote_plus(keyword)
        
        # Google News RSS URL with search query
        url = f"{self.GOOGLE_NEWS_RSS}?q={encoded_keyword}&hl=en-US&gl=US&ceid=US:en"
        
        # Check cache first
        cache_key = f"{keyword}_{days_back}"
        if cache_key in self._cache:
            cached_time, cached_articles = self._cache[cache_key]
            if time.time() - cached_time < 3600:  # Cache valid for 1 hour
                logger.debug(f"Using cached results for '{keyword}'")
                return cached_articles
        
        logger.info(f"Fetching news for keyword: '{keyword}'")
        xml_content = self._make_request(url)
        
        if not xml_content:
            return []
        
        articles = self._parse_rss_feed(xml_content)
        
        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        filtered_articles = []
        for article in articles:
            if article['published_at'] is None or article['published_at'] >= cutoff_date:
                article['keyword_matched'] = keyword
                filtered_articles.append(article)
        
        # Update cache
        self._cache[cache_key] = (time.time(), filtered_articles)
        
        logger.info(f"Found {len(filtered_articles)} articles for '{keyword}'")
        return filtered_articles
    
    def get_subscription_news(self, 
                               keywords: Optional[List[str]] = None,
                               days_back: int = 7) -> pd.DataFrame:
        """
        Aggregate news from all subscription-related keywords.
        
        Args:
            keywords: List of keywords to search (uses defaults if None)
            days_back: Number of days to look back
            
        Returns:
            DataFrame with deduplicated articles
        """
        if keywords is None:
            keywords = self.DEFAULT_KEYWORDS
        
        all_articles = []
        seen_hashes = set()
        
        for keyword in keywords:
            articles = self.search_news(keyword, days_back)
            
            for article in articles:
                # Deduplicate by URL hash
                if article['url_hash'] not in seen_hashes:
                    seen_hashes.add(article['url_hash'])
                    all_articles.append(article)
        
        if not all_articles:
            logger.warning("No articles found for any keywords")
            return pd.DataFrame(columns=[
                'title', 'url', 'published_at', 'description', 
                'source_name', 'url_hash', 'keyword_matched'
            ])
        
        df = pd.DataFrame(all_articles)
        df = df.sort_values('published_at', ascending=False, na_position='last')
        
        logger.info(f"Total unique articles collected: {len(df)}")
        return df
    
    def analyze_sentiment_simple(self, text: str) -> float:
        """
        Simple rule-based sentiment analysis.
        Returns score between -1 (negative) and 1 (positive).
        
        For production, consider using TextBlob or VADER.
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        positive_words = [
            'growth', 'increase', 'success', 'popular', 'gain', 'profit',
            'record', 'surge', 'win', 'best', 'great', 'love', 'happy'
        ]
        negative_words = [
            'cancel', 'churn', 'loss', 'decline', 'drop', 'expensive',
            'fatigue', 'frustration', 'angry', 'hate', 'worst', 'fail',
            'lose', 'cut', 'problem', 'issue', 'concern'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        return (positive_count - negative_count) / total
    
    def get_news_with_sentiment(self, 
                                 keywords: Optional[List[str]] = None,
                                 days_back: int = 7) -> pd.DataFrame:
        """
        Get news articles with sentiment scores.
        
        Args:
            keywords: List of keywords to search
            days_back: Number of days to look back
            
        Returns:
            DataFrame with articles and sentiment scores
        """
        df = self.get_subscription_news(keywords, days_back)
        
        if df.empty:
            df['sentiment_score'] = []
            return df
        
        # Calculate sentiment for each article
        df['sentiment_score'] = df.apply(
            lambda row: self.analyze_sentiment_simple(
                f"{row['title']} {row['description']}"
            ),
            axis=1
        )
        
        return df


# Convenience function for quick testing
def fetch_latest_streaming_news(days_back: int = 7) -> pd.DataFrame:
    """
    Convenience function to fetch latest streaming news.
    
    Args:
        days_back: Number of days to look back
        
    Returns:
        DataFrame with news articles and sentiment
    """
    scraper = NewsWebScraper(rate_limit_seconds=2.0)
    return scraper.get_news_with_sentiment(days_back=days_back)


if __name__ == "__main__":
    # Test the scraper
    logging.basicConfig(level=logging.INFO)
    
    print("Testing NewsWebScraper...")
    scraper = NewsWebScraper()
    
    # Test single keyword
    articles = scraper.search_news("Netflix subscription", days_back=7)
    print(f"\nFound {len(articles)} articles for 'Netflix subscription'")
    
    if articles:
        print(f"Sample article: {articles[0]['title'][:80]}...")
    
    # Test aggregation with sentiment
    df = scraper.get_news_with_sentiment(days_back=7)
    print(f"\nTotal unique articles: {len(df)}")
    
    if not df.empty:
        print("\nSample data:")
        print(df[['title', 'source_name', 'sentiment_score']].head())
