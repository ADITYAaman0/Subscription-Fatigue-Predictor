"""
# 📅 Content Value Modeling Engine
# Advanced logic for quantifying the retention impact of content libraries.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta

@dataclass
class ContentRelease:
    """Data class for a content release."""
    title: str
    release_date: datetime
    tier: str  # 'blockbuster', 'premium', 'standard', 'filler'
    quality_score: float  # 0-10
    genre: str
    is_exclusive: bool
    expected_engagement: float

@dataclass
class ContentScore:
    """Aggregated content value score."""
    content_score: float
    retention_boost_pct: float
    churn_adjustment: float
    release_count: int
    top_releases: List[Dict]

class ContentValueModel:
    """
    Advanced model to quantify the value of upcoming content releases.
    Uses time-decay, tier weighting, and exclusivity multipliers.
    """
    
    def __init__(self, content_df: pd.DataFrame):
        self.content_df = content_df
        self.content_df['release_date'] = pd.to_datetime(self.content_df['release_date'])
        
        # Advanced weighting configuration
        self.tier_weights = {
            'blockbuster': 10.0,
            'premium': 6.0,
            'standard': 3.0,
            'filler': 1.0
        }
        self.exclusivity_multiplier = 1.5
        self.decay_factor = 0.05  # Daily decay rate for past content (not used for future yet)
    
    def calculate_content_value_score(self, company_id: int, lookahead_days: int = 30) -> ContentScore:
        """
        Calculate the aggregate content value score for the next N days.
        """
        now = datetime.now()
        target_date = now + timedelta(days=lookahead_days)
        
        # Filter relevant content
        mask = (
            (self.content_df['company_id'] == company_id) & 
            (self.content_df['release_date'] >= now) & 
            (self.content_df['release_date'] <= target_date)
        )
        upcoming = self.content_df[mask].copy()
        
        if upcoming.empty:
            return ContentScore(0, 0, 1.0, 0, [])
        
        # Advanced Scoring Logic
        scores = []
        for _, row in upcoming.iterrows():
            # Base score from tier
            base_score = self.tier_weights.get(row['tier'], 1.0)
            
            # Quality adjustment
            quality_mult = row['quality_score'] / 5.0  # Normalize around 5.0
            
            # Exclusivity
            exc_mult = self.exclusivity_multiplier if row.get('is_exclusive', False) else 1.0
            
            # Time urgency (closer releases scored higher? or lower? Usually immediate value is higher)
            days_until = (row['release_date'] - now).days
            time_mult = 1.0 / (1.0 + 0.1 * max(0, days_until)) # Simple hyperbolic decay
            
            final_item_score = base_score * quality_mult * exc_mult * time_mult
            scores.append(final_item_score)
            
        upcoming['score'] = scores
        
        total_score = sum(scores)
        
        # Retention Boost Calculation
        # Maps total score to a churn reduction percentage using a sigmoid-like curve
        # Cap boost at 30% for massive content drops
        retention_boost = 30 * (1 - np.exp(-total_score / 50)) 
        
        # Churn adjustment factor (1.0 = no change, 0.8 = 20% reduction)
        churn_adjustment = max(0.7, 1.0 - (retention_boost / 100))
        
        # Get top releases
        top_releases = upcoming.sort_values('score', ascending=False).head(5)
        top_list = []
        for _, row in top_releases.iterrows():
            top_list.append({
                'title': row['title'],
                'release_date': row['release_date'].strftime('%Y-%m-%d'),
                'tier': row['tier'],
                'score': row['score']
            })
            
        return ContentScore(
            content_score=total_score,
            retention_boost_pct=round(retention_boost, 2),
            churn_adjustment=round(churn_adjustment, 3),
            release_count=len(upcoming),
            top_releases=top_list
        )
    
    def get_churn_features(self, company_id: int) -> Dict[str, float]:
        """Get features for churn prediction model."""
        score_30 = self.calculate_content_value_score(company_id, 30)
        score_90 = self.calculate_content_value_score(company_id, 90)
        
        return {
            'content_score_30d': score_30.content_score,
            'content_score_90d': score_90.content_score,
            'upcoming_releases_30d': score_30.release_count,
            'content_churn_adjustment': score_30.churn_adjustment
        }

def create_sample_content_data() -> pd.DataFrame:
    """Generate sample content calendar data."""
    companies = {1: 'Netflix', 2: 'Spotify', 3: 'Disney Plus', 4: 'HBO Max', 5: 'Amazon Prime'}
    tiers = ['blockbuster', 'premium', 'standard', 'filler']
    content_types = {1: 'series', 2: 'album', 3: 'movie', 4: 'series', 5: 'movie'}
    
    data = []
    base_date = datetime.now()
    
    # Generate 50 items per company
    for company_id, name in companies.items():
        for i in range(50):
            # Random date within next 180 days
            days_offset = np.random.randint(0, 180)
            date = base_date + timedelta(days=days_offset)
            
            tier = np.random.choice(tiers, p=[0.1, 0.2, 0.4, 0.3])
            quality = np.random.normal(7, 1.5) if tier in ['blockbuster', 'premium'] else np.random.normal(5, 2)
            quality = np.clip(quality, 1, 10)
            
            is_exclusive = np.random.random() > 0.3
            
            data.append({
                'company_id': company_id,
                'title': f"{name} {content_types[company_id].title()} {i+1}",
                'release_date': date,
                'tier': tier,
                'quality_score': quality,
                'genre': 'General',
                'is_exclusive': is_exclusive,
                'expected_engagement': quality * (1.5 if is_exclusive else 1.0)
            })
            
    return pd.DataFrame(data)
