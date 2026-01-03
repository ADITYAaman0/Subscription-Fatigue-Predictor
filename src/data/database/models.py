"""
Database models using SQLAlchemy ORM.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()


class Company(Base):
    __tablename__ = 'companies'
    
    company_id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    sector = Column(String)
    stock_symbol = Column(String)
    country = Column(String)
    
    # Relationships
    pricing = relationship("PricingHistory", back_populates="company")
    metrics = relationship("SubscriberMetrics", back_populates="company")


class PricingHistory(Base):
    __tablename__ = 'pricing_history'
    
    price_id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('companies.company_id'))
    tier_id = Column(Integer)
    price = Column(Float)
    currency = Column(String, default='USD')
    country = Column(String)
    effective_date = Column(Date)
    previous_price = Column(Float)
    change_percentage = Column(Float)
    
    company = relationship("Company", back_populates="pricing")


class SubscriberMetrics(Base):
    __tablename__ = 'subscriber_metrics'
    
    metric_id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('companies.company_id'))
    date = Column(Date)
    subscriber_count = Column(Integer)
    premium_subscribers = Column(Integer)
    market_share = Column(Float)
    arpu = Column(Float)  # Average Revenue Per User
    churn_rate = Column(Float)
    
    company = relationship("Company", back_populates="metrics")


class SearchTrends(Base):
    __tablename__ = 'search_trends'
    
    trend_id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('companies.company_id'))
    search_term = Column(String)
    date = Column(Date)
    search_volume = Column(Integer)  # 0-100 scale
    region = Column(String, default='US')


class ElasticityAnalysis(Base):
    __tablename__ = 'elasticity_analysis'
    
    elasticity_id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('companies.company_id'))
    period_start = Column(Date)
    period_end = Column(Date)
    price_change_percentage = Column(Float)
    quantity_change_percentage = Column(Float)
    elasticity_coefficient = Column(Float)
    revenue_impact_percentage = Column(Float)
    is_elastic = Column(Boolean)


class ChangePoints(Base):
    __tablename__ = 'change_points'
    
    changepoint_id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('companies.company_id'))
    metric_type = Column(String)
    detected_date = Column(Date)
    confidence = Column(Float)  # 0-1
    change_magnitude = Column(Float)
    associated_price_change = Column(Float)
    is_saturation = Column(Boolean)


def create_database(db_url='sqlite:///data/subscription_fatigue.db'):
    """Create database and tables."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine


def get_session(db_url='sqlite:///data/subscription_fatigue.db'):
    """Get a database session."""
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    return Session()
