#!/usr/bin/env python
"""
Initial setup script for Subscription Fatigue Predictor.
Run this after installing dependencies: python setup.py
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    dirs = [
        'data',
        'logs',
        'notebooks',
        'config',
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def create_database():
    """Initialize database."""
    try:
        from src.data.database.models import create_database
        create_database()
        print("✓ Database initialized")
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")

def generate_sample_data():
    """Generate sample data."""
    try:
        from src.data.collectors.data_ingestion import DataIngestionPipeline
        
        print("\nGenerating sample data...")
        pipeline = DataIngestionPipeline('data/subscription_fatigue.db')
        pipeline.run_full_pipeline()
        print("✓ Sample data generated")
    except Exception as e:
        print(f"✗ Sample data generation failed: {e}")

def main():
    """Run setup."""
    print("=" * 60)
    print("Subscription Fatigue Predictor - Setup")
    print("=" * 60)
    
    print("\n1. Creating directories...")
    create_directories()
    
    print("\n2. Initializing database...")
    create_database()
    
    print("\n3. Generating sample data...")
    generate_sample_data()
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run dashboard: streamlit run src/visualization/dashboard.py")
    print("2. Try examples: python examples.py")
    print("3. Read docs: Check docs/ folder")
    print("\nDashboard will open at: http://localhost:8501")

if __name__ == "__main__":
    main()
