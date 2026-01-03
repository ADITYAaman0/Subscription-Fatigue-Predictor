# Subscription Fatigue Predictor - Deployment Guide

## Production Deployment

### Prerequisites
- Python 3.10+
- PostgreSQL or SQLite for database
- 4GB+ RAM
- Virtual environment

### Installation

```bash
# Clone and setup
git clone <repo>
cd subscription-fatigue-predictor
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Database Setup

```bash
# Initialize database
python -c "from src.data.database.models import create_database; create_database()"

# Load initial data
python main.py
```

### Streamlit Deployment

#### Option 1: Streamlit Cloud

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect GitHub repo
4. Deploy

#### Option 2: Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY data/ data/

EXPOSE 8501
CMD ["streamlit", "run", "src/visualization/dashboard.py"]
```

Build and run:
```bash
docker build -t spc .
docker run -p 8501:8501 spc
```

#### Option 3: Heroku

```bash
# Create app
heroku create your-app-name

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# View logs
heroku logs --tail
```

### AWS EC2 Deployment

```bash
# SSH into EC2 instance
ssh -i key.pem ec2-user@instance-ip

# Install dependencies
sudo yum install python3 python3-pip
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run streamlit
streamlit run src/visualization/dashboard.py --server.port 80 &
```

### Performance Optimization

1. **Data Caching**
   ```python
   @st.cache_data
   def load_data():
       # Cached data loading
       pass
   ```

2. **Model Caching**
   ```python
   @st.cache_resource
   def get_model():
       return load_trained_model()
   ```

3. **Database Indexing**
   ```sql
   CREATE INDEX idx_pricing_date ON pricing_history(effective_date);
   CREATE INDEX idx_metrics_date ON subscriber_metrics(date);
   ```

4. **Async Data Collection**
   - Use APScheduler for background updates
   - Schedule daily data refresh at off-peak hours

### Monitoring

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### Environment Variables

Create `.env` file:
```
DATABASE_URL=postgresql://user:password@localhost/spc
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
NEWS_API_KEY=your_api_key
LOG_LEVEL=INFO
```

### Health Checks

```python
@app.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': datetime.now()}
```

### Backup & Recovery

```bash
# Backup database
python -m sqlite3 subscription_fatigue.db ".backup backup.db"

# Restore from backup
python -m sqlite3 subscription_fatigue.db ".restore backup.db"
```

---

Last updated: December 2025
