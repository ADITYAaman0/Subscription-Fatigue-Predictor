import os
import logging
import pandas as pd
import pandas as pd
from src.utils.config import KAGGLE_USERNAME, KAGGLE_KEY, DATA_DIR

logger = logging.getLogger(__name__)

class KaggleDataCollector:
    """
    Handles fetching real-world datasets using the Kaggle API.
    Supports streaming service pricing, subscriber data, and churn datasets.
    """
    
    # Dataset slugs for relevant datasets - EXPANDED for maximum real data coverage
    DATASETS = {
        # === CHURN & RETENTION DATASETS ===
        'telco_churn': 'blastchar/telco-customer-churn',
        'netflix_churn': 'shivamb/netflix-user-engagements-dataset',  # Netflix churn & engagement
        'music_streaming_churn': 'jsaguiar/predict-customer-churn-dataset',  # Music streaming churn
        'customer_retention': 'akashkothare/kkbox-churn-prediction-challenge',  # KKBOX music churn
        'subscription_churn': 'saikatkumardey/customer-churn-prediction',  # General subscription churn
        
        # === STREAMING SERVICE DATASETS ===
        'streaming_prices': 'justinrmiller/streaming-service-price-history',
        'netflix_subscribers': 'pariaagharabi/netflix-customer-subscription',
        'global_streaming': 'octopusteam/global-streaming-services-dataset',
        'video_subscriptions': 'muhammadehsan02/streaming-video-subscriptions-datasets',
        
        # === ADDITIONAL STREAMING/MUSIC DATASETS ===
        'spotify_data': 'nelgiriyewithana/top-spotify-songs-2023',  # Spotify engagement
        'netflix_titles': 'shivamb/netflix-shows',  # Netflix content catalog
        'amazon_prime': 'shivamb/amazon-prime-movies-and-tv-shows',  # Prime content
        'disney_plus': 'shivamb/disney-movies-and-tv-shows',  # Disney+ content
        
        # === CUSTOMER BEHAVIOR DATASETS ===
        'ecommerce_behavior': 'mkechinov/ecommerce-behavior-data-from-multi-category-store',
        'subscription_data': 'samuelcortinhas/subscription-data',  # Subscription patterns
    }
    
    def __init__(self):
        self._api = None
        
    @property
    def api(self):
        """Lazy initialization of the Kaggle API."""
        if self._api is None:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                self._api = KaggleApi()
                self._authenticate()
            except Exception as e:
                logger.error(f"Could not initialize Kaggle API: {e}")
                # We don't raise here to allow the class to be instantiated,
                # but individual download methods will fail gracefully.
                pass
        return self._api
        
    def _authenticate(self):
        """Authenticates with Kaggle using environment credentials."""
        try:
            # Set config dir to current directory so it picks up kaggle.json if present
            os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()
            
            os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
            os.environ['KAGGLE_KEY'] = KAGGLE_KEY
            self.api.authenticate()
            logger.info("Successfully authenticated with Kaggle API")
        except Exception as e:
            logger.error(f"Failed to authenticate with Kaggle: {e}")
            raise

    def download_dataset(self, dataset_slug, target_dir=None):
        """
        Downloads a specific dataset from Kaggle.
        
        Args:
            dataset_slug (str): Slug of the dataset
            target_dir (str): Directory to save the dataset
        """
        if target_dir is None:
            target_dir = os.path.join(DATA_DIR, 'raw', dataset_slug.split('/')[-1])
            
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            logger.info(f"Downloading dataset {dataset_slug} to {target_dir}...")
            self.api.dataset_download_files(dataset_slug, path=target_dir, unzip=True)
            logger.info(f"Successfully downloaded and unzipped {dataset_slug}")
            return target_dir
        except Exception as e:
            logger.warning(f"Kaggle API Client failed for {dataset_slug}: {e}")
            logger.info("Attempting Direct Download Fallback using requests...")
            return self._download_direct_fallback(dataset_slug, target_dir)

    def _download_direct_fallback(self, dataset_slug, target_dir):
        """
        Fallback method using direct HTTP requests with Basic Auth.
        Often bypasses client-side 403 errors.
        """
        import requests, zipfile, io
        
        try:
            owner, dataset = dataset_slug.split('/')
            url = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset}"
            auth = (KAGGLE_USERNAME, KAGGLE_KEY)
            
            logger.info(f"Direct downloading from {url}...")
            response = requests.get(url, auth=auth, stream=True)
            
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    z.extractall(target_dir)
                logger.info(f"✓ Direct download successful for {dataset_slug}")
                return target_dir
            else:
                logger.error(f"Direct download failed: Status {response.status_code} - {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"Direct download fallback exception: {e}")
            return None

    def get_telco_churn_data(self):
        """Download Telco Customer Churn dataset."""
        return self.download_dataset(self.DATASETS['telco_churn'])
    
    def get_streaming_price_history(self):
        """
        Download Streaming Service Price History dataset.
        Contains price changes since 2011 for Netflix, Disney+, HBO Max, etc.
        """
        return self.download_dataset(self.DATASETS['streaming_prices'])
    
    def get_netflix_subscribers(self):
        """
        Download Netflix Customer Subscription dataset.
        Contains quarterly subscriber counts from 2013-2023.
        """
        return self.download_dataset(self.DATASETS['netflix_subscribers'])
    
    def get_global_streaming_data(self):
        """
        Download Global Streaming Services Dataset.
        Contains data on 78 paid and 35 free streaming services with:
        - Subscriber metrics
        - Pricing information
        - 5-year historical growth (2020-2024)
        """
        return self.download_dataset(self.DATASETS['global_streaming'])
    
    def get_video_subscriptions(self):
        """
        Download Streaming Video Subscriptions dataset.
        Contains individual subscription records with costs and billing.
        """
        return self.download_dataset(self.DATASETS['video_subscriptions'])
    
    def download_all_datasets(self):
        """Download all relevant datasets for the project."""
        results = {}
        for name, slug in self.DATASETS.items():
            logger.info(f"Downloading {name}...")
            results[name] = self.download_dataset(slug)
        return results
    
    def load_streaming_prices_as_df(self):
        """Load streaming price history as a pandas DataFrame."""
        path = self.get_streaming_price_history()
        if path:
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    return pd.read_csv(os.path.join(path, file))
        return None
    
    def load_netflix_subscribers_as_df(self):
        """Load Netflix subscriber data as a pandas DataFrame."""
        path = self.get_netflix_subscribers()
        if path:
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    return pd.read_csv(os.path.join(path, file))
        return None
