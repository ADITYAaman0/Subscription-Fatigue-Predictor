"""
Module to download and reassemble chunked datasets from GitHub.
"""

import os
import streamlit as st
import requests
from pathlib import Path
from typing import Optional


def download_and_reassemble_chunks(
    github_repo: str,
    chunk_dir: str,
    output_file: str,
    num_chunks: int = 61,
    chunk_prefix: str = "ecommerce_chunk"
) -> str:
    """
    Download chunks from GitHub and reassemble them into a single file.
    
    Args:
        github_repo: GitHub repository URL (e.g., 'username/repo-name')
        chunk_dir: Directory in repo containing chunks (e.g., 'data/raw/ecommerce-chunks')
        output_file: Path where to save the reassembled file
        num_chunks: Total number of chunks to download
        chunk_prefix: Prefix of chunk filenames (without _00X.zip)
    
    Returns:
        Path to the reassembled file
    """
    
    output_path = Path(output_file)
    
    # If file already exists, return it
    if output_path.exists():
        st.success(f"✅ Dataset already loaded: {output_file}")
        return str(output_path)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    st.info("📥 Downloading dataset chunks from GitHub (this may take a few minutes)...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    raw_github_url = f"https://raw.githubusercontent.com/{github_repo}/main/{chunk_dir}"
    
    try:
        with open(output_path, 'wb') as outfile:
            for i in range(1, num_chunks + 1):
                chunk_filename = f"{chunk_prefix}_{i:03d}.zip"
                chunk_url = f"{raw_github_url}/{chunk_filename}"
                
                status_text.text(f"Downloading chunk {i}/{num_chunks}: {chunk_filename}")
                
                try:
                    response = requests.get(chunk_url, timeout=30)
                    response.raise_for_status()
                    outfile.write(response.content)
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Failed to download {chunk_filename}: {str(e)}")
                    raise
                
                progress = i / num_chunks
                progress_bar.progress(progress)
        
        st.success(f"✅ Dataset reassembled successfully: {output_file}")
        return str(output_path)
    
    except Exception as e:
        st.error(f"❌ Error downloading/reassembling dataset: {str(e)}")
        # Clean up partial file
        if output_path.exists():
            output_path.unlink()
        raise


def load_ecommerce_data(
    github_repo: Optional[str] = None,
    local_path: str = "./data/ecommerce-data.zip"
) -> str:
    """
    Load ecommerce dataset, downloading from GitHub if necessary.
    Uses Streamlit caching to avoid re-downloading on every run.
    
    Args:
        github_repo: GitHub repository in format 'username/repo-name'
        local_path: Local path to store the downloaded file
    
    Returns:
        Path to the dataset file
    """
    
    @st.cache_resource
    def _download():
        if github_repo is None:
            st.error("Please provide github_repo parameter")
            return None
        
        return download_and_reassemble_chunks(
            github_repo=github_repo,
            chunk_dir="data/raw/ecommerce-chunks",
            output_file=local_path,
            num_chunks=61,
            chunk_prefix="ecommerce_chunk"
        )
    
    return _download()


# Example usage in your Streamlit app:
"""
import streamlit as st
from src.utils.data_downloader import load_ecommerce_data

st.set_page_config(page_title="E-commerce Analysis", layout="wide")
st.title("E-commerce Data Analysis")

# Load data from GitHub
data_file = load_ecommerce_data(
    github_repo="your-username/your-repo-name",
    local_path="./data/ecommerce-data.zip"
)

if data_file:
    st.write(f"Data loaded from: {data_file}")
    # Now use the data_file path to extract and load your data
"""
