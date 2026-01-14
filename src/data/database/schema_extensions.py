"""
Database schema extensions for advanced analytics features.
Content calendars, attribution, experiments, and regulatory scenarios.
"""

SCHEMA_EXTENSIONS = """
-- ============================================================================
-- CONTENT VALUE MODELING
-- ============================================================================

CREATE TABLE IF NOT EXISTS content_calendar (
    content_id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER,
    release_date DATE NOT NULL,
    title TEXT NOT NULL,
    content_type TEXT,  -- 'movie', 'series', 'album', 'podcast', etc.
    tier TEXT DEFAULT 'standard',  -- 'blockbuster', 'premium', 'standard', 'filler'
    quality_score REAL DEFAULT 5.0,
    expected_engagement REAL DEFAULT 1.0,
    genre TEXT,
    is_exclusive BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (company_id) REFERENCES companies(company_id)
);

CREATE INDEX IF NOT EXISTS idx_content_release ON content_calendar(company_id, release_date);

-- ============================================================================
-- NEWS ARTICLES (FROM WEB SCRAPING)
-- ============================================================================

CREATE TABLE IF NOT EXISTS news_articles (
    article_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT,
    title TEXT NOT NULL,
    description TEXT,
    url TEXT UNIQUE,
    url_hash TEXT UNIQUE,
    published_at DATETIME,
    keyword_matched TEXT,
    sentiment_score REAL DEFAULT 0.0,
    ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_news_keyword ON news_articles(keyword_matched);

-- ============================================================================
-- DATA PROVENANCE TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS data_provenance (
    provenance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    source_type TEXT NOT NULL,  -- 'kaggle', 'web_scrape', 'synthetic'
    source_identifier TEXT,     -- dataset slug or URL
    record_count INTEGER,
    ingestion_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    version TEXT,
    checksum TEXT,
    justification TEXT,         -- Required when source_type='synthetic'
    metadata TEXT               -- JSON for additional info
);

CREATE INDEX IF NOT EXISTS idx_provenance_table ON data_provenance(table_name);
CREATE INDEX IF NOT EXISTS idx_provenance_source ON data_provenance(source_type);

-- ============================================================================
-- DATA QUALITY LOGS
-- ============================================================================

CREATE TABLE IF NOT EXISTS data_quality_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    check_type TEXT NOT NULL,   -- 'null_check', 'duplicate_check', 'range_check'
    check_passed BOOLEAN,
    failed_records INTEGER DEFAULT 0,
    details TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_quality_table ON data_quality_log(table_name);
"""




