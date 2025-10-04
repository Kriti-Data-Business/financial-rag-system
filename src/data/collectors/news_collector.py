# Financial news
# src/data/collectors/news_collector.py
# Financial news collector for Australian sources

import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time

from ...utils.config import Config
from ...utils.logger import get_logger

logger = get_logger(__name__)

class NewsCollector:
    """
    Collects financial news from Australian RSS feeds and news sources.
    Processes articles into structured format for RAG ingestion.
    """
    
    def __init__(self, config: Config):
        """Initialize news collector with configuration."""
        self.config = config
        self.data_dir = Path(config.get('data.news_path', 'data/raw/financial_news'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Australian financial news RSS feeds
        self.news_feeds = {
            'AFR_Companies': 'https://www.afr.com/rss/companies',
            'AFR_Markets': 'https://www.afr.com/rss/markets', 
            'ABC_Finance': 'https://www.abc.net.au/news/feed/51120/rss.xml',
            'RBA_Media': 'https://www.rba.gov.au/media-releases/rss-feed.xml',
            'RBA_Speeches': 'https://www.rba.gov.au/speeches/rss-feed.xml',
            'SBS_Business': 'https://www.sbs.com.au/news/business/feed',
            'Guardian_Australia_Business': 'https://www.theguardian.com/australia-news/business/rss',
            'SmartCompany': 'https://www.smartcompany.com.au/feed/',
            'InvestorDaily': 'https://www.investordaily.com.au/rss'
        }
        
        logger.info("NewsCollector initialized with Australian financial news sources")
    
    def collect_rss_feed(self, feed_name: str, feed_url: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Collect articles from a single RSS feed.
        
        Args:
            feed_name: Name identifier for the feed
            feed_url: RSS feed URL
            days_back: Number of days to look back for articles
            
        Returns:
            List of processed article dictionaries
        """
        articles = []
        
        try:
            logger.info(f"Collecting news from {feed_name}...")
            
            # Parse RSS feed with timeout
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                logger.warning(f"Feed {feed_name} may have parsing issues")
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for entry in feed.entries[:20]:  # Limit to 20 most recent articles
                try:
                    # Parse publication date
                    pub_date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    
                    # Only include recent articles
                    if pub_date >= cutoff_date:
                        article = {
                            'source': feed_name,
                            'title': entry.get('title', 'No Title'),
                            'link': entry.get('link', ''),
                            'published': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                            'summary': entry.get('summary', ''),
                            'content': self._extract_full_content(entry),
                            'tags': self._extract_tags(entry),
                            'category': 'financial_news',
                            'collection_date': datetime.now().isoformat()
                        }
                        
                        # Add content if available
                        if hasattr(entry, 'content') and entry.content:
                            article['content'] = self._clean_html(entry.content[0].value)
                        
                        articles.append(article)
                        
                except Exception as e:
                    logger.error(f"Error processing article from {feed_name}: {e}")
                    continue
            
            logger.info(f"Collected {len(articles)} articles from {feed_name}")
            
        except Exception as e:
            logger.error(f"Error collecting RSS feed {feed_name}: {e}")
        
        return articles
    
    def _extract_full_content(self, entry) -> str:
        """Extract full content from RSS entry."""
        content = ""
        
        # Try different content fields
        if hasattr(entry, 'content') and entry.content:
            content = entry.content[0].value
        elif hasattr(entry, 'summary'):
            content = entry.summary
        elif hasattr(entry, 'description'):
            content = entry.description
        
        return self._clean_html(content) if content else ""
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML tags and formatting from content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text and clean whitespace
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning HTML content: {e}")
            return html_content
    
    def _extract_tags(self, entry) -> List[str]:
        """Extract tags/categories from RSS entry."""
        tags = []
        
        if hasattr(entry, 'tags'):
            tags.extend([tag.term for tag in entry.tags if hasattr(tag, 'term')])
        
        if hasattr(entry, 'category'):
            if isinstance(entry.category, str):
                tags.append(entry.category)
            elif hasattr(entry.category, 'term'):
                tags.append(entry.category.term)
        
        return tags
    
    def collect_all_news(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Collect news from all configured RSS feeds.
        
        Args:
            days_back: Number of days to look back for articles
            
        Returns:
            List of all collected articles
        """
        all_articles = []
        
        for feed_name, feed_url in self.news_feeds.items():
            try:
                articles = self.collect_rss_feed(feed_name, feed_url, days_back)
                all_articles.extend(articles)
                
                # Rate limiting between feeds
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting from {feed_name}: {e}")
                continue
        
        # Remove duplicates based on title and link
        unique_articles = []
        seen = set()
        
        for article in all_articles:
            identifier = (article['title'].lower(), article['link'])
            if identifier not in seen:
                unique_articles.append(article)
                seen.add(identifier)
        
        logger.info(f"Collected {len(unique_articles)} unique articles from {len(self.news_feeds)} sources")
        
        # Save to CSV
        self._save_articles_to_csv(unique_articles)
        
        return unique_articles
    
    def _save_articles_to_csv(self, articles: List[Dict[str, Any]]) -> None:
        """Save articles to CSV file for later processing."""
        try:
            if articles:
                df = pd.DataFrame(articles)
                
                # Save main articles file
                file_path = self.data_dir / 'financial_news_articles.csv'
                df.to_csv(file_path, index=False)
                
                # Save summary file
                summary_df = df.groupby('source').agg({
                    'title': 'count',
                    'published': ['min', 'max']
                }).round(2)
                summary_df.columns = ['article_count', 'earliest_date', 'latest_date']
                summary_df.to_csv(self.data_dir / 'news_collection_summary.csv')
                
                logger.info(f"Saved {len(articles)} articles to {file_path}")
        
        except Exception as e:
            logger.error(f"Error saving articles to CSV: {e}")
    
    def process_for_rag(self, articles: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process news articles into RAG-ready documents.
        
        Args:
            articles: List of articles (loads from file if None)
            
        Returns:
            List of processed documents for RAG ingestion
        """
        if articles is None:
            # Load from saved file
            try:
                articles_file = self.data_dir / 'financial_news_articles.csv'
                if articles_file.exists():
                    df = pd.read_csv(articles_file)
                    articles = df.to_dict('records')
                else:
                    logger.warning("No saved articles found, collecting fresh news...")
                    articles = self.collect_all_news()
            except Exception as e:
                logger.error(f"Error loading articles: {e}")
                return []
        
        processed_documents = []
        
        for article in articles:
            try:
                # Create content for RAG
                content_parts = [
                    f"Australian Financial News - {article['title']}"
                ]
                
                if article.get('summary'):
                    content_parts.append(f"Summary: {article['summary']}")
                
                if article.get('content'):
                    # Limit content length for RAG processing
                    content = article['content'][:2000] + "..." if len(article['content']) > 2000 else article['content']
                    content_parts.append(f"Content: {content}")
                
                if article.get('tags'):
                    content_parts.append(f"Topics: {', '.join(article['tags'])}")
                
                content_parts.append(f"Source: {article['source']}, Published: {article['published']}")
                
                doc = {
                    'content': '\n\n'.join(content_parts),
                    'metadata': {
                        'source': article['source'],
                        'dataset': 'financial_news',
                        'category': 'news_article',
                        'title': article['title'],
                        'published_date': article['published'],
                        'url': article.get('link', ''),
                        'tags': article.get('tags', []),
                        'collection_date': article.get('collection_date', datetime.now().isoformat())
                    }
                }
                
                processed_documents.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing article for RAG: {e}")
                continue
        
        logger.info(f"Processed {len(processed_documents)} news articles for RAG system")
        return processed_documents
    
    def get_latest_market_news(self, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get latest market-relevant news articles.
        
        Args:
            keywords: List of keywords to filter articles
            
        Returns:
            Filtered list of recent market news
        """
        if keywords is None:
            keywords = [
                'asx', 'shares', 'stock', 'market', 'trading', 'investment',
                'rba', 'interest rate', 'inflation', 'economy', 'gold', 'silver',
                'superannuation', 'super', 'property', 'banks', 'mining'
            ]
        
        # Collect recent news (last 3 days)
        recent_articles = self.collect_all_news(days_back=3)
        
        # Filter by keywords
        filtered_articles = []
        for article in recent_articles:
            text_to_search = f"{article['title']} {article.get('summary', '')} {article.get('content', '')}".lower()
            
            if any(keyword.lower() in text_to_search for keyword in keywords):
                filtered_articles.append(article)
        
        # Sort by publication date (newest first)
        filtered_articles.sort(key=lambda x: x['published'], reverse=True)
        
        logger.info(f"Found {len(filtered_articles)} market-relevant news articles")
        return filtered_articles[:10]  # Return top 10 most recent
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of news collection status."""
        try:
            summary = {
                'collector': 'NewsCollector',
                'data_directory': str(self.data_dir),
                'configured_feeds': len(self.news_feeds),
                'feed_sources': list(self.news_feeds.keys())
            }
            
            # Check for existing files
            articles_file = self.data_dir / 'financial_news_articles.csv'
            if articles_file.exists():
                df = pd.read_csv(articles_file)
                summary.update({
                    'total_articles': len(df),
                    'sources_collected': df['source'].nunique(),
                    'date_range': f"{df['published'].min()} to {df['published'].max()}"
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting collection summary: {e}")
            return {'error': str(e)}