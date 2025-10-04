# src/data/collectors/metals_collector.py
# Precious Metals Data Collector - Updated with Real APIs

import requests
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from ...utils.config import Config
from ...utils.logger import get_logger

logger = get_logger(__name__)

class MetalsDataCollector:
    """
    Collector for precious metals prices in AUD.
    Uses multiple APIs and data sources for comprehensive coverage.
    """
    
    def __init__(self, config: Config):
        """Initialize metals collector with real API endpoints."""
        self.config = config
        self.data_dir = Path(config.get('data.metals_path', 'data/raw/precious_metals'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Real API endpoints for precious metals
        self.api_endpoints = {
            'goldapi': {
                'base_url': 'https://www.goldapi.io/api',
                'headers': {'x-access-token': config.get('metals.goldapi_key', '')},
                'currencies': ['AUD']
            },
            'metalpriceapi': {
                'base_url': 'https://api.metalpriceapi.com/v1',
                'api_key': config.get('metals.metalpriceapi_key', ''),
                'metals': ['XAU', 'XAG', 'XPT', 'XPD'],  # Gold, Silver, Platinum, Palladium
                'currency': 'AUD'
            },
            'currencyapi': {
                'base_url': 'https://api.currencyapi.com/v3/latest',
                'api_key': config.get('metals.currencyapi_key', ''),
                'base_currency': 'USD'
            }
        }
        
        # Perth Mint and other Australian sources
        self.australian_sources = {
            'perth_mint_csv': 'https://www.perthmint.com/charts/data/gold-price-chart-aud.csv',
            'asx_gold_etfs': ['GOLD.AX', 'QAU.AX', 'PMGOLD.AX'],  # ASX Gold ETFs
            'asx_mining_stocks': ['NST.AX', 'EVN.AX', 'NCM.AX', 'RSG.AX', 'SLR.AX']  # Gold miners
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; Australian Financial RAG Data Collector)',
            'Accept': 'application/json, text/csv'
        }
        
        logger.info("MetalsDataCollector initialized with multiple real APIs")
    
    def collect_current_gold_prices(self) -> List[Dict[str, Any]]:
        """
        Collect current gold prices from multiple sources.
        
        Returns:
            List of gold price records
        """
        logger.info("Collecting current gold prices in AUD...")
        gold_prices = []
        
        # Try GoldAPI first
        gold_price = self._fetch_goldapi_price('XAU', 'AUD')
        if gold_price:
            gold_prices.append(gold_price)
        
        # Try MetalPriceAPI
        metal_price = self._fetch_metalpriceapi_price('XAU', 'AUD')
        if metal_price:
            gold_prices.append(metal_price)
        
        # Try Perth Mint data
        perth_price = self._fetch_perth_mint_gold()
        if perth_price:
            gold_prices.append(perth_price)
        
        # If no API data available, use sample data
        if not gold_prices:
            logger.warning("No live gold price data available, using sample data")
            gold_prices.append(self._get_sample_gold_price())
        
        # Save gold prices
        if gold_prices:
            df = pd.DataFrame(gold_prices)
            file_path = self.data_dir / 'current_gold_prices_aud.csv'
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(gold_prices)} gold price records to {file_path}")
        
        return gold_prices
    
    def collect_current_silver_prices(self) -> List[Dict[str, Any]]:
        """
        Collect current silver prices from multiple sources.
        
        Returns:
            List of silver price records
        """
        logger.info("Collecting current silver prices in AUD...")
        silver_prices = []
        
        # Try GoldAPI for silver
        silver_price = self._fetch_goldapi_price('XAG', 'AUD')
        if silver_price:
            silver_price['metal'] = 'silver'
            silver_prices.append(silver_price)
        
        # Try MetalPriceAPI for silver
        metal_price = self._fetch_metalpriceapi_price('XAG', 'AUD')
        if metal_price:
            metal_price['metal'] = 'silver'
            silver_prices.append(metal_price)
        
        # Fallback to sample data
        if not silver_prices:
            logger.warning("No live silver price data available, using sample data")
            silver_prices.append(self._get_sample_silver_price())
        
        # Save silver prices
        if silver_prices:
            df = pd.DataFrame(silver_prices)
            file_path = self.data_dir / 'current_silver_prices_aud.csv'
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(silver_prices)} silver price records to {file_path}")
        
        return silver_prices
    
    def _fetch_goldapi_price(self, metal: str, currency: str) -> Optional[Dict[str, Any]]:
        """
        Fetch price from GoldAPI.
        
        Args:
            metal: Metal symbol (XAU, XAG, etc.)
            currency: Currency code (AUD, USD, etc.)
            
        Returns:
            Price record or None
        """
        try:
            api_key = self.api_endpoints['goldapi']['headers']['x-access-token']
            if not api_key:
                logger.warning("GoldAPI key not configured")
                return None
            
            url = f"{self.api_endpoints['goldapi']['base_url']}/{metal}/{currency}"
            headers = self.api_endpoints['goldapi']['headers']
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                price_record = {
                    'source': 'GoldAPI',
                    'metal': metal.lower().replace('x', ''),
                    'currency': currency,
                    'price_per_oz': data.get('price'),
                    'price_per_gram': data.get('price_gram'),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'change_24h': data.get('ch'),
                    'change_24h_pct': data.get('chp'),
                    'high_24h': data.get('high_price'),
                    'low_24h': data.get('low_price')
                }
                
                logger.debug(f"Fetched {metal} price from GoldAPI: ${data.get('price')}")
                return price_record
            else:
                logger.warning(f"GoldAPI returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching from GoldAPI: {e}")
        
        return None
    
    def _fetch_metalpriceapi_price(self, metal: str, currency: str) -> Optional[Dict[str, Any]]:
        """
        Fetch price from MetalPriceAPI.
        
        Args:
            metal: Metal symbol (XAU, XAG, etc.)
            currency: Currency code
            
        Returns:
            Price record or None
        """
        try:
            api_key = self.api_endpoints['metalpriceapi']['api_key']
            if not api_key:
                logger.warning("MetalPriceAPI key not configured")
                return None
            
            url = f"{self.api_endpoints['metalpriceapi']['base_url']}/latest"
            params = {
                'api_key': api_key,
                'base': metal,
                'symbols': currency
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'rates' in data and currency in data['rates']:
                    # Convert rate to price (rates are usually inverted)
                    rate = data['rates'][currency]
                    price_per_oz = 1 / rate if rate > 0 else 0
                    
                    price_record = {
                        'source': 'MetalPriceAPI',
                        'metal': metal.lower().replace('x', ''),
                        'currency': currency,
                        'price_per_oz': round(price_per_oz, 2),
                        'price_per_gram': round(price_per_oz / 31.1035, 2),  # Troy ounce to gram
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'api_timestamp': data.get('updated_date')
                    }
                    
                    logger.debug(f"Fetched {metal} price from MetalPriceAPI: ${price_per_oz:.2f}")
                    return price_record
            else:
                logger.warning(f"MetalPriceAPI returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching from MetalPriceAPI: {e}")
        
        return None
    
    def _fetch_perth_mint_gold(self) -> Optional[Dict[str, Any]]:
        """
        Fetch gold price from Perth Mint CSV data.
        
        Returns:
            Price record or None
        """
        try:
            csv_url = self.australian_sources['perth_mint_csv']
            response = requests.get(csv_url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                # Save raw CSV
                raw_file = self.data_dir / 'perth_mint_gold_raw.csv'
                with open(raw_file, 'w') as f:
                    f.write(response.text)
                
                # Parse CSV to get latest price
                df = pd.read_csv(raw_file)
                
                if not df.empty:
                    latest_row = df.iloc[-1]
                    
                    # Assuming CSV has columns like 'Date', 'Price_AUD'
                    price_record = {
                        'source': 'Perth Mint',
                        'metal': 'gold',
                        'currency': 'AUD',
                        'price_per_oz': float(latest_row.iloc[1]) if len(latest_row) > 1 else 0,
                        'price_per_gram': float(latest_row.iloc[1]) / 31.1035 if len(latest_row) > 1 else 0,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'date': str(latest_row.iloc[0]) if len(latest_row) > 0 else datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    logger.debug("Fetched gold price from Perth Mint")
                    return price_record
            
        except Exception as e:
            logger.error(f"Error fetching from Perth Mint: {e}")
        
        return None
    
    def _get_sample_gold_price(self) -> Dict[str, Any]:
        """Sample gold price based on recent market levels."""
        return {
            'source': 'Sample Data',
            'metal': 'gold',
            'currency': 'AUD',
            'price_per_oz': 3850.00,
            'price_per_gram': 123.75,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'change_24h': 25.50,
            'change_24h_pct': 0.67,
            'description': 'Gold continues strong performance with central bank demand and geopolitical uncertainty'
        }
    
    def _get_sample_silver_price(self) -> Dict[str, Any]:
        """Sample silver price based on recent market levels."""
        return {
            'source': 'Sample Data',
            'metal': 'silver',
            'currency': 'AUD',
            'price_per_oz': 72.50,
            'price_per_gram': 2.33,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'change_24h': 1.25,
            'change_24h_pct': 1.75,
            'description': 'Silver outperforming with industrial demand and investment flows'
        }
    
    def collect_australian_gold_stocks(self) -> List[Dict[str, Any]]:
        """
        Collect Australian gold mining stock prices.
        
        Returns:
            List of gold stock records
        """
        logger.info("Collecting Australian gold mining stock prices...")
        
        import yfinance as yf
        gold_stocks = []
        
        mining_stocks = self.australian_sources['asx_mining_stocks']
        
        for symbol in mining_stocks:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period='1d')
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    
                    stock_record = {
                        'symbol': symbol,
                        'name': info.get('longName', symbol.replace('.AX', '')),
                        'current_price_aud': round(float(current_price), 2),
                        'market_cap': info.get('marketCap'),
                        'sector': 'Gold Mining',
                        'industry': info.get('industry', 'Metals & Mining'),
                        'pe_ratio': info.get('trailingPE'),
                        'dividend_yield_pct': round(float(info.get('dividendYield', 0)) * 100, 2),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'correlation_with_gold': self._get_gold_correlation(symbol),
                        'investment_thesis': self._get_mining_stock_thesis(symbol)
                    }
                    
                    gold_stocks.append(stock_record)
                    logger.debug(f"Collected data for gold stock {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                continue
        
        # Save gold stocks data
        if gold_stocks:
            df = pd.DataFrame(gold_stocks)
            file_path = self.data_dir / 'australian_gold_stocks.csv'
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(gold_stocks)} gold stock records to {file_path}")
        
        return gold_stocks
    
    def _get_gold_correlation(self, symbol: str) -> str:
        """Get gold price correlation description for mining stock."""
        correlations = {
            'NST.AX': 'High positive correlation with gold prices',
            'EVN.AX': 'Strong correlation, leveraged to gold price movements',
            'NCM.AX': 'Moderate correlation, diversified mining operations',
            'RSG.AX': 'High correlation, pure gold play',
            'SLR.AX': 'Strong correlation with precious metals complex'
        }
        return correlations.get(symbol, 'Correlated with precious metals prices')
    
    def _get_mining_stock_thesis(self, symbol: str) -> str:
        """Get investment thesis for mining stock."""
        theses = {
            'NST.AX': 'Australia largest gold producer, strong operational performance',
            'EVN.AX': 'Mid-tier gold producer with Australian and Canadian operations',
            'NCM.AX': 'Major gold producer with diversified asset base',
            'RSG.AX': 'Focused gold producer with development pipeline',
            'SLR.AX': 'Gold and silver producer with growth projects'
        }
        return theses.get(symbol, 'Gold mining exposure with precious metals upside')
    
    def collect_precious_metals_etfs(self) -> List[Dict[str, Any]]:
        """
        Collect precious metals ETF data from ASX.
        
        Returns:
            List of precious metals ETF records
        """
        logger.info("Collecting precious metals ETFs...")
        
        import yfinance as yf
        metals_etfs = []
        
        etf_symbols = self.australian_sources['asx_gold_etfs']
        
        for symbol in etf_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period='1d')
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    
                    etf_record = {
                        'symbol': symbol,
                        'name': info.get('longName', symbol),
                        'current_price_aud': round(float(current_price), 2),
                        'net_assets': info.get('totalAssets'),
                        'expense_ratio_pct': round(float(info.get('annualReportExpenseRatio', 0)) * 100, 2),
                        'underlying_asset': self._get_etf_underlying(symbol),
                        'investment_strategy': self._get_etf_metals_strategy(symbol),
                        'recommended_allocation': self._get_etf_allocation_advice(symbol),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    metals_etfs.append(etf_record)
                    logger.debug(f"Collected ETF data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting ETF data for {symbol}: {e}")
                continue
        
        # Save metals ETFs data
        if metals_etfs:
            df = pd.DataFrame(metals_etfs)
            file_path = self.data_dir / 'precious_metals_etfs.csv'
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(metals_etfs)} metals ETF records to {file_path}")
        
        return metals_etfs
    
    def _get_etf_underlying(self, symbol: str) -> str:
        """Get underlying asset for precious metals ETF."""
        underlying = {
            'GOLD.AX': 'Physical gold bullion',
            'QAU.AX': 'Physical gold bullion',
            'PMGOLD.AX': 'Perth Mint physical gold'
        }
        return underlying.get(symbol, 'Precious metals exposure')
    
    def _get_etf_metals_strategy(self, symbol: str) -> str:
        """Get investment strategy for metals ETF."""
        strategies = {
            'GOLD.AX': 'Currency hedged gold exposure for Australian investors',
            'QAU.AX': 'Unhedged gold exposure, benefits from AUD weakness',
            'PMGOLD.AX': 'Physical gold backed by Perth Mint, direct ownership'
        }
        return strategies.get(symbol, 'Precious metals investment exposure')
    
    def _get_etf_allocation_advice(self, symbol: str) -> str:
        """Get allocation advice for metals ETF."""
        advice = {
            'GOLD.AX': '5-10% portfolio allocation for inflation hedge',
            'QAU.AX': '5-10% allocation with currency diversification benefit',
            'PMGOLD.AX': '5-10% allocation for direct precious metals ownership'
        }
        return advice.get(symbol, '5-10% portfolio allocation as defensive asset')
    
    def run_all_metals_collection(self) -> Dict[str, Any]:
        """
        Run complete precious metals data collection.
        
        Returns:
            Dictionary with collection results
        """
        logger.info("Running complete precious metals data collection...")
        
        results = {}
        
        try:
            # Collect current gold prices
            gold_prices = self.collect_current_gold_prices()
            results['gold_prices'] = len(gold_prices)
            
            # Collect current silver prices  
            silver_prices = self.collect_current_silver_prices()
            results['silver_prices'] = len(silver_prices)
            
            # Collect Australian gold mining stocks
            gold_stocks = self.collect_australian_gold_stocks()
            results['gold_stocks'] = len(gold_stocks)
            
            # Collect precious metals ETFs
            metals_etfs = self.collect_precious_metals_etfs()
            results['metals_etfs'] = len(metals_etfs)
            
            results['status'] = 'completed'
            results['collection_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Precious metals collection completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in metals collection: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
            return results
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of metals data collection."""
        try:
            csv_files = list(self.data_dir.glob('*.csv'))
            
            summary = {
                'collector': 'MetalsDataCollector',
                'data_directory': str(self.data_dir),
                'api_endpoints_configured': len(self.api_endpoints),
                'csv_files_created': len(csv_files),
                'file_names': [f.name for f in csv_files],
                'collection_timestamp': datetime.now().isoformat()
            }
            
            # Get file record counts
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    summary[f"{file.stem}_records"] = len(df)
                except:
                    pass
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting metals collection summary: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    # Test the collector
    from ...utils.config import Config
    config = Config()
    collector = MetalsDataCollector(config)
    results = collector.run_all_metals_collection()
    print(f"Metals collection results: {results}")