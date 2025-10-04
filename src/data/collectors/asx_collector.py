# src/data/collectors/asx_collector.py
# ASX Market Data Collector - Updated with Real Market Data

import yfinance as yf
import pandas as pd
import requests
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

from ...utils.config import Config
from ...utils.logger import get_logger

logger = get_logger(__name__)

class ASXDataCollector:
    """
    Collector for ASX market data using yfinance and direct market sources.
    Fetches real-time and historical data for Australian securities.
    """
    
    def __init__(self, config: Config):
        """Initialize ASX data collector."""
        self.config = config
        self.data_dir = Path(config.get('data.asx_path', 'data/raw/asx_data'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Comprehensive list of Australian securities
        self.symbols = config.get('asx.symbols', [
            # Australian ETFs
            'VAS.AX',    # Vanguard Australian Shares Index
            'VGS.AX',    # Vanguard Global Shares Index
            'NDQ.AX',    # BetaShares NASDAQ 100
            'A200.AX',   # BetaShares Australian 200
            'VAF.AX',    # Vanguard Australian Fixed Interest
            'VAP.AX',    # Vanguard Australian Property Securities
            'VGB.AX',    # Vanguard Government Bond Index
            'VDHG.AX',   # Vanguard Diversified High Growth
            
            # Major Australian Stocks
            'CBA.AX',    # Commonwealth Bank
            'ANZ.AX',    # ANZ Banking Group
            'WBC.AX',    # Westpac Banking Corporation
            'NAB.AX',    # National Australia Bank
            'BHP.AX',    # BHP Group
            'RIO.AX',    # Rio Tinto
            'CSL.AX',    # CSL Limited
            'WOW.AX',    # Woolworths Group
            'TLS.AX',    # Telstra Corporation
            'GMG.AX',    # Goodman Group
            
            # Gold/Mining Stocks
            'NST.AX',    # Northern Star Resources
            'EVN.AX',    # Evolution Mining
            'NCM.AX',    # Newcrest Mining
            'S32.AX',    # South32
            'FMG.AX',    # Fortescue Metals Group
            'OZL.AX',    # OZ Minerals
        ])
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ASX Data Collector)'
        }
        
        logger.info(f"ASXDataCollector initialized with {len(self.symbols)} symbols")
    
    def collect_current_prices(self) -> List[Dict[str, Any]]:
        """
        Collect current ASX prices for configured symbols.
        
        Returns:
            List of current price records
        """
        logger.info("Collecting current ASX prices...")
        records = []
        now = datetime.now()
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get current price data
                info = ticker.info
                hist = ticker.history(period='5d')
                
                if not hist.empty:
                    latest_price = hist['Close'].iloc[-1]
                    previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else latest_price
                    
                    # Calculate daily change
                    daily_change = latest_price - previous_close
                    daily_change_pct = (daily_change / previous_close) * 100 if previous_close != 0 else 0
                    
                    # Get additional info
                    volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                    market_cap = info.get('marketCap', None)
                    pe_ratio = info.get('trailingPE', None)
                    dividend_yield = info.get('dividendYield', None)
                    
                    record = {
                        'symbol': symbol,
                        'name': info.get('longName', symbol.replace('.AX', '')),
                        'current_price_aud': round(float(latest_price), 2),
                        'previous_close': round(float(previous_close), 2),
                        'daily_change_aud': round(float(daily_change), 2),
                        'daily_change_pct': round(float(daily_change_pct), 2),
                        'volume': int(volume) if volume else 0,
                        'market_cap_aud': market_cap,
                        'pe_ratio': pe_ratio,
                        'dividend_yield_pct': round(float(dividend_yield * 100), 2) if dividend_yield else None,
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                        'date': now.strftime('%Y-%m-%d')
                    }
                    
                    records.append(record)
                    logger.debug(f"Collected price for {symbol}: ${latest_price:.2f}")
                
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
                continue
        
        # Save current prices
        if records:
            df = pd.DataFrame(records)
            file_path = self.data_dir / 'current_asx_prices.csv'
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(records)} current ASX prices to {file_path}")
        
        return records
    
    def collect_historical_prices(self, period: str = '1y') -> List[Dict[str, Any]]:
        """
        Collect historical price data for ASX symbols.
        
        Args:
            period: Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            
        Returns:
            List of historical price records
        """
        logger.info(f"Collecting ASX historical prices for period: {period}")
        all_records = []
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                for date, row in hist.iterrows():
                    record = {
                        'symbol': symbol,
                        'date': date.strftime('%Y-%m-%d'),
                        'open_aud': round(float(row['Open']), 2),
                        'high_aud': round(float(row['High']), 2),
                        'low_aud': round(float(row['Low']), 2),
                        'close_aud': round(float(row['Close']), 2),
                        'volume': int(row['Volume']) if row['Volume'] else 0,
                        'adj_close_aud': round(float(row['Adj Close']), 2),
                    }
                    all_records.append(record)
                
                logger.debug(f"Collected {len(hist)} historical records for {symbol}")
                
            except Exception as e:
                logger.error(f"Error fetching historical data for {symbol}: {e}")
                continue
        
        # Save historical data
        if all_records:
            df = pd.DataFrame(all_records)
            file_path = self.data_dir / f'asx_historical_{period}.csv'
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(all_records)} historical records to {file_path}")
        
        return all_records
    
    def collect_etf_details(self) -> List[Dict[str, Any]]:
        """
        Collect detailed information about Australian ETFs.
        
        Returns:
            List of ETF detail records
        """
        logger.info("Collecting ETF details...")
        etf_symbols = [s for s in self.symbols if any(etf in s for etf in ['VAS', 'VGS', 'NDQ', 'A200', 'VAF', 'VAP', 'VGB', 'VDHG'])]
        
        etf_records = []
        
        for symbol in etf_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get current price
                hist = ticker.history(period='1d')
                current_price = hist['Close'].iloc[-1] if not hist.empty else None
                
                etf_detail = {
                    'symbol': symbol,
                    'name': info.get('longName', symbol),
                    'current_price_aud': round(float(current_price), 2) if current_price else None,
                    'expense_ratio_pct': round(float(info.get('annualReportExpenseRatio', 0)) * 100, 2),
                    'dividend_yield_pct': round(float(info.get('dividendYield', 0)) * 100, 2),
                    'net_assets': info.get('totalAssets', None),
                    'inception_date': info.get('fundInceptionDate', None),
                    'category': info.get('category', 'Unknown'),
                    'fund_family': info.get('fundFamily', 'Unknown'),
                    'investment_strategy': self._get_etf_strategy(symbol),
                    'asset_allocation': self._get_etf_allocation(symbol),
                    'recommended_for': self._get_etf_recommendation(symbol),
                    'collection_date': datetime.now().strftime('%Y-%m-%d')
                }
                
                etf_records.append(etf_detail)
                logger.debug(f"Collected ETF details for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting ETF details for {symbol}: {e}")
                continue
        
        # Save ETF details
        if etf_records:
            df = pd.DataFrame(etf_records)
            file_path = self.data_dir / 'asx_etf_details.csv'
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(etf_records)} ETF details to {file_path}")
        
        return etf_records
    
    def _get_etf_strategy(self, symbol: str) -> str:
        """Get investment strategy description for ETF."""
        strategies = {
            'VAS.AX': 'Tracks ASX 300 index, providing broad Australian equity exposure',
            'VGS.AX': 'Global equity exposure excluding Australia, market cap weighted',
            'NDQ.AX': 'Tracks NASDAQ 100, focused on US technology giants',
            'A200.AX': 'Tracks ASX 200 index, concentrated on largest Australian companies',
            'VAF.AX': 'Australian government and corporate bond exposure',
            'VAP.AX': 'Australian listed property securities and REITs',
            'VGB.AX': 'Australian government bonds, defensive asset allocation',
            'VDHG.AX': 'Diversified high growth allocation across global markets'
        }
        return strategies.get(symbol, 'Diversified investment strategy')
    
    def _get_etf_allocation(self, symbol: str) -> str:
        """Get asset allocation description for ETF."""
        allocations = {
            'VAS.AX': '100% Australian equities',
            'VGS.AX': '100% International developed markets equities',
            'NDQ.AX': '100% US technology and growth stocks',
            'A200.AX': '100% Australian large-cap equities',
            'VAF.AX': '100% Australian fixed income',
            'VAP.AX': '100% Australian property securities',
            'VGB.AX': '100% Australian government bonds',
            'VDHG.AX': '90% growth assets, 10% defensive assets'
        }
        return allocations.get(symbol, 'Diversified allocation')
    
    def _get_etf_recommendation(self, symbol: str) -> str:
        """Get recommendation for who should consider this ETF."""
        recommendations = {
            'VAS.AX': 'Core Australian equity holding for all investors',
            'VGS.AX': 'International diversification, long-term growth',
            'NDQ.AX': 'Technology exposure, higher risk tolerance investors',
            'A200.AX': 'Simple Australian equity exposure, beginners',
            'VAF.AX': 'Conservative investors, defensive allocation',
            'VAP.AX': 'Property exposure, income-focused investors',
            'VGB.AX': 'Capital preservation, pre-retirement investors',
            'VDHG.AX': 'Single diversified solution, growth-oriented investors'
        }
        return recommendations.get(symbol, 'Diversified investment exposure')
    
    def collect_market_indices(self) -> List[Dict[str, Any]]:
        """
        Collect Australian market index data.
        
        Returns:
            List of market index records
        """
        logger.info("Collecting Australian market indices...")
        
        indices = [
            '^AXJO',  # ASX 200
            '^AORD',  # All Ordinaries
            '^AXKO',  # ASX 300
        ]
        
        index_records = []
        
        for index_symbol in indices:
            try:
                ticker = yf.Ticker(index_symbol)
                hist = ticker.history(period='5d')
                info = ticker.info
                
                if not hist.empty:
                    current_level = hist['Close'].iloc[-1]
                    previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_level
                    
                    change = current_level - previous_close
                    change_pct = (change / previous_close) * 100 if previous_close != 0 else 0
                    
                    record = {
                        'index_symbol': index_symbol,
                        'name': self._get_index_name(index_symbol),
                        'current_level': round(float(current_level), 2),
                        'previous_close': round(float(previous_close), 2),
                        'daily_change': round(float(change), 2),
                        'daily_change_pct': round(float(change_pct), 2),
                        'year_high': round(float(hist['High'].max()), 2),
                        'year_low': round(float(hist['Low'].min()), 2),
                        'description': self._get_index_description(index_symbol),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    index_records.append(record)
                    logger.debug(f"Collected index data for {index_symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting index data for {index_symbol}: {e}")
                continue
        
        # Save index data
        if index_records:
            df = pd.DataFrame(index_records)
            file_path = self.data_dir / 'asx_market_indices.csv'
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(index_records)} market indices to {file_path}")
        
        return index_records
    
    def _get_index_name(self, symbol: str) -> str:
        """Get full name for index symbol."""
        names = {
            '^AXJO': 'ASX 200',
            '^AORD': 'All Ordinaries',
            '^AXKO': 'ASX 300'
        }
        return names.get(symbol, symbol)
    
    def _get_index_description(self, symbol: str) -> str:
        """Get description for index."""
        descriptions = {
            '^AXJO': 'Market capitalization weighted index of 200 largest ASX-listed companies',
            '^AORD': 'Market capitalization weighted index of largest and most liquid ASX companies',
            '^AXKO': 'Market capitalization weighted index of 300 largest ASX-listed companies'
        }
        return descriptions.get(symbol, 'Australian stock market index')
    
    def run_all(self) -> Dict[str, Any]:
        """
        Run full ASX data collection workflow.
        
        Returns:
            Dictionary with collection results
        """
        logger.info("Running complete ASX data collection...")
        
        results = {}
        
        try:
            # Collect current prices
            current_prices = self.collect_current_prices()
            results['current_prices'] = len(current_prices)
            
            # Collect historical data (1 year)
            historical_data = self.collect_historical_prices('1y')
            results['historical_records'] = len(historical_data)
            
            # Collect ETF details
            etf_details = self.collect_etf_details()
            results['etf_details'] = len(etf_details)
            
            # Collect market indices
            market_indices = self.collect_market_indices()
            results['market_indices'] = len(market_indices)
            
            results['status'] = 'completed'
            results['collection_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"ASX data collection completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in ASX data collection: {e}")
            results['status'] = 'error'
            results['error'] = str(e)
            return results
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of ASX data collection."""
        try:
            csv_files = list(self.data_dir.glob('*.csv'))
            
            summary = {
                'collector': 'ASXDataCollector',
                'data_directory': str(self.data_dir),
                'symbols_configured': len(self.symbols),
                'csv_files_created': len(csv_files),
                'file_names': [f.name for f in csv_files],
                'collection_timestamp': datetime.now().isoformat()
            }
            
            # Get file sizes and record counts
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    summary[f"{file.stem}_records"] = len(df)
                except:
                    pass
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting ASX collection summary: {e}")
            return {'error': str(e)}


if __name__ == "__main__":
    # Test the collector
    from ...utils.config import Config
    config = Config()
    collector = ASXDataCollector(config)
    results = collector.run_all()
    print(f"ASX collection results: {results}")