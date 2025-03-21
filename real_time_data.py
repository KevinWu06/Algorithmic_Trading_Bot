import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Tuple
from threading import Thread, Lock
from queue import Queue

class MarketDataStream:
    def __init__(self, tickers: List[str], update_interval: int = 60):
        """
        Initialize real-time market data stream
        
        Args:
            tickers: List of stock tickers to monitor
            update_interval: How often to update data (seconds)
        """
        self.tickers = tickers
        self.update_interval = update_interval
        self.data_queue = Queue()
        self.running = False
        self.data_lock = Lock()
        
        # Store latest data
        self.current_data: Dict[str, Dict] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
        # Setup logging
        self.logger = logging.getLogger('MarketDataStream')
        
        # Initialize metrics storage
        self.expected_returns = {}
        self.covariance_matrix = None
        self.last_metrics_update = None
        
    def start(self):
        """Start the data stream"""
        self.running = True
        self.update_thread = Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
    def stop(self):
        """Stop the data stream"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
            
    def _update_loop(self):
        """Main update loop for market data"""
        while self.running:
            try:
                self._fetch_latest_data()
                self._update_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in update loop: {str(e)}")
                
    def _fetch_latest_data(self):
        """Fetch latest market data for all tickers"""
        for ticker in self.tickers:
            try:
                # Get real-time data
                stock = yf.Ticker(ticker)
                latest = stock.history(period='1d', interval='1m').iloc[-1]
                
                with self.data_lock:
                    self.current_data[ticker] = {
                        'price': latest['Close'],
                        'volume': latest['Volume'],
                        'timestamp': datetime.now()
                    }
                    
                # Put data in queue for subscribers
                self.data_queue.put({
                    'ticker': ticker,
                    'data': self.current_data[ticker]
                })
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
                
    def _update_metrics(self):
        """Update expected returns and covariance matrix"""
        now = datetime.now()
        
        # Update metrics quarterly
        if (self.last_metrics_update is None or 
            (now - self.last_metrics_update).days >= 90):
            
            try:
                # Get historical data for all tickers
                end_date = now
                start_date = end_date - timedelta(days=365*2)  # 2 years of data
                
                data_frames = []
                for ticker in self.tickers:
                    df = yf.download(ticker, start=start_date, end=end_date)['Close']
                    data_frames.append(df)
                
                # Combine all data
                historical_prices = pd.concat(data_frames, axis=1)
                historical_prices.columns = self.tickers
                
                # Calculate returns
                returns = historical_prices.pct_change().dropna()
                
                with self.data_lock:
                    # Update expected returns (annualized)
                    self.expected_returns = returns.mean() * 252
                    
                    # Update covariance matrix (annualized)
                    self.covariance_matrix = returns.cov() * 252
                    
                    # Store historical data
                    self.historical_data = historical_prices
                    
                self.last_metrics_update = now
                self.logger.info("Updated market metrics")
                
            except Exception as e:
                self.logger.error(f"Error updating metrics: {str(e)}")
                
    def get_current_price(self, ticker: str) -> float:
        """Get latest price for a ticker"""
        with self.data_lock:
            if ticker in self.current_data:
                return self.current_data[ticker]['price']
        return None
        
    def get_market_metrics(self) -> Tuple[pd.Series, pd.DataFrame]:
        """Get current expected returns and covariance matrix"""
        with self.data_lock:
            return self.expected_returns.copy(), self.covariance_matrix.copy() 