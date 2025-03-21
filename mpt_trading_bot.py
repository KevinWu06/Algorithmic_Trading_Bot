import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from portfolio_optimization import MarkowitzOptimizer
import time
import logging
from typing import Dict, List, Tuple

class MPTTradingBot:
    def __init__(self, 
                 initial_capital: float,
                 tickers: List[str],
                 rebalance_frequency: str = 'monthly',  # 'monthly' or 'quarterly'
                 max_position_size: float = 0.25,       # Maximum 25% in single position
                 stop_loss_pct: float = 0.15,          # 15% stop loss
                 risk_free_rate: float = 0.04):        # 4% risk-free rate
        """
        Initialize the MPT Trading Bot
        
        Args:
            initial_capital: Starting capital
            tickers: List of stock tickers
            rebalance_frequency: How often to rebalance
            max_position_size: Maximum allocation for any single position
            stop_loss_pct: Stop loss percentage
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.tickers = tickers
        self.rebalance_frequency = rebalance_frequency
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.risk_free_rate = risk_free_rate
        
        # Initialize portfolio
        self.positions: Dict[str, dict] = {}  # Current positions
        self.target_weights: Dict[str, float] = {}  # Target portfolio weights
        self.last_rebalance_date = None
        
        # Setup logging
        logging.basicConfig(
            filename='mpt_trading_bot.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('MPTTradingBot')
        
    def initialize_portfolio(self):
        """
        Build initial portfolio based on MPT optimization
        """
        try:
            # Get optimal portfolio weights
            optimizer = MarkowitzOptimizer(self.tickers)
            returns = optimizer.fetch_data()
            weights, exp_return, volatility, sharpe = optimizer.optimize_portfolio()
            
            # Apply position size constraints
            weights = self._apply_position_constraints(weights)
            
            # Store target weights
            self.target_weights = dict(zip(self.tickers, weights))
            
            # Calculate initial positions
            for ticker, weight in self.target_weights.items():
                if weight > 0:
                    # Get current price
                    current_price = self._get_current_price(ticker)
                    shares = int((weight * self.initial_capital) / current_price)
                    
                    if shares > 0:
                        self.positions[ticker] = {
                            'shares': shares,
                            'entry_price': current_price,
                            'stop_loss': current_price * (1 - self.stop_loss_pct),
                            'current_price': current_price
                        }
            
            self.last_rebalance_date = datetime.now()
            self.logger.info(f"Portfolio initialized with {len(self.positions)} positions")
            self._log_portfolio_status()
            
        except Exception as e:
            self.logger.error(f"Error initializing portfolio: {str(e)}")
            raise
    
    def _apply_position_constraints(self, weights: np.ndarray) -> np.ndarray:
        """
        Apply maximum position size constraints
        """
        constrained_weights = np.minimum(weights, self.max_position_size)
        # Renormalize weights to sum to 1
        return constrained_weights / constrained_weights.sum()
    
    def _get_current_price(self, ticker: str) -> float:
        """
        Get current price for a ticker
        """
        try:
            stock = yf.Ticker(ticker)
            return stock.history(period='1d')['Close'].iloc[-1]
        except Exception as e:
            self.logger.error(f"Error getting price for {ticker}: {str(e)}")
            raise
    
    def check_rebalance(self):
        """
        Check if portfolio needs rebalancing
        """
        if not self.last_rebalance_date:
            return True
            
        days_since_rebalance = (datetime.now() - self.last_rebalance_date).days
        
        if self.rebalance_frequency == 'monthly' and days_since_rebalance >= 30:
            return True
        elif self.rebalance_frequency == 'quarterly' and days_since_rebalance >= 90:
            return True
            
        return False
    
    def rebalance_portfolio(self):
        """
        Rebalance portfolio to target weights
        """
        try:
            # Update current prices and calculate current weights
            total_value = 0
            current_weights = {}
            
            for ticker, position in self.positions.items():
                current_price = self._get_current_price(ticker)
                position['current_price'] = current_price
                position_value = position['shares'] * current_price
                total_value += position_value
            
            # Calculate rebalancing trades
            trades = []
            for ticker, target_weight in self.target_weights.items():
                current_value = 0
                if ticker in self.positions:
                    current_value = (
                        self.positions[ticker]['shares'] * 
                        self.positions[ticker]['current_price']
                    )
                
                target_value = total_value * target_weight
                value_difference = target_value - current_value
                
                if abs(value_difference) > total_value * 0.01:  # 1% threshold
                    current_price = self._get_current_price(ticker)
                    shares_to_trade = int(value_difference / current_price)
                    if shares_to_trade != 0:
                        trades.append((ticker, shares_to_trade))
            
            # Execute trades
            for ticker, shares in trades:
                self._execute_trade(ticker, shares)
            
            self.last_rebalance_date = datetime.now()
            self.logger.info("Portfolio rebalanced")
            self._log_portfolio_status()
            
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {str(e)}")
            raise
    
    def _execute_trade(self, ticker: str, shares: int):
        """
        Execute a trade (simulate for now)
        """
        try:
            current_price = self._get_current_price(ticker)
            
            if shares > 0:  # Buy
                if ticker not in self.positions:
                    self.positions[ticker] = {
                        'shares': shares,
                        'entry_price': current_price,
                        'stop_loss': current_price * (1 - self.stop_loss_pct),
                        'current_price': current_price
                    }
                else:
                    self.positions[ticker]['shares'] += shares
                    # Update entry price and stop loss
                    total_shares = self.positions[ticker]['shares']
                    old_value = (total_shares - shares) * self.positions[ticker]['entry_price']
                    new_value = shares * current_price
                    new_entry_price = (old_value + new_value) / total_shares
                    self.positions[ticker]['entry_price'] = new_entry_price
                    self.positions[ticker]['stop_loss'] = new_entry_price * (1 - self.stop_loss_pct)
                
            else:  # Sell
                if ticker in self.positions:
                    self.positions[ticker]['shares'] += shares  # shares is negative
                    if self.positions[ticker]['shares'] <= 0:
                        del self.positions[ticker]
            
            self.logger.info(f"Executed trade: {ticker} {shares} shares at {current_price}")
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {ticker}: {str(e)}")
            raise
    
    def check_stop_losses(self):
        """
        Check and handle stop losses
        """
        for ticker in list(self.positions.keys()):
            try:
                current_price = self._get_current_price(ticker)
                if current_price <= self.positions[ticker]['stop_loss']:
                    # Sell entire position
                    shares_to_sell = -self.positions[ticker]['shares']
                    self._execute_trade(ticker, shares_to_sell)
                    self.logger.warning(f"Stop loss triggered for {ticker}")
            except Exception as e:
                self.logger.error(f"Error checking stop loss for {ticker}: {str(e)}")
    
    def _log_portfolio_status(self):
        """
        Log current portfolio status
        """
        total_value = sum(
            pos['shares'] * pos['current_price'] 
            for pos in self.positions.values()
        )
        
        self.logger.info(f"Portfolio Value: ${total_value:,.2f}")
        for ticker, position in self.positions.items():
            value = position['shares'] * position['current_price']
            weight = value / total_value
            self.logger.info(
                f"{ticker}: {position['shares']} shares, "
                f"Weight: {weight:.2%}, "
                f"Value: ${value:,.2f}"
            )

def run_trading_bot():
    """
    Main function to run the trading bot
    """
    # Initialize with top 30 stocks from S&P 500
    from S_and_P500_top_30 import get_sp500_top_stocks
    
    initial_capital = 1000000  # $1M initial capital
    top_stocks = get_sp500_top_stocks(30)
    
    bot = MPTTradingBot(
        initial_capital=initial_capital,
        tickers=top_stocks,
        rebalance_frequency='monthly',
        max_position_size=0.25,
        stop_loss_pct=0.15
    )
    
    try:
        # Initialize portfolio
        bot.initialize_portfolio()
        
        while True:
            # Check stop losses daily
            bot.check_stop_losses()
            
            # Check if rebalancing is needed
            if bot.check_rebalance():
                bot.rebalance_portfolio()
            
            # Wait for next check (e.g., daily)
            time.sleep(86400)  # 24 hours
            
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        bot.logger.error(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    run_trading_bot() 