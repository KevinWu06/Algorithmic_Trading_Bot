import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class MarkowitzOptimizer:
    def __init__(self, tickers):
        """
        Initialize the Markowitz Portfolio Optimizer
        
        Args:
            tickers (list): List of stock ticker symbols
        """
        self.tickers = tickers
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
    def fetch_data(self, start_date=None, end_date=None):
        """
        Fetch historical stock data and calculate returns
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
            
        # Download stock data
        data = pd.DataFrame()
        valid_tickers = []
        
        for ticker in self.tickers:
            try:
                stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not stock.empty:
                    data[ticker] = stock['Close']
                    valid_tickers.append(ticker)
            except Exception as e:
                print(f"Error downloading {ticker}: {str(e)}")
                continue
        
        if data.empty:
            raise ValueError("No valid data downloaded for any tickers")
            
        # Update tickers list to only include valid ones
        self.tickers = valid_tickers
        
        # Calculate daily returns with explicit fill_method=None
        self.returns = data.pct_change(fill_method=None).dropna()
        
        # Check if we have enough data
        if self.returns.empty:
            raise ValueError("Not enough data to calculate returns")
            
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        return self.returns
    
    def optimize_portfolio(self, num_portfolios=1000):
        """
        Generate random portfolios and find the optimal one
        
        Args:
            num_portfolios (int): Number of random portfolios to generate
            
        Returns:
            tuple: (optimal_weights, expected_return, volatility, sharpe_ratio)
        """
        if self.returns is None or self.returns.empty:
            raise ValueError("No return data available. Run fetch_data first.")
            
        results = []
        
        for _ in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(len(self.tickers))
            weights = weights / np.sum(weights)
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(self.mean_returns * weights) * 252  # Annualized return
            portfolio_std = np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix * 252, weights))
            )  # Annualized volatility
            
            # Calculate Sharpe Ratio (assuming risk-free rate = 0.01)
            # Add check for zero volatility
            if portfolio_std == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = (portfolio_return - 0.01) / portfolio_std
            
            results.append({
                'weights': weights,
                'return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe': sharpe_ratio
            })
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Handle case where all Sharpe ratios are NaN
        if results_df['sharpe'].isna().all():
            raise ValueError("Unable to calculate valid portfolio metrics")
            
        # Find the portfolio with highest Sharpe Ratio
        optimal_idx = results_df['sharpe'].fillna(-np.inf).idxmax()
        if pd.isna(optimal_idx):
            raise ValueError("No valid optimal portfolio found")
            
        optimal_portfolio = results_df.iloc[int(optimal_idx)]
        
        return (
            optimal_portfolio['weights'],
            optimal_portfolio['return'],
            optimal_portfolio['volatility'],
            optimal_portfolio['sharpe']
        )
    
    def plot_efficient_frontier(self, results_df):
        """
        Plot the efficient frontier
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(
            results_df['volatility'],
            results_df['return'],
            c=results_df['sharpe'],
            cmap='viridis',
            marker='o',
            s=10,
            alpha=0.3
        )
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.show() 