import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
'''
This implementation includes:
1. A MarkowitzOptimizer class that handles all portfolio optimization calculations
2. Data fetching using yfinance for historical stock prices
3. Portfolio optimization using Monte Carlo simulation to find the optimal weights

4. Calculation of key metrics:
        Expected returns
        Portfolio volatility (risk)
        Sharpe ratio (risk-adjusted return)
5. Visualization of the efficient frontier

To use this code, install: pip install numpy pandas yfinance matplotlib

The model uses several key linear algebra concepts:
    1. Matrix multiplication for calculating portfolio variance
    2. Covariance matrix computation
    3. Vector operations for portfolio returns

This is implementing the formula: σp = √(wᵀΣw), where:
    w is the weight vector
    Σ is the covariance matrix
    wᵀ is the transpose of the weight vector

To use the model, simply create an instance with desired stock tickers and run the optimization:

# Example usage
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
optimizer = MarkowitzOptimizer(tickers)
optimizer.fetch_data()
optimal_weights, exp_return, volatility, sharpe = optimizer.optimize_portfolio()

The model will return the optimal portfolio weights that maximize the Sharpe ratio (risk-adjusted return) based on historical data. 
'''
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
        for ticker in self.tickers:
            stock = yf.download(ticker, start=start_date, end=end_date)
            data[ticker] = stock['Close']
            
        # Calculate daily returns
        self.returns = data.pct_change().dropna()
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
            sharpe_ratio = (portfolio_return - 0.01) / portfolio_std
            
            results.append({
                'weights': weights,
                'return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe': sharpe_ratio
            })
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Find the portfolio with highest Sharpe Ratio
        optimal_idx = results_df['sharpe'].idxmax()
        optimal_portfolio = results_df.iloc[optimal_idx]
        
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

# Example usage
if __name__ == "__main__":
    # Define stock tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
    
    # Create optimizer instance
    optimizer = MarkowitzOptimizer(tickers)
    
    # Fetch data for the last year
    returns = optimizer.fetch_data()
    
    # Find optimal portfolio
    optimal_weights, exp_return, volatility, sharpe = optimizer.optimize_portfolio()
    
    # Print results
    print("\nOptimal Portfolio Weights:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.4f}")
    
    print(f"\nExpected Annual Return: {exp_return:.4f}")
    print(f"Annual Volatility: {volatility:.4f}")
    print(f"Sharpe Ratio: {sharpe:.4f}") 