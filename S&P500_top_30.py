from portfolio_optimization import MarkowitzOptimizer
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm

def get_sp500_top_stocks(limit=30):
    """
    Get top performing stocks from S&P 500 based on 15 years of historical data
    """
    # Get S&P 500 tickers
    try:
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers = sp500['Symbol'].tolist()
        print(f"Found {len(tickers)} stocks in S&P 500")
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        return []
    
    # Set date range for 15 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15*365)
    
    # Initialize metrics dictionary
    stock_metrics = {}
    print(f"Analyzing S&P 500 stocks from {start_date.date()} to {end_date.date()}...")
    
    # Use tqdm for progress bar
    for ticker in tqdm(tickers):
        try:
            # Clean ticker symbol
            ticker = ticker.replace('.','-')
            
            # Download all data at once with error handling
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                timeout=10
            )
            
            # Skip if no data or not enough data
            if df.empty or len(df) < 1250:  # Roughly 5 years of trading days
                continue
                
            # Calculate daily returns using pct_change
            df['Returns'] = df['Close'].pct_change()
            
            # Remove any NaN values
            df = df.dropna()
            
            # Get clean arrays
            close_prices = df['Close'].values
            returns = df['Returns'].values
            
            # Skip if not enough data after cleaning
            if len(close_prices) < 1250:
                continue
            
            # Basic metrics
            total_return = (close_prices[-1] / close_prices[0]) - 1
            years = len(df) / 252  # Actual number of years
            annual_return = (1 + total_return) ** (1/years) - 1
            volatility = returns.std() * np.sqrt(252)
            
            # Avoid division by zero and extreme values
            if volatility == 0 or np.isnan(volatility) or np.abs(annual_return) > 1:
                continue
                
            sharpe_ratio = annual_return / volatility
            
            # Calculate drawdown
            peak = np.maximum.accumulate(close_prices)
            drawdowns = (close_prices - peak) / peak
            max_drawdown = drawdowns.min()
            
            # Calculate rolling returns (252-day window)
            rolling_returns = pd.Series(returns).rolling(window=252).mean()
            rolling_std = pd.Series(returns).rolling(window=252).std()
            
            # Calculate return stability
            valid_returns = rolling_returns.dropna()
            valid_std = rolling_std.dropna()
            
            if len(valid_returns) > 0 and valid_std.mean() != 0:
                return_stability = valid_returns.mean() / valid_std.mean()
            else:
                continue
            
            # Skip if we have invalid metrics
            if (np.isnan(return_stability) or np.isnan(sharpe_ratio) or 
                np.isnan(max_drawdown) or np.isinf(return_stability)):
                continue
            
            # Composite score calculation
            score = (
                0.3 * sharpe_ratio +           # Risk-adjusted returns
                0.3 * annual_return +          # Raw returns
                0.2 * return_stability +       # Stability of returns
                0.2 * (1 + max_drawdown)       # Drawdown protection
            )
            
            # Skip extreme scores
            if np.isnan(score) or np.abs(score) > 100:
                continue
            
            # Store metrics
            stock_metrics[ticker] = {
                'annual_return': annual_return,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'return_stability': return_stability,
                'score': score
            }
                
        except Exception as e:
            continue
    
    print(f"\nFound {len(stock_metrics)} valid stocks for analysis")
    
    if not stock_metrics:
        print("No valid stocks found")
        return []
    
    # Sort stocks by score
    sorted_stocks = dict(sorted(
        stock_metrics.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    ))
    
    # Get top N tickers
    top_tickers = list(sorted_stocks.keys())[:limit]
    # Print top stocks info
    print("\n\nTop Stocks Analysis:")
    print("Ticker | Ann Return | Total Return | Sharpe | Max Drawdown | Stability")
    print("-" * 75)
    for ticker in top_tickers:
        m = sorted_stocks[ticker]
        # Convert numpy values to Python floats for proper formatting
        print(f"{ticker:6} | {float(m['annual_return'])*100:9.2f}% | "
              f"{float(m['total_return'])*100:9.2f}% | "
              f"{float(m['sharpe_ratio']):6.2f} | "
              f"{float(m['max_drawdown'])*100:9.2f}% | "
              f"{float(m['return_stability']):8.2f}")
    
    return top_tickers

if __name__ == "__main__":
    try:
        # Get top stocks
        print("\nSelecting top stocks based on long-term performance...")
        top_stocks = get_sp500_top_stocks(30)
        
        if not top_stocks:
            print("No stocks selected. Exiting.")
            exit()
        
        print(f"\nSelected {len(top_stocks)} stocks for portfolio optimization")
        
        # Create optimizer instance
        optimizer = MarkowitzOptimizer(top_stocks)
        
        # Fetch data and optimize portfolio
        returns = optimizer.fetch_data(
            start_date=datetime.now() - timedelta(days=15*365),
            end_date=datetime.now()
        )
        
        optimal_weights, exp_return, volatility, sharpe = optimizer.optimize_portfolio()
        
        # Print results
        print("\nOptimal Portfolio Weights:")
        sorted_weights = sorted(zip(top_stocks, optimal_weights), key=lambda x: float(x[1]), reverse=True)
        for ticker, weight in sorted_weights:
            if float(weight) > 0.001:  # Only show weights > 0.1%
                print(f"{ticker:6}: {float(weight)*100:.2f}%")
        
        print(f"\nPortfolio Metrics:")
        print(f"Expected Annual Return: {float(exp_return)*100:.2f}%")
        print(f"Annual Volatility: {float(volatility)*100:.2f}%")
        print(f"Sharpe Ratio: {float(sharpe):.2f}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}") 