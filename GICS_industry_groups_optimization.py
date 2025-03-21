from portfolio_optimization import MarkowitzOptimizer
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_sp500_top_stocks(limit=30):
    """
    Get top performing stocks from S&P 500 based on 15 years of historical data
    """
    # Get S&P 500 tickers
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = sp500['Symbol'].tolist()
    
    # Set date range for 15 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15*365)
    
    # Initialize metrics dictionary
    stock_metrics = {}
    print("Analyzing S&P 500 stocks over 15-year period...")
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            # Get 15 years of historical data
            hist = stock.history(start=start_date, end=end_date)
            if len(hist) < 2500:  # Ensure enough trading days (roughly 15 years)
                continue
                
            # Calculate long-term metrics
            returns = hist['Close'].pct_change()
            total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
            annual_return = (1 + total_return) ** (1/15) - 1  # Annualized 15-year return
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility != 0 else 0
            
            # Calculate rolling metrics
            rolling_returns = returns.rolling(window=252).mean()  # 1-year rolling returns
            rolling_vol = returns.rolling(window=252).std() * np.sqrt(252)
            consistency = rolling_returns.mean() / rolling_returns.std()  # Return consistency
            
            try:
                info = stock.info
                market_cap = info.get('marketCap', 0)
                pe_ratio = info.get('forwardPE', 0)
                profit_margin = info.get('profitMargins', 0)
                
                # Calculate composite score with emphasis on long-term performance
                score = (
                    0.25 * sharpe_ratio +          # Risk-adjusted returns
                    0.25 * annual_return +         # Long-term returns
                    0.20 * consistency +           # Return consistency
                    0.15 * (market_cap / 1e12) +  # Size/stability
                    0.10 * (1/pe_ratio if pe_ratio else 0) +  # Valuation
                    0.05 * (profit_margin if profit_margin else 0)  # Profitability
                )
                
                stock_metrics[ticker] = {
                    'score': score,
                    'annual_return': annual_return,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'consistency': consistency,
                    'market_cap': market_cap
                }
                
            except Exception as e:
                continue
                
        except Exception as e:
            continue
            
        # Print progress
        print(f"Analyzed {ticker}", end='\r')
    
    # Sort stocks by composite score
    sorted_stocks = dict(sorted(
        stock_metrics.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    ))
    
    # Get top N tickers
    top_tickers = list(sorted_stocks.keys())[:limit]
    
    # Print top stocks info
    print("\n\nTop 30 S&P 500 Stocks (15-Year Analysis):")
    print("Ticker | 15Y Ann. Return | Total Return | Sharpe | Consistency")
    print("-" * 65)
    for ticker in top_tickers:
        metrics = sorted_stocks[ticker]
        print(f"{ticker:6} | {metrics['annual_return']:11.2%} | {metrics['total_return']:10.2%} | {metrics['sharpe_ratio']:6.2f} | {metrics['consistency']:10.2f}")
    
    return top_tickers

if __name__ == "__main__":
    # Get top 30 S&P 500 stocks
    print("\nSelecting top 30 stocks based on 15-year performance...")
    top_stocks = get_sp500_top_stocks(30)
    
    # Create optimizer instance with 15-year data period
    optimizer = MarkowitzOptimizer(top_stocks)
    
    try:
        # Fetch 15 years of data
        returns = optimizer.fetch_data(
            start_date=datetime.now() - timedelta(days=15*365),
            end_date=datetime.now()
        )
        
        # Find optimal portfolio
        optimal_weights, exp_return, volatility, sharpe = optimizer.optimize_portfolio()
        
        # Print results
        print("\nOptimal Portfolio Weights:")
        sorted_weights = sorted(zip(top_stocks, optimal_weights), key=lambda x: x[1], reverse=True)
        for ticker, weight in sorted_weights:
            if weight > 0.001:  # Only show weights > 0.1%
                print(f"{ticker:6}: {weight:.2%}")
        
        print(f"\nPortfolio Metrics (Based on 15-Year History):")
        print(f"Expected Annual Return: {exp_return:.2%}")
        print(f"Annual Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Try reducing the number of companies or checking data availability.") 