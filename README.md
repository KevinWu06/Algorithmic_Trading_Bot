# AlgorithmicTradingBot
S&P 500 Stock Selection and Portfolio Optimizer

## Description
This tool analyzes the S&P 500 stocks over a 15-year period to select the top 30 performing stocks based on multiple metrics including:
- Annual and total returns
- Sharpe ratio (risk-adjusted returns)
- Maximum drawdown
- Return stability

It then uses Modern Portfolio Theory (Markowitz Optimization) to determine optimal portfolio weights for the selected stocks.


## Key Features

### Stock Selection
- Analyzes S&P 500 stocks using 15 years of historical data
- Selects top 30 stocks based on:
  - Risk-adjusted returns (Sharpe ratio)
  - Historical performance
  - Return stability
  - Maximum drawdown

### Portfolio Optimization
- Uses Modern Portfolio Theory (MPT)
- Optimizes for maximum Sharpe ratio
- Implements position size constraints
- Handles risk management through diversification

### Real-Time Trading
- Live market data monitoring
- Automated portfolio rebalancing
- Stop-loss implementation
- Performance tracking

### Backtesting
- Tests strategy on historical data
- Optimizes trading parameters
- Calculates key performance metrics
- Visualizes results

## Requirements
- Python 3.6+

## How to Run

1. Clone the repository
2. Install dependencies: 
```bash
pip install -r requirements
```
## Dependencies
- yfinance: Market data access
- pandas: Data analysis
- numpy: Numerical computations
- scipy: Portfolio optimization
- matplotlib: Data visualization
- alpha_vantage: Alternative data source
- websocket-client: Real-time data
- tqdm: Progress tracking

## Usage

### Run Backtesting
Test the strategy on historical data: 
```bash
python backtesting.py
```
### Start Trading Bot
Launch the live trading system: 
```bash
python mpt_trading_bot.py
```
## Risk Management
- Maximum position size limits
- Automated stop-loss orders
- Regular portfolio rebalancing
- Correlation-based diversification

## Performance Metrics
- Total Return
- Annual Return
- Sharpe Ratio
- Maximum Drawdown
- Portfolio Volatility
- Number of Trades
