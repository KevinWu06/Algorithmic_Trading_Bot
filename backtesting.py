import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class PortfolioBacktester:
    def __init__(self,
                 initial_capital: float,
                 start_date: datetime,
                 end_date: datetime,
                 rebalance_frequency: str = 'monthly',
                 max_position_size: float = 0.25,
                 stop_loss_pct: float = 0.15,
                 risk_free_rate: float = 0.04):
        """
        Initialize backtester
        """
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.rebalance_frequency = rebalance_frequency
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.risk_free_rate = risk_free_rate
        
        # Results storage
        self.portfolio_values = []
        self.portfolio_weights = []
        self.trades = []
        self.metrics = {}
        
    def optimize_portfolio(self, returns: pd.DataFrame) -> np.array:
        """
        Optimize portfolio weights using MPT with robust handling of singular matrices
        """
        n_assets = len(returns.columns)
        
        try:
            # Calculate mean returns and covariance
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Add small diagonal values to ensure matrix is not singular
            regularization = 1e-8
            cov_matrix = cov_matrix + np.eye(n_assets) * regularization
            
            # Check matrix condition
            if np.linalg.cond(cov_matrix) > 1e15:  # Matrix is ill-conditioned
                print("Warning: Covariance matrix is ill-conditioned, using alternative optimization")
                return self._optimize_alternative(returns)
            
            def portfolio_stats(weights):
                weights = np.clip(weights, 0, self.max_position_size)
                weights = weights / np.sum(weights)  # Normalize
                
                ret = np.sum(mean_returns * weights)
                vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else -np.inf
                return ret, vol, sharpe
            
            def objective(weights):
                _, _, sharpe = portfolio_stats(weights)
                return -sharpe  # Minimize negative Sharpe ratio
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
            
            # Initial guess - equal weights
            init_weights = np.array([1/n_assets] * n_assets)
            
            result = minimize(
                objective,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-8,
                    'disp': False
                }
            )
            
            if result.success:
                # Normalize and clip final weights
                final_weights = np.clip(result.x, 0, self.max_position_size)
                return final_weights / np.sum(final_weights)
                
            return self._optimize_alternative(returns)
            
        except Exception as e:
            print(f"Error in primary optimization: {str(e)}")
            return self._optimize_alternative(returns)
        
    def _optimize_alternative(self, returns: pd.DataFrame) -> np.array:
        """
        Alternative optimization method when primary method fails
        Uses risk parity approach with momentum overlay
        """
        n_assets = len(returns.columns)
        
        try:
            # Calculate momentum scores
            momentum = returns.mean() * np.sqrt(252)
            momentum = (momentum - momentum.mean()) / momentum.std()
            
            # Calculate volatilities
            vols = returns.std() * np.sqrt(252)
            inv_vols = 1 / vols
            
            # Combine momentum and inverse volatility
            scores = momentum * inv_vols
            
            # Convert to weights
            weights = np.maximum(scores, 0)  # Only long positions
            
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.array([1/n_assets] * n_assets)
            
            # Apply position size constraints
            weights = np.clip(weights, 0, self.max_position_size)
            weights = weights / weights.sum()
            
            return weights
            
        except Exception as e:
            print(f"Error in alternative optimization: {str(e)}")
            # Fallback to equal weights
            weights = np.array([1/n_assets] * n_assets)
            return weights 
    def run_backtest(self, tickers: List[str]) -> Dict:
        """
        Run backtest simulation
        """
        print("Downloading historical data...")
        data = yf.download(tickers, start=self.start_date, end=self.end_date)['Close']
        
        if data.empty:
            raise ValueError("No data downloaded")
            
        print("Running backtest simulation...")
        portfolio_value = self.initial_capital
        current_positions = {}
        lookback_window = 252 * 15  # 15 years of trading days
        
        # Initialize portfolio values list with initial capital
        self.portfolio_values = [self.initial_capital]
        
        for i in range(lookback_window, len(data)):
            current_date = data.index[i]
            
            # Get historical data for optimization
            hist_data = data.iloc[i-lookback_window:i]
            returns = hist_data.pct_change().dropna()
            
            # Rebalance if needed
            if self._should_rebalance(current_date) or i == lookback_window:
                # Get optimal weights
                weights = self.optimize_portfolio(returns)
                target_weights = dict(zip(tickers, weights))
                
                # Calculate trades needed
                for ticker, target_weight in target_weights.items():
                    current_price = data[ticker].iloc[i]
                    target_value = portfolio_value * target_weight
                    
                    current_value = 0
                    if ticker in current_positions:
                        current_value = current_positions[ticker]['shares'] * current_price
                        
                    value_diff = target_value - current_value
                    
                    if abs(value_diff) > portfolio_value * 0.01:  # 1% threshold
                        shares_to_trade = int(value_diff / current_price)
                        
                        if ticker in current_positions:
                            current_positions[ticker]['shares'] += shares_to_trade
                            if current_positions[ticker]['shares'] <= 0:
                                del current_positions[ticker]
                        else:
                            current_positions[ticker] = {
                                'shares': shares_to_trade,
                                'entry_price': current_price
                            }
                            
                        self.trades.append({
                            'date': current_date,
                            'ticker': ticker,
                            'shares': shares_to_trade,
                            'price': current_price
                        })
            
            # Check stop losses
            for ticker in list(current_positions.keys()):
                current_price = data[ticker].iloc[i]
                stop_price = current_positions[ticker]['entry_price'] * (1 - self.stop_loss_pct)
                
                if current_price <= stop_price:
                    shares = current_positions[ticker]['shares']
                    self.trades.append({
                        'date': current_date,
                        'ticker': ticker,
                        'shares': -shares,
                        'price': current_price,
                        'type': 'stop_loss'
                    })
                    del current_positions[ticker]
            
            # Calculate portfolio value
            portfolio_value = sum(
                pos['shares'] * data[ticker].iloc[i]
                for ticker, pos in current_positions.items()
            )
            
            self.portfolio_values.append(portfolio_value)
            
        # Calculate performance metrics
        self._calculate_metrics()
        
        return self.metrics
        
    def _should_rebalance(self, date: datetime) -> bool:
        """Check if portfolio should be rebalanced"""
        if self.rebalance_frequency == 'monthly':
            return date.day == 1
        elif self.rebalance_frequency == 'quarterly':
            return date.day == 1 and date.month in [1, 4, 7, 10]
        return False
        
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        values = pd.Series(self.portfolio_values)
        returns = values.pct_change().dropna()
        
        # Calculate metrics one by one
        total_return = (values.iloc[-1] / values.iloc[0]) - 1
        
        self.metrics = {
            'total_return': total_return,
            'annual_return': ((1 + total_return) ** (252/len(returns)) - 1),
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() - self.risk_free_rate/252) / returns.std() * np.sqrt(252),
            'max_drawdown': (values / values.expanding().max() - 1).min(),
            'num_trades': len(self.trades)
        }
        
    def plot_results(self):
        """Plot backtest results"""
        plt.figure(figsize=(15, 10))
        
        # Portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(self.portfolio_values)
        plt.title('Portfolio Value Over Time')
        plt.grid(True)
        
        # Drawdown
        plt.subplot(2, 1, 2)
        values = pd.Series(self.portfolio_values)
        drawdown = values / values.expanding().max() - 1
        plt.plot(drawdown)
        plt.title('Portfolio Drawdown')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def run_optimization_test():
    """Run optimization and backtesting analysis"""
    from S_and_P500_top_30 import get_sp500_top_stocks
    
    # Parameters to test
    rebalance_frequencies = ['monthly', 'quarterly']
    position_sizes = [0.15, 0.25, 0.35]
    stop_losses = [0.10, 0.15, 0.20]
    
    # Test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years
    
    # Get stocks
    tickers = get_sp500_top_stocks(30)
    
    # Store results
    results = []
    
    # Run tests
    for freq in rebalance_frequencies:
        for pos_size in position_sizes:
            for stop_loss in stop_losses:
                print(f"\nTesting: {freq}, {pos_size}, {stop_loss}")
                backtester = PortfolioBacktester(
                    initial_capital=1000000,
                    start_date=start_date,
                    end_date=end_date,
                    rebalance_frequency=freq,
                    max_position_size=pos_size,
                    stop_loss_pct=stop_loss
                )
                
                try:
                    metrics = backtester.run_backtest(tickers)
                    metrics.update({
                        'rebalance_freq': freq,
                        'position_size': pos_size,
                        'stop_loss': stop_loss
                    })
                    results.append(metrics)
                except Exception as e:
                    print(f"Error in backtest: {str(e)}")
                    continue
    
    if not results:
        print("No successful backtest results")
        return None
    
    # Convert to DataFrame and sort by Sharpe ratio
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    # Print best parameters
    best_params = results_df.iloc[0]
    print("\nBest Parameters:")
    print(f"Rebalance Frequency: {best_params['rebalance_freq']}")
    print(f"Maximum Position Size: {best_params['position_size']:.2%}")
    print(f"Stop Loss: {best_params['stop_loss']:.2%}")
    print(f"Sharpe Ratio: {best_params['sharpe_ratio']:.2f}")
    print(f"Annual Return: {best_params['annual_return']:.2%}")
    print(f"Maximum Drawdown: {best_params['max_drawdown']:.2%}")
    
    return best_params

if __name__ == "__main__":
    best_params = run_optimization_test() 