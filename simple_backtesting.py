#!/usr/bin/env python3
"""
Simple Backtesting Framework for TradingAgents
=============================================

A streamlined backtesting framework that:
1. Runs for day T and makes purchasing decisions on day T+1 (prevents data leakage)
2. Tests for 2 days with 1 stock initially
3. Uses realistic bid-ask spreads from real data
4. Uses realistic trading costs ($0 commission as per modern brokers)
5. Tracks all trades and model reasoning
6. Computes performance metrics vs benchmark

Usage:
    python simple_backtesting.py                    # Use real TradingAgents LLM
    python simple_backtesting.py --stub            # Use stubbed random decisions
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass, asdict
import logging
import argparse
import hashlib

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deterministic_hash(s: str) -> int:
    """Create a deterministic hash that's consistent between runs"""
    # Use SHA-256 which is deterministic
    hash_object = hashlib.sha256(s.encode())
    # Convert first 8 bytes to integer
    hash_int = int.from_bytes(hash_object.digest()[:8], byteorder='big')
    # Modulo to get number in range [0, 1000000)
    return hash_int % 1000000

@dataclass
class SimpleBacktestConfig:
    """Configuration for simple backtesting"""
    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # Maximum 10% of portfolio per position
    commission_per_trade: float = 0.0  # $0 commission (modern brokers)
    benchmark_ticker: str = "SPY"
    risk_free_rate: float = 0.02  # 2% annual risk-free rate

@dataclass
class Trade:
    """Individual trade record"""
    ticker: str
    decision_date: datetime
    execution_date: datetime
    decision: str
    reasoning: str
    entry_price: float
    exit_price: Optional[float] = None
    quantity: int = 0
    commission: float = 0.0
    pnl: float = 0.0
    hold_days: int = 0

@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    benchmark_return: float
    alpha: float
    beta: float

class SimpleBacktester:
    """Simple backtesting engine for TradingAgents"""
    
    def __init__(self, config: SimpleBacktestConfig = None, ta_config: Dict = None, use_stub: bool = False):
        self.config = config or SimpleBacktestConfig()
        self.ta_config = ta_config or DEFAULT_CONFIG.copy()
        self.use_stub = use_stub
        self.global_seed = None  # Will be set in run_simple_backtest
        
        # Initialize TradingAgents only if not using stub
        if not self.use_stub:
            self.trading_agents = TradingAgentsGraph(
                debug=False,
                config=self.ta_config
            )
        else:
            self.trading_agents = None
        
        # Trading state
        self.trades = []
        self.portfolio = {}
        self.cash = self.config.initial_capital
        self.equity_curve = []
        
    def get_realistic_price_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch price data with realistic bid-ask spreads"""
        try:
            logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
            
            # Download data with auto_adjust=False to get real OHLC data
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            
            if data.empty:
                logger.warning(f"No data for {ticker}")
                return pd.DataFrame()
            
            logger.info(f"Raw data for {ticker}: {len(data)} rows")
            logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
            logger.info(f"Columns: {data.columns.tolist()}")
            
            # Handle multi-level columns (when downloading single ticker, yfinance sometimes returns multi-level)
            if isinstance(data.columns, pd.MultiIndex):
                logger.info(f"Detected multi-level columns, flattening...")
                # Flatten the multi-level columns by taking the first level
                data.columns = data.columns.get_level_values(0)
                logger.info(f"Flattened columns: {data.columns.tolist()}")
            
            # Calculate realistic bid-ask spreads based on real volatility
            # Higher volatility = wider spreads
            volatility = data['Close'].pct_change().rolling(window=20).std()
            
            # For the first few days where we don't have enough data for rolling volatility,
            # use a simple approach: calculate volatility from available data
            if len(data) < 20:
                # Use all available data for volatility calculation
                volatility = data['Close'].pct_change().std()
                # Fill any remaining NaN with a small default
                volatility = volatility if not pd.isna(volatility) else 0.01
            
            # Calculate spread as base + volatility adjustment
            spread_pct = 0.001 + (volatility * 2)  # Base 0.1% + volatility adjustment
            
            # Add bid/ask prices
            data['Bid'] = data['Close'] * (1 - spread_pct / 2)
            data['Ask'] = data['Close'] * (1 + spread_pct / 2)
            
            # Add ticker column
            data['Ticker'] = ticker
            
            logger.info(f"Loaded data for {ticker}: {len(data)} rows")
            logger.info(f"Real volatility: {volatility:.4f} ({volatility*100:.2f}%)")
            logger.info(f"Spread: {spread_pct*100:.3f}%")
            logger.info(f"Sample prices: Close=${data['Close'].iloc[0]:.2f}, Bid=${data['Bid'].iloc[0]:.2f}, Ask=${data['Ask'].iloc[0]:.2f}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_execution_price(self, ticker: str, date: datetime, decision: str, 
                          price_data: pd.DataFrame) -> float:
        """Get realistic execution price including spread"""
        try:
            day_data = price_data[price_data.index.date == date.date()]
            
            if day_data.empty:
                logger.warning(f"No data for {ticker} on {date.date()}")
                return None
            
            close_price = day_data['Close'].iloc[0]
            bid_price = day_data['Bid'].iloc[0]
            ask_price = day_data['Ask'].iloc[0]
            
            if decision == "BUY":
                return ask_price  # Buy at ask
            elif decision == "SELL":
                return bid_price  # Sell at bid
            else:
                return close_price  # Hold at close
                
        except Exception as e:
            logger.error(f"Error getting execution price for {ticker} on {date}: {e}")
            return None
    
    def run_simple_backtest(self, ticker: str, start_date: str, end_date: str) -> Dict:
        """Run simple backtest for 2 days with 1 stock"""
        logger.info(f"Starting simple backtest for {ticker}: {start_date} to {end_date}")
        
        # Set global seed for reproducibility if using stub
        if self.use_stub:
            self.global_seed = deterministic_hash(f"{ticker}_{start_date}_{end_date}")
            np.random.seed(self.global_seed)
            logger.info(f"Global seed set to {self.global_seed} for reproducibility")
        
        # Reset state
        self.trades = []
        self.portfolio = {}
        self.cash = self.config.initial_capital
        self.equity_curve = []
        
        # Get price data
        price_data = self.get_realistic_price_data(ticker, start_date, end_date)
        if price_data.empty:
            logger.error(f"No price data available for {ticker}")
            return {}
        
        # Get trading dates (simple approach for 2 days)
        trading_dates = price_data.index.tolist()
        if len(trading_dates) < 2:
            logger.warning(f"Only {len(trading_dates)} trading days found for {ticker}, trying wider date range")
            
            # Try a wider date range to get more trading days
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            wider_start = (start_dt - timedelta(days=7)).strftime('%Y-%m-%d')
            wider_end = (end_dt + timedelta(days=7)).strftime('%Y-%m-%d')
            
            logger.info(f"Trying wider date range: {wider_start} to {wider_end}")
            price_data = self.get_realistic_price_data(ticker, wider_start, wider_end)
            
            if price_data.empty:
                logger.error(f"No price data available for {ticker} even with wider range")
                return {}
            
            trading_dates = price_data.index.tolist()
            if len(trading_dates) < 2:
                logger.error(f"Insufficient trading days for {ticker} even with wider range")
                return {}
            
            logger.info(f"Using wider date range, found {len(trading_dates)} trading days")
        
        logger.info(f"Trading dates: {[d.strftime('%Y-%m-%d') for d in trading_dates]}")
        
        # Run backtesting loop
        for i in range(len(trading_dates) - 1):
            current_date = trading_dates[i]
            next_date = trading_dates[i + 1]
            
            logger.info(f"Processing day {i+1}: {current_date.strftime('%Y-%m-%d')} -> {next_date.strftime('%Y-%m-%d')}")
            
            # Make trading decision using data up to current_date
            decision, reasoning = self._make_trading_decision(ticker, current_date.strftime('%Y-%m-%d'))
            
            # Execute trade on next_date
            self._execute_trade(ticker, decision, reasoning, current_date, next_date, price_data)
            
            # Update equity curve
            portfolio_value = self._calculate_portfolio_value(current_date, price_data)
            total_value = portfolio_value + self.cash
            
            self.equity_curve.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'total_value': total_value
            })
            
            logger.info(f"Day {i+1} summary:")
            logger.info(f"  Decision: {decision}")
            logger.info(f"  Portfolio value: ${portfolio_value:.2f}")
            logger.info(f"  Cash: ${self.cash:.2f}")
            logger.info(f"  Total value: ${total_value:.2f}")
        
        # Close all remaining positions
        self._close_all_positions(trading_dates[-1], price_data)
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(start_date, end_date)
        
        # Generate report with actual trading period
        actual_start = trading_dates[0].strftime('%Y-%m-%d')
        actual_end = trading_dates[-1].strftime('%Y-%m-%d')
        report = self._generate_report(ticker, actual_start, actual_end, performance)
        
        return {
            'ticker': ticker,
            'performance': performance,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'report': report,
            'actual_trading_period': {
                'start': actual_start,
                'end': actual_end,
                'trading_days': len(trading_dates)
            }
        }
    
    def _make_trading_decision(self, ticker: str, date: str) -> Tuple[str, str]:
        """Make trading decision using TradingAgents or stubbed random decisions"""
        try:
            if self.use_stub:
                # Create a deterministic seed by combining the global seed with the hash
                # of the ticker+date. This ensures the same decisions for the same
                # global seed while maintaining unique seeds per decision.
                decision_seed = deterministic_hash(f"{self.global_seed}_{ticker}_{date}")
                
                # Create a new random state for this specific decision
                rng = np.random.RandomState(decision_seed)
                
                # Generate random decision: 40% BUY, 30% SELL, 30% HOLD
                rand_val = rng.random()
                print(rand_val)
                if rand_val < 0.4:
                    decision = "BUY"
                    reasoning = f"Random BUY decision for {ticker} on {date} (seed: {decision_seed})"
                elif rand_val < 0.7:
                    decision = "SELL"
                    reasoning = f"Random SELL decision for {ticker} on {date} (seed: {decision_seed})"
                else:
                    decision = "HOLD"
                    reasoning = f"Random HOLD decision for {ticker} on {date} (seed: {decision_seed})"
                
                logger.info(f"Stubbed decision for {ticker} on {date}: '{decision}' (seed: {decision_seed})")
                
                return decision, reasoning
            else:
                # Use TradingAgents to make decision
                _, decision = self.trading_agents.propagate(ticker, date)
                
                logger.info(f"TradingAgents decision for {ticker} on {date}: '{decision}'")
                
                # Extract BUY/SELL/HOLD from decision
                decision_upper = decision.upper()
                
                # Check for BUY signals
                buy_keywords = ["BUY", "PURCHASE", "ACQUIRE", "LONG", "BULLISH", "POSITIVE", "STRONG BUY"]
                if any(keyword in decision_upper for keyword in buy_keywords):
                    return "BUY", decision
                
                # Check for SELL signals
                sell_keywords = ["SELL", "SHORT", "BEARISH", "NEGATIVE", "STRONG SELL", "DUMP", "EXIT"]
                if any(keyword in decision_upper for keyword in sell_keywords):
                    return "SELL", decision
                
                # Default to HOLD
                return "HOLD", decision
                
        except Exception as e:
            logger.error(f"Error making decision for {ticker}: {e}")
            return "HOLD", f"Error: {e}"
    
    def _execute_trade(self, ticker: str, decision: str, reasoning: str, 
                      decision_date: datetime, execution_date: datetime, price_data: pd.DataFrame):
        """Execute trade with realistic costs"""
        try:
            execution_price = self.get_execution_price(ticker, execution_date, decision, price_data)
            
            if execution_price is None:
                logger.warning(f"No execution price for {ticker} on {execution_date.strftime('%Y-%m-%d')}")
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(execution_price)
            logger.info(f"Position size calculation for {ticker}: {position_size} shares at ${execution_price:.2f}")
            
            if decision == "BUY" and position_size > 0:
                # Buy logic
                total_cost = execution_price * position_size
                commission = self.config.commission_per_trade
                
                if total_cost + commission <= self.cash:
                    self.cash -= (total_cost + commission)
                    
                    if ticker in self.portfolio:
                        self.portfolio[ticker] += position_size
                    else:
                        self.portfolio[ticker] = position_size
                    
                    trade = Trade(
                        ticker=ticker,
                        decision_date=decision_date,
                        execution_date=execution_date,
                        decision=decision,
                        reasoning=reasoning,
                        entry_price=execution_price,
                        quantity=position_size,
                        commission=commission
                    )
                    self.trades.append(trade)
                    logger.info(f"BUY trade executed for {ticker}: {position_size} shares at ${execution_price:.2f}")
                else:
                    logger.warning(f"BUY order rejected for {ticker}: insufficient cash")
            
            elif decision == "SELL" and ticker in self.portfolio and self.portfolio[ticker] > 0:
                # Sell logic
                quantity = self.portfolio[ticker]
                total_proceeds = execution_price * quantity
                commission = self.config.commission_per_trade
                
                self.cash += (total_proceeds - commission)
                self.portfolio[ticker] = 0
                
                # Find corresponding buy trade to calculate PnL
                for trade in reversed(self.trades):
                    if trade.ticker == ticker and trade.decision == "BUY" and trade.exit_price is None:
                        trade.exit_price = execution_price
                        trade.hold_days = (execution_date - trade.decision_date).days
                        trade.pnl = (execution_price - trade.entry_price) * trade.quantity - trade.commission - commission
                        break
                
                logger.info(f"SELL trade executed for {ticker}: {quantity} shares at ${execution_price:.2f}")
                
            elif decision == "HOLD":
                logger.info(f"HOLD decision for {ticker} - no trade executed")
            elif decision == "BUY":
                logger.info(f"BUY decision for {ticker} - no trade executed (position size: {position_size})")
            elif decision == "SELL":
                logger.info(f"SELL decision for {ticker} - no trade executed (no position to sell)")
            else:
                logger.info(f"Unknown decision '{decision}' for {ticker}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _calculate_position_size(self, price: float) -> int:
        """Calculate position size based on risk management rules"""
        if pd.isna(price) or price <= 0:
            return 0
        
        max_position_value = self.cash * self.config.max_position_size
        max_shares = int(max_position_value / price)
        
        # Ensure we don't exceed cash available
        affordable_shares = int(self.cash / price)
        
        return min(max_shares, affordable_shares)
    
    def _calculate_portfolio_value(self, date: datetime, price_data: pd.DataFrame) -> float:
        """Calculate current portfolio value"""
        portfolio_value = 0
        
        for ticker, quantity in self.portfolio.items():
            if quantity > 0:
                date_data = price_data[price_data.index.date == date.date()]
                if not date_data.empty:
                    current_price = date_data['Close'].iloc[0]
                    portfolio_value += current_price * quantity
        
        return portfolio_value
    
    def _close_all_positions(self, final_date: datetime, price_data: pd.DataFrame):
        """Close all remaining positions"""
        logger.info(f"Closing all positions on {final_date.strftime('%Y-%m-%d')}")
        
        for ticker, quantity in self.portfolio.items():
            if quantity > 0:
                final_data = price_data[price_data.index.date == final_date.date()]
                
                if not final_data.empty:
                    final_price = final_data['Close'].iloc[0]
                    proceeds = final_price * quantity
                    commission = self.config.commission_per_trade
                    
                    self.cash += (proceeds - commission)
                    
                    # Update trade records
                    for trade in reversed(self.trades):
                        if trade.ticker == ticker and trade.decision == "BUY" and trade.exit_price is None:
                            trade.exit_price = final_price
                            trade.hold_days = (final_date - trade.decision_date).days
                            trade.pnl = (final_price - trade.entry_price) * trade.quantity - trade.commission - commission
                            break
        
        self.portfolio = {}
    
    def _calculate_performance_metrics(self, start_date: str, end_date: str) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        # Get benchmark data
        try:
            benchmark_data = yf.download(self.config.benchmark_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not benchmark_data.empty and 'Close' in benchmark_data.columns and len(benchmark_data) > 0:
                benchmark_return = (benchmark_data['Close'].iloc[-1] / benchmark_data['Close'].iloc[0]) - 1
            else:
                logger.warning(f"No benchmark data available for {self.config.benchmark_ticker}")
                benchmark_return = 0.0
        except Exception as e:
            logger.warning(f"Error fetching benchmark data: {e}")
            benchmark_return = 0.0
        
        # Calculate returns
        initial_value = self.config.initial_capital
        final_value = self.equity_curve[-1]['total_value'] if self.equity_curve else initial_value
        total_return = (final_value / initial_value) - 1
        
        # Calculate annual return
        days = (datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days
        annual_return = float((1 + total_return) ** (365 / days) - 1) if days > 0 else 0.0
        
        # Calculate Sharpe ratio
        if self.equity_curve:
            returns = pd.Series([eq['total_value'] for eq in self.equity_curve]).pct_change().dropna()
            excess_returns = returns - (self.config.risk_free_rate / 365)
            sharpe_ratio = float(excess_returns.mean() / excess_returns.std() * np.sqrt(365)) if excess_returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        if self.equity_curve:
            values = [eq['total_value'] for eq in self.equity_curve]
            peak = values[0]
            max_drawdown = 0
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0
        
        # Calculate trade statistics
        completed_trades = [t for t in self.trades if t.exit_price is not None]
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        
        win_rate = float(len(winning_trades) / len(completed_trades)) if completed_trades else 0.0
        
        # Calculate alpha and beta (simplified)
        alpha = float(annual_return - benchmark_return)
        beta = 1.0  # Simplified - would need correlation analysis for accurate beta
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(completed_trades),
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta
        )
    
    def _generate_report(self, ticker: str, start_date: str, end_date: str, 
                        performance: PerformanceMetrics) -> str:
        """Generate comprehensive performance report"""
        
        # Helper function to convert pandas Series to scalar
        def to_scalar(value):
            if hasattr(value, 'iloc'):
                return float(value.iloc[0]) if len(value) > 0 else 0.0
            elif hasattr(value, 'item'):
                return float(value.item())
            else:
                return float(value) if value is not None else 0.0
        
        # Get final portfolio value
        final_value = 0.0
        if self.equity_curve:
            final_value = to_scalar(self.equity_curve[-1]['total_value'])
        
        report = f"""
Simple Backtest Report
====================

Ticker: {ticker}
Period: {start_date} to {end_date}

Performance Summary:
- Total Return: {to_scalar(performance.total_return):.2%}
- Annual Return: {to_scalar(performance.annual_return):.2%}
- Sharpe Ratio: {to_scalar(performance.sharpe_ratio):.2f}
- Max Drawdown: {to_scalar(performance.max_drawdown):.2%}

Trading Statistics:
- Total Trades: {int(to_scalar(performance.total_trades))}
- Win Rate: {to_scalar(performance.win_rate):.2%}

Risk-Adjusted Performance:
- Alpha vs {self.config.benchmark_ticker}: {to_scalar(performance.alpha):.2%}
- Beta: {to_scalar(performance.beta):.2f}
- Benchmark Return: {to_scalar(performance.benchmark_return):.2%}

Final Portfolio Value: ${final_value:,.2f}
Initial Capital: ${to_scalar(self.config.initial_capital):,.2f}

Trade Details:
"""
        
        for i, trade in enumerate(self.trades, 1):
            report += f"\nTrade {i}:"
            report += f"\n  Date: {trade.decision_date.strftime('%Y-%m-%d')}"
            report += f"\n  Decision: {trade.decision}"
            report += f"\n  Reasoning: {trade.reasoning[:200]}..."  # Truncate long reasoning
            report += f"\n  Entry Price: ${trade.entry_price:.2f}"
            if trade.exit_price:
                report += f"\n  Exit Price: ${trade.exit_price:.2f}"
                report += f"\n  PnL: ${trade.pnl:.2f}"
                report += f"\n  Hold Days: {trade.hold_days}"
        
        return report
    
    def save_results(self, results: Dict, output_file: str = "simple_backtest_results.json"):
        """Save backtest results to file"""
        # Convert dataclasses to dictionaries
        results_dict = {
            'ticker': results['ticker'],
            'performance': asdict(results['performance']),
            'trades': [asdict(trade) for trade in results['trades']],
            'equity_curve': results['equity_curve']
        }
        
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")


def main():
    """Example usage of the simple backtesting framework"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simple Backtesting Framework for TradingAgents')
    parser.add_argument('--stub', action='store_true', 
                       help='Use stubbed random decisions instead of TradingAgents LLM')
    args = parser.parse_args()
    
    # Configure backtesting parameters
    backtest_config = SimpleBacktestConfig(
        initial_capital=100000.0,  # $100k initial capital
        max_position_size=.2,  # 20% max position size
        commission_per_trade=0.0,  # $0 commission (modern brokers)
    )
    
    # Configure TradingAgents (only if not using stub)
    ta_config = None
    if not args.stub:
        ta_config = DEFAULT_CONFIG.copy()
        ta_config["llm_provider"] = "openai"
        ta_config["deep_think_llm"] = "gpt-4o-mini"
        ta_config["quick_think_llm"] = "gpt-4o-mini"
        ta_config["online_tools"] = False  # Use offline data for backtesting
    
    # Initialize backtester
    backtester = SimpleBacktester(
        config=backtest_config,
        ta_config=ta_config,
        use_stub=args.stub
    )
    
    # Print mode
    mode = "STUBBED RANDOM" if args.stub else "TRADING AGENTS LLM"
    print(f"Running backtest in {mode} mode")
    
    # Run simple backtest for 2 days with 1 stock
    # Use dates that are definitely trading days (avoid holidays/weekends)
    results = backtester.run_simple_backtest(
        ticker="AAPL",
        start_date="2024-01-08",  # Monday (first full week of 2024)
        end_date="2024-01-09"     # Tuesday
    )
    
    if results:
        # Save results
        output_file = "simple_backtest_results_stub.json" if args.stub else "simple_backtest_results.json"
        backtester.save_results(results, output_file)
        
        # Print actual trading period info
        if 'actual_trading_period' in results:
            period = results['actual_trading_period']
            print(f"\nActual Trading Period: {period['start']} to {period['end']} ({period['trading_days']} trading days)")
        
        # Print report
        print(results['report'])
    else:
        print("Backtest failed - no results generated")


if __name__ == "__main__":
    main() 