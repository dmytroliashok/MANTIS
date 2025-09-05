#!/usr/bin/env python3
"""
Test script to validate your embeddings' individual selection performance.
Run this before deploying to see if your signals can pass the first hurdle.
"""

import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class IndividualSelectionTester:
    """Test if your embeddings can pass individual selection."""
    
    def __init__(self, asset_symbol: str = "BTC-USD"):
        self.asset_symbol = asset_symbol
        self.lag_blocks = 60  # Same as validator LAG
        self.chunk_size = 2000  # Same as validator CHUNK_SIZE
    
    def get_price_data(self, period: str = "3mo") -> pd.DataFrame:
        """Get historical price data."""
        ticker = yf.Ticker(self.asset_symbol)
        hist = ticker.history(period=period, interval="1h")
        return hist
    
    def calculate_returns(self, prices: pd.DataFrame) -> np.ndarray:
        """Calculate 1-hour forward returns (same as validator)."""
        close_prices = prices['Close'].values
        returns = []
        
        for i in range(len(close_prices) - self.lag_blocks):
            current_price = close_prices[i]
            future_price = close_prices[i + self.lag_blocks]
            ret = (future_price - current_price) / current_price
            returns.append(ret)
        
        return np.array(returns)
    
    def generate_sample_embeddings(self, prices: pd.DataFrame) -> np.ndarray:
        """Generate sample embeddings for testing."""
        close = prices['Close'].values
        high = prices['High'].values
        low = prices['Low'].values
        
        embeddings = []
        
        for i in range(len(close) - self.lag_blocks):
            # Use data only up to current point (no future leakage)
            current_close = close[:i+1]
            current_high = high[:i+1]
            current_low = low[:i+1]
            
            if len(current_close) < 50:  # Need minimum history
                embeddings.append([0.0] * 10)
                continue
            
            # Sample technical indicators (replace with your actual logic)
            features = []
            
            # 1. Short-term momentum
            momentum_5 = (current_close[-1] / current_close[-6] - 1) * 100 if len(current_close) >= 6 else 0
            features.append(np.tanh(momentum_5))
            
            # 2. RSI-like indicator
            price_changes = np.diff(current_close[-15:]) if len(current_close) >= 15 else [0]
            gains = np.mean([x for x in price_changes if x > 0]) if any(x > 0 for x in price_changes) else 0
            losses = np.mean([-x for x in price_changes if x < 0]) if any(x < 0 for x in price_changes) else 1
            rs = gains / losses if losses > 0 else 1
            rsi = 100 - (100 / (1 + rs))
            features.append((rsi / 100.0 - 0.5) * 2)
            
            # 3. Price position in recent range
            recent_high = np.max(current_high[-20:]) if len(current_high) >= 20 else current_close[-1]
            recent_low = np.min(current_low[-20:]) if len(current_low) >= 20 else current_close[-1]
            if recent_high > recent_low:
                position = (current_close[-1] - recent_low) / (recent_high - recent_low)
                features.append((position - 0.5) * 2)
            else:
                features.append(0.0)
            
            # 4. Volatility measure
            if len(current_close) >= 20:
                volatility = np.std(current_close[-20:]) / np.mean(current_close[-20:])
                features.append(np.tanh(volatility * 100))
            else:
                features.append(0.0)
            
            # 5. Trend strength
            if len(current_close) >= 10:
                trend = (current_close[-1] - current_close[-10]) / current_close[-10]
                features.append(np.tanh(trend * 20))
            else:
                features.append(0.0)
            
            # Add more features to reach 10
            while len(features) < 10:
                # Add combinations of existing features
                if len(features) >= 2:
                    combo = features[0] * features[1]
                    features.append(np.tanh(combo))
                else:
                    features.append(0.0)
            
            embeddings.append(features[:10])
        
        return np.array(embeddings)
    
    def test_individual_selection(self, embeddings: np.ndarray, returns: np.ndarray) -> dict:
        """Test individual selection performance using validator's method."""
        # Convert returns to binary (up/down)
        y_binary = (returns > 0).astype(int)
        
        # Ensure same length
        min_len = min(len(embeddings), len(y_binary))
        X = embeddings[:min_len]
        y = y_binary[:min_len]
        
        if len(X) < 200:
            return {"error": "Not enough data for testing"}
        
        # Walk-forward validation (similar to validator)
        results = []
        window_size = 500
        
        for start_idx in range(0, len(X) - window_size, 100):
            train_end = start_idx + 300
            val_start = train_end + 60  # Embargo period
            val_end = min(val_start + 100, len(X))
            
            if val_end <= val_start or train_end <= 50:
                continue
            
            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                continue
            
            # Train individual model (same as validator)
            clf = LogisticRegression(
                penalty="l2",
                C=0.5,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=200,
            )
            
            try:
                clf.fit(X_train, y_train)
                scores = clf.decision_function(X_val)
                auc = roc_auc_score(y_val, scores)
                results.append(auc)
            except Exception as e:
                continue
        
        if not results:
            return {"error": "No valid windows for testing"}
        
        return {
            "mean_auc": np.mean(results),
            "std_auc": np.std(results),
            "min_auc": np.min(results),
            "max_auc": np.max(results),
            "num_windows": len(results),
            "selection_rate": np.mean([auc > 0.55 for auc in results]),  # Estimated threshold
            "all_aucs": results
        }
    
    def plot_results(self, results: dict):
        """Plot the AUC results."""
        if "error" in results:
            print(f"Error: {results['error']}")
            return
        
        aucs = results["all_aucs"]
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: AUC over time
        plt.subplot(2, 2, 1)
        plt.plot(aucs, 'b-', alpha=0.7)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Random (0.5)')
        plt.axhline(y=0.55, color='orange', linestyle='--', label='Selection Threshold (~0.55)')
        plt.axhline(y=results["mean_auc"], color='g', linestyle='-', label=f'Mean ({results["mean_auc"]:.3f})')
        plt.title('Individual AUC Over Time')
        plt.xlabel('Window')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: AUC distribution
        plt.subplot(2, 2, 2)
        plt.hist(aucs, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Random')
        plt.axvline(x=0.55, color='orange', linestyle='--', label='Selection Threshold')
        plt.axvline(x=results["mean_auc"], color='g', linestyle='-', label='Mean')
        plt.title('AUC Distribution')
        plt.xlabel('AUC')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Selection success rate
        plt.subplot(2, 2, 3)
        selection_success = [auc > 0.55 for auc in aucs]
        cumulative_success = np.cumsum(selection_success) / np.arange(1, len(selection_success) + 1)
        plt.plot(cumulative_success, 'g-', linewidth=2)
        plt.axhline(y=0.5, color='r', linestyle='--', label='50% Selection Rate')
        plt.title(f'Cumulative Selection Rate: {results["selection_rate"]:.1%}')
        plt.xlabel('Window')
        plt.ylabel('Selection Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Performance summary
        plt.subplot(2, 2, 4)
        metrics = ['Mean AUC', 'Selection Rate', 'Consistency']
        values = [
            results["mean_auc"],
            results["selection_rate"],
            1 - (results["std_auc"] / results["mean_auc"]) if results["mean_auc"] > 0 else 0
        ]
        colors = ['green' if v > 0.6 else 'orange' if v > 0.4 else 'red' for v in values]
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.title('Performance Summary')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, results: dict):
        """Generate a detailed report."""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print("🔍 INDIVIDUAL SELECTION TEST RESULTS")
        print("=" * 50)
        print(f"Asset: {self.asset_symbol}")
        print(f"Test Windows: {results['num_windows']}")
        print()
        
        print("📊 AUC PERFORMANCE")
        print(f"  Mean AUC: {results['mean_auc']:.3f}")
        print(f"  Std AUC:  {results['std_auc']:.3f}")
        print(f"  Min AUC:  {results['min_auc']:.3f}")
        print(f"  Max AUC:  {results['max_auc']:.3f}")
        print()
        
        print("🎯 SELECTION ANALYSIS")
        selection_rate = results['selection_rate']
        print(f"  Selection Rate: {selection_rate:.1%}")
        
        if selection_rate > 0.8:
            print("  ✅ EXCELLENT - Very likely to be selected")
        elif selection_rate > 0.6:
            print("  ✅ GOOD - Likely to be selected most of the time")
        elif selection_rate > 0.4:
            print("  ⚠️  MARGINAL - Sometimes selected, needs improvement")
        elif selection_rate > 0.2:
            print("  ❌ POOR - Rarely selected, major improvements needed")
        else:
            print("  ❌ FAILING - Almost never selected, complete rework needed")
        
        print()
        print("💡 RECOMMENDATIONS")
        
        if results['mean_auc'] < 0.52:
            print("  🔴 Your signals have no predictive power")
            print("     - Check for data leakage (using future information)")
            print("     - Verify your indicators are calculated correctly")
            print("     - Consider completely different signal sources")
        elif results['mean_auc'] < 0.55:
            print("  🟡 Your signals are weak but show some promise")
            print("     - Focus on faster-reacting indicators")
            print("     - Add market microstructure signals")
            print("     - Reduce signal smoothing/averaging")
        elif results['std_auc'] > 0.1:
            print("  🟡 Your signals are inconsistent")
            print("     - Add regime detection (bull/bear markets)")
            print("     - Use adaptive parameters")
            print("     - Combine multiple signal types for stability")
        else:
            print("  🟢 Your signals look good for individual selection!")
            print("     - Focus on uniqueness to avoid redundancy")
            print("     - Consider expanding to other assets")
            print("     - Monitor performance after deployment")

def main():
    """Test your embeddings' individual selection performance."""
    print("🧪 MANTIS Individual Selection Tester")
    print("=" * 50)
    
    # Initialize tester
    tester = IndividualSelectionTester("BTC-USD")
    
    print("📥 Fetching price data...")
    prices = tester.get_price_data("3mo")
    
    print("📊 Calculating returns...")
    returns = tester.calculate_returns(prices)
    
    print("🔧 Generating sample embeddings...")
    embeddings = tester.generate_sample_embeddings(prices)
    
    print("🧮 Testing individual selection performance...")
    results = tester.test_individual_selection(embeddings, returns)
    
    # Generate report
    tester.generate_report(results)
    
    # Plot results
    if "error" not in results:
        print("\n📈 Generating performance plots...")
        tester.plot_results(results)
    
    print("\n💡 Next Steps:")
    print("1. Replace generate_sample_embeddings() with your actual embedding logic")
    print("2. Test on multiple assets (BTC, ETH, EURUSD, etc.)")
    print("3. Iterate on your signals based on the results")
    print("4. Only deploy when you consistently achieve >60% selection rate")

if __name__ == "__main__":
    main()