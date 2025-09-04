# MANTIS Miner Strategies

This document outlines different approaches to building successful MANTIS miners based on the validator's scoring mechanism.

## Understanding the Validator

The validator uses **counterfactual modeling** with XGBoost and permutation importance:

1. **Trains a model** on all miners' embeddings to predict 1-hour price changes
2. **Measures each miner's contribution** by removing their data and seeing how much the model performance degrades
3. **Rewards unique, predictive signals** - redundant information gets low scores

## Key Success Factors

### 1. **Uniqueness is Critical**
- Avoid copying other miners' strategies
- The more unique your signal, the higher your potential reward
- If multiple miners submit similar data, all get penalized

### 2. **Predictive Power Matters**
- Your embeddings must actually help predict price movements
- Random noise or irrelevant data will score zero
- Focus on signals that have genuine predictive value

### 3. **Multi-Asset Approach**
- BTC gets 100 dimensions (primary focus)
- Other assets get 2 dimensions each
- Your total score is aggregated across all assets

## Winning Strategies

### Strategy 1: Technical Analysis Master
**Focus**: Advanced technical indicators and market microstructure

```python
# Key signals to include:
- Multi-timeframe RSI, MACD, Bollinger Bands
- Order book analysis (bid/ask spreads, depth)
- Volume profile and flow analysis
- Support/resistance levels
- Chart pattern recognition
- Cross-asset technical correlations
```

### Strategy 2: Alternative Data Specialist
**Focus**: Non-traditional data sources

```python
# Unique data sources:
- Social media sentiment (Twitter, Reddit, Discord)
- Google Trends and search volume
- GitHub activity for crypto projects
- Whale wallet movements
- Exchange flow analysis
- News sentiment with NLP
- Economic calendar events
```

### Strategy 3: Macro Economic Analyst
**Focus**: Macroeconomic factors and cross-market relationships

```python
# Macro signals:
- Interest rate expectations
- Currency correlations
- Commodity relationships (gold, oil)
- Stock market correlations
- VIX and fear indicators
- Central bank policy signals
- Economic data releases
```

### Strategy 4: High-Frequency Microstructure
**Focus**: Short-term market dynamics

```python
# Microstructure signals:
- Tick-by-tick price action
- Order flow imbalances
- Market maker behavior
- Latency arbitrage signals
- Cross-exchange spreads
- Funding rate dynamics
- Options flow (for crypto)
```

### Strategy 5: Machine Learning Ensemble
**Focus**: Advanced ML techniques

```python
# ML approaches:
- LSTM/GRU for time series
- Transformer models for sequence prediction
- Ensemble methods combining multiple models
- Reinforcement learning for adaptive strategies
- Anomaly detection for regime changes
- Feature engineering with polynomial/interaction terms
```

## Implementation Tips

### For BTC (100 dimensions):
```python
# Distribute your 100 dimensions across:
- 30-40 dims: Technical indicators (various timeframes)
- 20-30 dims: Alternative data (sentiment, on-chain)
- 15-20 dims: Macro factors (correlations, economic)
- 10-15 dims: Microstructure (order book, flow)
- 5-10 dims: Time-based features (seasonality, cycles)
```

### For Other Assets (2 dimensions each):
```python
# Focus on the most predictive signals:
- Dimension 1: Primary trend/momentum signal
- Dimension 2: Confirmation/volatility signal

# Examples:
EURUSD: [interest_rate_differential, risk_sentiment]
XAUUSD: [dollar_strength, inflation_expectations]
```

## Advanced Techniques

### 1. **Feature Engineering**
```python
# Create non-linear combinations:
- Polynomial features: x1 * x2, x1^2
- Trigonometric transforms: sin(x), cos(x)
- Statistical moments: skewness, kurtosis
- Rolling statistics: moving averages, volatility
```

### 2. **Regime Detection**
```python
# Adapt to market conditions:
- Bull/bear market detection
- High/low volatility regimes
- Risk-on/risk-off environments
- Market session effects (Asia/Europe/US)
```

### 3. **Cross-Asset Intelligence**
```python
# Use relationships between assets:
- BTC-ETH correlation patterns
- Forex carry trade signals
- Gold-dollar inverse relationship
- Stock-crypto correlation shifts
```

### 4. **Temporal Features**
```python
# Time-based patterns:
- Hour of day effects
- Day of week seasonality
- Month-end rebalancing
- Options expiry effects
- Economic calendar timing
```

## Common Pitfalls to Avoid

### ❌ **Don't Do This:**
- Copy other miners' strategies exactly
- Use only basic indicators everyone knows
- Submit random noise hoping to get lucky
- Focus only on BTC and ignore other assets
- Use outdated or stale data
- Hardcode values without market adaptation

### ✅ **Do This Instead:**
- Develop unique, proprietary signals
- Combine multiple data sources creatively
- Validate your signals on historical data
- Adapt to changing market conditions
- Monitor your performance and iterate
- Focus on genuine predictive value

## Performance Monitoring

Track these metrics to optimize your strategy:

```python
# Key metrics to monitor:
- Your salience score over time
- Correlation with other miners (avoid high correlation)
- Prediction accuracy on validation data
- Feature importance in your own models
- Market regime performance (bull vs bear)
```

## Getting Started

1. **Start Simple**: Use the `simple_miner.py` as a baseline
2. **Add Uniqueness**: Incorporate one unique data source
3. **Validate Locally**: Test your signals on historical data
4. **Deploy and Monitor**: Track your performance
5. **Iterate and Improve**: Continuously refine your approach

Remember: The goal is not just to predict prices, but to provide **unique predictive value** that other miners aren't already providing. Innovation and creativity are your biggest advantages!