# Individual Selection - The First Hurdle in MANTIS

The individual selection process is the **most critical bottleneck** in the MANTIS salience calculation. Let me break down exactly how it works and why many miners fail here.

## 🎯 **What Is Individual Selection?**

Before the validator even considers your contribution to the ensemble, it first tests whether your embeddings alone can predict price movements. This is done using **individual logistic regression models**.

## 📊 **Step-by-Step Process**

### Step 1: Individual Model Training
```python
# From model.py - for each miner j
sel_auc = np.zeros(H, dtype=np.float32)  # H = number of hotkeys
for j in range(H):
    if first_nz_idx[j] >= sel_fit_end:  # Skip if no data
        sel_auc[j] = 0.5
        continue
    
    # Get ONLY miner j's embeddings
    Xi_fit = X[:sel_fit_end, j, :].astype(np.float32, copy=False)
    yi_fit = y_bin[:sel_fit_end]  # Binary price direction (up/down)
    
    # Train individual logistic regression
    clf = LogisticRegression(
        penalty="l2",
        C=0.5,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=200,
    )
    clf.fit(Xi_fit, yi_fit)
    
    # Test on validation period
    Xi_eval = X[sel_eval_start:sel_eval_end, j, :].astype(np.float32, copy=False)
    yi_eval = y_bin[sel_eval_start:sel_eval_end]
    
    # Calculate AUC score
    scores = clf.decision_function(Xi_eval)
    sel_auc[j] = float(roc_auc_score(yi_eval, scores))
```

**Key Points:**
- Each miner gets their own separate logistic regression model
- Model uses ONLY that miner's embeddings as features
- Target is binary: will price go up (1) or down (0) in the next hour?
- Performance measured by AUC (Area Under ROC Curve)

### Step 2: Top-K Selection
```python
# Select only the best performers
top_k = min(TOP_K, H)  # TOP_K = 20
selected_idx = np.argsort(-sel_auc)[:top_k]
selected_idx.sort()
```

**The Brutal Reality:**
- Only top 20 miners per time window get selected
- If you're not in top 20, your salience = 0 (no rewards)
- This happens for EVERY time window independently

## 🔍 **What Does AUC Mean?**

AUC (Area Under ROC Curve) measures how well your embeddings predict price direction:

- **AUC = 0.5**: Random guessing (no predictive power)
- **AUC = 0.6**: Weak but positive signal
- **AUC = 0.7**: Good predictive power
- **AUC = 0.8**: Strong predictive power
- **AUC = 0.9+**: Exceptional (rare in real markets)

## 📈 **Example Scenarios**

### Scenario 1: Strong Individual Miner
```python
# Miner with good technical analysis
embeddings = [rsi_signal, macd_signal, bb_position, momentum, ...]
# Individual AUC = 0.68 → Selected for ensemble
```

### Scenario 2: Weak Individual Miner
```python
# Miner with poor/random signals
embeddings = [random_noise, irrelevant_data, ...]
# Individual AUC = 0.52 → Not selected, salience = 0
```

### Scenario 3: Good Signal, Wrong Timing
```python
# Miner with signals that predict 4-hour moves, not 1-hour
embeddings = [long_term_trend, weekly_momentum, ...]
# Individual AUC = 0.48 → Not selected (predicts wrong timeframe)
```

## ⚠️ **Common Failure Modes**

### 1. **Random Noise Submissions**
```python
# This will fail individual selection
embeddings = [random.uniform(-1, 1) for _ in range(100)]
# AUC ≈ 0.5 → Never selected
```

### 2. **Wrong Timeframe Focus**
```python
# Predicting daily moves when target is 1-hour
embeddings = [monthly_rsi, weekly_trend, daily_momentum]
# AUC < 0.5 → Never selected
```

### 3. **Stale/Delayed Data**
```python
# Using yesterday's signals for today's prediction
embeddings = [yesterday_sentiment, old_technical_signals]
# AUC ≈ 0.5 → Never selected
```

### 4. **Over-Smoothed Signals**
```python
# Signals that change too slowly to predict hourly moves
embeddings = [30_day_moving_avg, quarterly_trend]
# AUC ≈ 0.5 → Never selected
```

## 🎯 **How to Pass Individual Selection**

### ✅ **Focus on 1-Hour Prediction Horizon**
```python
# Signals that predict next 1-hour price movement
embeddings = [
    short_term_rsi,           # 14-period on 5min data
    momentum_1h,              # 1-hour momentum
    order_book_imbalance,     # Current bid/ask pressure
    recent_volume_spike,      # Volume in last 30min
    micro_trend_signal        # Trend over last 2 hours
]
```

### ✅ **Use High-Frequency, Responsive Indicators**
```python
# Fast-reacting technical indicators
embeddings = [
    rsi_7,                    # 7-period RSI (faster than 14)
    ema_cross_signal,         # EMA crossover (responsive)
    bollinger_squeeze,        # Volatility breakout signal
    williams_r,               # Fast oscillator
    rate_of_change_1h         # 1-hour rate of change
]
```

### ✅ **Include Market Microstructure**
```python
# Real-time market structure signals
embeddings = [
    bid_ask_spread,           # Current spread tightness
    order_book_depth,         # Liquidity at key levels
    trade_size_distribution,  # Large vs small trades
    price_impact_measure,     # How much volume moves price
    tick_direction_bias       # Recent tick up/down bias
]
```

### ✅ **Validate Your Individual Performance**
```python
# Test your embeddings before deploying
def test_individual_performance(embeddings_history, price_returns):
    X = np.array(embeddings_history)
    y = (np.array(price_returns) > 0).astype(int)
    
    clf = LogisticRegression(penalty="l2", C=0.5, class_weight="balanced")
    clf.fit(X[:-100], y[:-100])  # Train on earlier data
    
    scores = clf.decision_function(X[-100:])  # Test on recent data
    auc = roc_auc_score(y[-100:], scores)
    
    print(f"Individual AUC: {auc:.3f}")
    return auc > 0.55  # Aim for at least 0.55 to be competitive
```

## 🔬 **Technical Deep Dive**

### The Logistic Regression Model
```python
# What the validator actually does
clf = LogisticRegression(
    penalty="l2",        # L2 regularization prevents overfitting
    C=0.5,              # Regularization strength (higher = less regularization)
    class_weight="balanced",  # Handle imbalanced up/down samples
    solver="lbfgs",     # Optimization algorithm
    max_iter=200,       # Maximum iterations
)
```

### The Training/Validation Split
```python
# Time-based split (no data leakage)
sel_fit_end = max(0, sel_eval_start - EMBARGO_IDX)  # EMBARGO_IDX = LAG = 60
# Train: blocks 0 to sel_fit_end
# Gap: sel_fit_end to sel_eval_start (prevents leakage)
# Test: sel_eval_start to sel_eval_end
```

### The AUC Calculation
```python
# Decision function gives continuous scores
scores = clf.decision_function(Xi_eval)  # Raw logistic regression scores
# AUC measures ranking quality (not just accuracy)
sel_auc[j] = float(roc_auc_score(yi_eval, scores))
```

## 📊 **Real Performance Benchmarks**

Based on the validator code and market reality:

- **AUC > 0.65**: Excellent, almost guaranteed selection
- **AUC 0.60-0.65**: Good, likely selected in most windows
- **AUC 0.55-0.60**: Marginal, sometimes selected
- **AUC 0.50-0.55**: Weak, rarely selected
- **AUC < 0.50**: Poor, never selected

## 🚨 **Why Most Miners Fail Here**

1. **Wrong Timeframe**: Optimizing for daily/weekly moves instead of 1-hour
2. **Lagged Signals**: Using yesterday's news to predict today's price
3. **Over-Smoothing**: Signals that change too slowly
4. **No Validation**: Not testing individual performance before deploying
5. **Random Noise**: Hoping randomness will somehow work
6. **Copy-Paste**: Using the same indicators everyone else uses

## 💡 **Pro Tips for Passing Selection**

1. **Test Locally First**: Always validate your individual AUC before deploying
2. **Focus on Speed**: Use fast-reacting, high-frequency signals
3. **Monitor Performance**: Track which of your signals work best
4. **Iterate Quickly**: Adjust based on selection success rate
5. **Think Microstructure**: Market internals often predict short-term moves better than macro trends

Remember: **If you can't pass individual selection, nothing else matters.** This is the absolute prerequisite for earning any rewards in MANTIS!