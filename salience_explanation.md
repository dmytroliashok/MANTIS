# MANTIS Salience Calculation - Deep Dive

The salience calculation is the heart of MANTIS that determines how much each miner gets rewarded. Let me break down exactly how it works based on the validator code.

## 🎯 **Core Concept: Counterfactual Analysis**

The validator uses **permutation importance** with **counterfactual modeling**:
1. Train a model on ALL miners' data
2. For each miner, remove their data and retrain
3. Measure how much the model performance degrades
4. The degradation = that miner's salience (importance)

## 📊 **Step-by-Step Process**

### Step 1: Data Preparation
```python
# From model.py - the validator processes each asset separately
for asset_name, (hist, asset_returns) in training_data.items():
    X_flat, hk2idx = hist  # X_flat: embeddings, hk2idx: hotkey->index mapping
    y = asset_returns      # Price returns (target to predict)
```

**Key Points:**
- Each asset is processed independently
- `X_flat` contains all miners' embeddings flattened: shape `(timesteps, num_hotkeys * embedding_dim)`
- `y` contains the corresponding price returns to predict

### Step 2: Walk-Forward Validation
```python
# The validator uses walk-forward windows
indices: List[Tuple[int, int, int]] = []
start = 0
while True:
    val_start_idx = start + LAG  # LAG = 60 blocks
    if val_start_idx >= T:
        break
    end_idx = min(start + CHUNK_SIZE, T)  # CHUNK_SIZE = 2000
    indices.append((start, val_start_idx, end_idx))
    start = end_idx
```

**This creates time windows:**
- Training: blocks 0-1940
- Validation: blocks 2000-4000
- Next training: blocks 2000-3940
- Next validation: blocks 4000-6000
- And so on...

### Step 3: Miner Selection (Top-K)
```python
# For each window, select top miners based on individual AUC
sel_auc = np.zeros(H, dtype=np.float32)  # H = number of hotkeys
for j in range(H):
    # Train individual logistic regression for each miner
    clf = LogisticRegression(penalty="l2", C=0.5, class_weight="balanced")
    clf.fit(Xi_fit, yi_fit)  # Xi_fit = miner j's embeddings
    scores = clf.decision_function(Xi_eval)
    sel_auc[j] = roc_auc_score(yi_eval, scores)

# Select top K miners (K=20 by default)
top_k = min(TOP_K, H)
selected_idx = np.argsort(-sel_auc)[:top_k]
```

**Why This Matters:**
- Only the top 20 miners per window are included in the ensemble
- This prevents weak/noisy miners from diluting the model
- Your individual predictive power determines if you're selected

### Step 4: Ensemble Model Training
```python
# Train XGBoost on selected miners' combined data
dtrain = xgb.DMatrix(X_train_sel, label=y_train_head)
bst = xgb.train(xgb_params, dtrain, num_boost_round=250)

# Get baseline performance
dval = xgb.DMatrix(X_val_sel, label=y_val)
base_probs = bst.predict(dval)
base_auc = roc_auc_score(y_val, base_probs)
```

**The Model:**
- XGBoost with 250 rounds
- Uses all selected miners' features together
- Optimized for AUC (Area Under ROC Curve)

### Step 5: Permutation Importance (The Key!)
```python
window_imp = np.zeros(H, dtype=np.float32)
for local_col, j in enumerate(selected_idx):
    # Save original column
    col = X_val_sel[:, local_col].copy()
    
    # PERMUTE the miner's data (shuffle randomly)
    perm_idx = np.random.permutation(col.shape[0])
    X_val_sel[:, local_col] = col[perm_idx]
    
    # Retrain and measure performance drop
    dval_perm = xgb.DMatrix(X_val_sel, label=y_val)
    perm_probs = bst.predict(dval_perm)
    perm_auc = roc_auc_score(y_val, perm_probs)
    
    # Calculate importance as performance drop
    delta = base_auc - perm_auc
    window_imp[j] = delta if delta > 0.0 else 0.0
    
    # Restore original data
    X_val_sel[:, local_col] = col
```

**This is the Magic:**
- Permutation breaks the relationship between miner's data and target
- If AUC drops significantly → miner was important
- If AUC barely changes → miner was redundant/useless

### Step 6: Scaling by Model Quality
```python
# Scale importance by how good the overall model is
scale = max((base_auc - 0.5) / 0.5, 0.0)
if scale <= 0:
    window_imp[:] = 0.0  # If model is no better than random, no rewards
else:
    window_imp *= scale  # Better models give higher absolute rewards
```

**Why This Matters:**
- If the ensemble model is bad (AUC ≤ 0.5), nobody gets rewards
- Better ensemble models amplify individual contributions
- Encourages miners to submit data that improves overall prediction

### Step 7: Recency Weighting
```python
# More recent windows get higher weight
WINDOWS_HALF_LIFE = 10
recency_gamma = 0.5 ** (1.0 / WINDOWS_HALF_LIFE)
w = recency_gamma ** (max(0, len(indices) - 1 - window_index))

# Accumulate weighted importance
total_hk_imp += (w * window_imp).astype(np.float32)
total_weight += w
```

**Recency Matters:**
- Recent performance is weighted more heavily
- Half-life of 10 windows means older performance decays
- Encourages consistent, recent good performance

### Step 8: Multi-Asset Aggregation
```python
# Each asset contributes to final salience
asset_contrib = {a: sum(max(0.0, v) for v in imp.values()) 
                for a, imp in asset_hotkey_importance.items()}
total_contrib = sum(asset_contrib.values())

# Final salience is sum across all assets
final_imp = {hk: 0.0 for hk in all_hotkeys}
for imp in asset_hotkey_importance.values():
    for hk, score in imp.items():
        final_imp[hk] += max(0.0, score)
```

**Multi-Asset Strategy:**
- Your total salience = sum of salience across all assets
- BTC (100 dims) can contribute more than forex (2 dims each)
- But being good at multiple assets is valuable

## 🎯 **What This Means for Miners**

### ✅ **To Maximize Salience:**

1. **Be Unique**: If your signal is similar to others, permuting it won't hurt the model much
2. **Be Predictive**: Your individual AUC must be good enough to get selected (top 20)
3. **Be Consistent**: Recent performance matters more than old performance
4. **Be Multi-Asset**: Contribute to multiple assets, not just BTC
5. **Help the Ensemble**: Your data should make the combined model better

### ❌ **What Hurts Your Salience:**

1. **Redundancy**: If others already provide your signal, you get zero
2. **Noise**: Random data that doesn't help prediction gets zero
3. **Inconsistency**: Good performance followed by bad performance
4. **Single Asset Focus**: Missing opportunities in other assets
5. **Poor Individual Performance**: Not making it into the top-20 selection

## 🔬 **Mathematical Example**

Let's say there are 3 miners for BTC:
- **Miner A**: Technical analysis (RSI, MACD, etc.)
- **Miner B**: Sentiment analysis (news, social media)  
- **Miner C**: Copy of Miner A's strategy

**Scenario 1: All 3 selected**
- Ensemble AUC with all: 0.65
- Remove Miner A: AUC drops to 0.58 → Salience = 0.07
- Remove Miner B: AUC drops to 0.60 → Salience = 0.05  
- Remove Miner C: AUC drops to 0.64 → Salience = 0.01 (redundant!)

**Scenario 2: Miner C not selected (poor individual AUC)**
- Only A and B in ensemble
- Remove Miner A: AUC drops to 0.55 → Salience = 0.10
- Remove Miner B: AUC drops to 0.58 → Salience = 0.07
- Miner C: Salience = 0.00 (not selected)

## 🚀 **Advanced Insights**

### The Selection Bottleneck
```python
top_k = min(TOP_K, H)  # TOP_K = 20
selected_idx = np.argsort(-sel_auc)[:top_k]
```
- Only top 20 miners per window get included
- Your individual predictive power must be in top 20
- This is the first hurdle - many miners fail here

### The Ensemble Effect
- Even if you're selected, your salience depends on uniqueness
- The XGBoost model learns to combine all features optimally
- If your features are redundant, the model ignores them

### The Scaling Factor
```python
scale = max((base_auc - 0.5) / 0.5, 0.0)
```
- If ensemble AUC = 0.6, scale = 0.2
- If ensemble AUC = 0.7, scale = 0.4  
- If ensemble AUC = 0.8, scale = 0.6

**Higher ensemble performance amplifies everyone's rewards!**

## 💡 **Strategic Implications**

1. **Cooperation vs Competition**: You want the overall ensemble to be good (higher scale factor), but you also want to be uniquely valuable within it

2. **Niche Specialization**: Being the best at one unique signal type might be better than being mediocre at everything

3. **Multi-Asset Diversification**: Since assets are processed separately, you can specialize in different assets

4. **Temporal Consistency**: The recency weighting means you need sustained good performance, not just occasional spikes

This system brilliantly incentivizes genuine innovation and predictive value while penalizing redundancy and noise!