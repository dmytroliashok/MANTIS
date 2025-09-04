#!/usr/bin/env python3
"""
Simple but effective MANTIS miner focused on technical analysis.
Good starting point that can be easily extended.
"""

import asyncio
import json
import logging
import os
import secrets
import time
from typing import List
import numpy as np
import yfinance as yf
import talib
from datetime import datetime

from timelock import Timelock
from config import ASSETS, ASSET_EMBEDDING_DIMS
from deploy_to_r2 import R2Deployer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

class SimpleMiner:
    """Simple miner focused on technical analysis."""
    
    def __init__(self, hotkey: str):
        self.hotkey = hotkey
        self.symbol_map = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD", 
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            "CADUSD": "CADUSD=X",
            "NZDUSD": "NZDUSD=X",
            "CHFUSD": "CHFUSD=X",
            "XAUUSD": "GC=F",
            "XAGUSD": "SI=F"
        }
    
    def generate_embeddings(self) -> List[List[float]]:
        """Generate embeddings for all assets."""
        embeddings = []
        
        for asset in ASSETS:
            try:
                embedding = self._generate_asset_embedding(asset)
                embeddings.append(embedding)
                logger.info(f"Generated {asset}: {len(embedding)} dims, norm={np.linalg.norm(embedding):.3f}")
            except Exception as e:
                logger.error(f"Failed to generate {asset}: {e}")
                embeddings.append([0.0] * ASSET_EMBEDDING_DIMS[asset])
        
        return embeddings
    
    def _generate_asset_embedding(self, asset: str) -> List[float]:
        """Generate embedding for a single asset using technical analysis."""
        target_dim = ASSET_EMBEDDING_DIMS[asset]
        
        try:
            # Get price data
            symbol = self.symbol_map.get(asset, f"{asset}-USD")
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2mo", interval="1h")
            
            if hist.empty:
                return [0.0] * target_dim
            
            close = hist['Close'].values
            high = hist['High'].values
            low = hist['Low'].values
            volume = hist['Volume'].values if 'Volume' in hist.columns else np.ones_like(close)
            
            # Calculate technical indicators
            features = []
            
            # 1. RSI indicators
            rsi_14 = talib.RSI(close, timeperiod=14)[-1]
            features.append((rsi_14 / 100.0 - 0.5) * 2)  # Normalize to [-1, 1]
            
            rsi_7 = talib.RSI(close, timeperiod=7)[-1]
            features.append((rsi_7 / 100.0 - 0.5) * 2)
            
            # 2. Moving average trends
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            
            # Price vs SMA
            price_vs_sma20 = (close[-1] - sma_20[-1]) / sma_20[-1]
            features.append(np.tanh(price_vs_sma20 * 10))
            
            # SMA crossover signal
            sma_cross = (sma_20[-1] - sma_50[-1]) / sma_50[-1]
            features.append(np.tanh(sma_cross * 20))
            
            # 3. Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            bb_position = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
            features.append((bb_position - 0.5) * 2)  # Center around 0
            
            # 4. MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            features.append(np.tanh(macd_hist[-1] * 1000))
            
            # 5. Momentum indicators
            momentum_5 = (close[-1] / close[-6] - 1) * 100
            features.append(np.tanh(momentum_5))
            
            momentum_20 = (close[-1] / close[-21] - 1) * 100
            features.append(np.tanh(momentum_20 / 2))
            
            # 6. Volatility (ATR)
            atr = talib.ATR(high, low, close, timeperiod=14)
            volatility = atr[-1] / close[-1]
            features.append(np.tanh((volatility - 0.02) * 100))
            
            # 7. Volume trend (if available)
            if not np.all(volume == 1):
                volume_sma = talib.SMA(volume, timeperiod=20)
                volume_trend = volume[-1] / volume_sma[-1] - 1
                features.append(np.tanh(volume_trend * 2))
            else:
                features.append(0.0)
            
            # 8. Stochastic oscillator
            slowk, slowd = talib.STOCH(high, low, close)
            stoch_signal = (slowk[-1] / 100.0 - 0.5) * 2
            features.append(stoch_signal)
            
            # 9. Williams %R
            willr = talib.WILLR(high, low, close, timeperiod=14)
            williams_signal = willr[-1] / 100.0 + 0.5  # Convert to [0,1] then to [-0.5,0.5]
            features.append((williams_signal - 0.5) * 2)
            
            # 10. Time-based features
            now = datetime.now()
            hour_signal = np.sin(2 * np.pi * now.hour / 24)
            features.append(hour_signal)
            
            # Convert to numpy array and handle NaN
            features = np.array(features)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Resize to target dimension
            if len(features) >= target_dim:
                # Truncate or use PCA-like selection
                if target_dim == 2:
                    # For 2D assets, use most important signals
                    return [features[0], features[2]]  # RSI and price trend
                else:
                    # For higher dimensions, use all features and pad/truncate
                    return self._resize_features(features, target_dim)
            else:
                # Pad with engineered features
                return self._pad_features(features, target_dim)
            
        except Exception as e:
            logger.warning(f"Technical analysis failed for {asset}: {e}")
            return [0.0] * target_dim
    
    def _resize_features(self, features: np.ndarray, target_dim: int) -> List[float]:
        """Resize feature vector to target dimension."""
        if len(features) == target_dim:
            return features.tolist()
        elif len(features) > target_dim:
            # Select most important features (highest absolute values)
            importance = np.abs(features)
            top_indices = np.argsort(importance)[-target_dim:]
            return features[top_indices].tolist()
        else:
            # This case is handled by _pad_features
            return self._pad_features(features, target_dim)
    
    def _pad_features(self, features: np.ndarray, target_dim: int) -> List[float]:
        """Pad features to target dimension with engineered features."""
        result = features.tolist()
        
        while len(result) < target_dim:
            if len(result) < len(features):
                # Add combinations of existing features
                idx1 = len(result) % len(features)
                idx2 = (len(result) + 1) % len(features)
                new_feature = np.tanh(features[idx1] * features[idx2])
                result.append(new_feature)
            else:
                # Add statistical transformations
                mean_feature = np.mean(features)
                std_feature = np.std(features)
                new_feature = np.tanh(mean_feature + np.random.normal(0, std_feature * 0.1))
                result.append(new_feature)
        
        # Ensure all values are in [-1, 1]
        result = np.clip(result, -1.0, 1.0)
        return result.tolist()

async def main():
    """Main entry point."""
    from dotenv import load_dotenv
    load_dotenv()
    
    hotkey = os.getenv("MY_HOTKEY")
    if not hotkey or hotkey == "5D...":
        logger.error("❌ Please set MY_HOTKEY in your .env file")
        return
    
    logger.info(f"🚀 Starting Simple MANTIS Miner")
    logger.info(f"🔑 Hotkey: {hotkey}")
    
    miner = SimpleMiner(hotkey)
    deployer = R2Deployer()
    
    try:
        # Generate embeddings
        logger.info("📊 Generating embeddings...")
        embeddings = miner.generate_embeddings()
        
        # Deploy
        logger.info("☁️ Deploying to R2...")
        public_url = deployer.deploy(hotkey)
        
        logger.info(f"✅ Successfully deployed to: {public_url}")
        
    except Exception as e:
        logger.error(f"❌ Mining failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())