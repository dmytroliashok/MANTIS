#!/usr/bin/env python3
"""
Advanced MANTIS Miner - Multi-Source Intelligence System

This miner combines multiple data sources and advanced techniques to generate
high-quality embeddings for all MANTIS assets. It's designed to maximize
salience scores by providing unique, predictive signals.
"""

import asyncio
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from textblob import TextBlob
import talib
from scipy import stats
from scipy.signal import savgol_filter
import aiohttp

from timelock import Timelock
from config import ASSETS, ASSET_EMBEDDING_DIMS
from deploy_to_r2 import R2Deployer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Drand configuration
DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

class DataSource:
    """Base class for data sources."""
    
    def __init__(self, name: str):
        self.name = name
        self.cache = {}
        self.last_update = {}
    
    async def get_data(self, asset: str) -> Dict:
        """Override in subclasses."""
        raise NotImplementedError

class TechnicalAnalysisSource(DataSource):
    """Technical analysis indicators."""
    
    def __init__(self):
        super().__init__("technical")
        self.lookback_periods = [5, 10, 20, 50, 100]
    
    async def get_data(self, asset: str) -> Dict:
        try:
            # Map MANTIS assets to Yahoo Finance symbols
            symbol_map = {
                "BTC": "BTC-USD",
                "ETH": "ETH-USD", 
                "EURUSD": "EURUSD=X",
                "GBPUSD": "GBPUSD=X",
                "CADUSD": "CADUSD=X",
                "NZDUSD": "NZDUSD=X",
                "CHFUSD": "CHFUSD=X",
                "XAUUSD": "GC=F",  # Gold futures
                "XAGUSD": "SI=F"   # Silver futures
            }
            
            symbol = symbol_map.get(asset, f"{asset}-USD")
            
            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo", interval="1h")
            
            if hist.empty:
                return {"error": "No data available"}
            
            close = hist['Close'].values
            high = hist['High'].values
            low = hist['Low'].values
            volume = hist['Volume'].values if 'Volume' in hist.columns else np.ones_like(close)
            
            indicators = {}
            
            # Price-based indicators
            indicators['rsi_14'] = talib.RSI(close, timeperiod=14)[-1] / 100.0 - 0.5
            indicators['rsi_7'] = talib.RSI(close, timeperiod=7)[-1] / 100.0 - 0.5
            
            # Moving averages and trends
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            indicators['sma_trend'] = np.tanh((close[-1] - sma_20[-1]) / sma_20[-1] * 10)
            indicators['sma_cross'] = np.tanh((sma_20[-1] - sma_50[-1]) / sma_50[-1] * 20)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            bb_position = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) - 0.5
            indicators['bb_position'] = np.clip(bb_position, -1, 1)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            indicators['macd_signal'] = np.tanh(macd_hist[-1] * 1000)
            
            # Volatility indicators
            atr = talib.ATR(high, low, close, timeperiod=14)
            indicators['volatility'] = np.tanh((atr[-1] / close[-1]) * 100 - 0.02)
            
            # Volume indicators (if available)
            if not np.all(volume == 1):
                volume_sma = talib.SMA(volume, timeperiod=20)
                indicators['volume_trend'] = np.tanh((volume[-1] / volume_sma[-1] - 1) * 2)
            else:
                indicators['volume_trend'] = 0.0
            
            # Price momentum
            momentum_5 = (close[-1] / close[-6] - 1) * 100
            momentum_20 = (close[-1] / close[-21] - 1) * 100
            indicators['momentum_5'] = np.tanh(momentum_5)
            indicators['momentum_20'] = np.tanh(momentum_20 / 2)
            
            # Support/Resistance levels
            recent_high = np.max(high[-50:])
            recent_low = np.min(low[-50:])
            resistance_distance = (recent_high - close[-1]) / close[-1]
            support_distance = (close[-1] - recent_low) / close[-1]
            indicators['resistance_proximity'] = np.tanh(resistance_distance * 20)
            indicators['support_proximity'] = np.tanh(support_distance * 20)
            
            # Stochastic oscillator
            slowk, slowd = talib.STOCH(high, low, close)
            indicators['stoch_signal'] = (slowk[-1] / 100.0 - 0.5) * 2
            
            # Williams %R
            willr = talib.WILLR(high, low, close, timeperiod=14)
            indicators['williams_r'] = willr[-1] / 100.0 + 0.5
            
            return indicators
            
        except Exception as e:
            logger.warning(f"Technical analysis failed for {asset}: {e}")
            return {"error": str(e)}

class SentimentAnalysisSource(DataSource):
    """News and social sentiment analysis."""
    
    def __init__(self):
        super().__init__("sentiment")
        self.news_cache_duration = 300  # 5 minutes
    
    async def get_data(self, asset: str) -> Dict:
        try:
            # Check cache
            cache_key = f"{asset}_sentiment"
            if (cache_key in self.cache and 
                time.time() - self.last_update.get(cache_key, 0) < self.news_cache_duration):
                return self.cache[cache_key]
            
            sentiment_data = {}
            
            # Get news headlines (using a free news API or web scraping)
            headlines = await self._get_news_headlines(asset)
            
            if headlines:
                sentiments = [TextBlob(headline).sentiment.polarity for headline in headlines]
                sentiment_data['news_sentiment'] = np.tanh(np.mean(sentiments) * 2)
                sentiment_data['news_volatility'] = np.tanh(np.std(sentiments) * 5)
                sentiment_data['news_count'] = min(len(headlines) / 10.0, 1.0) - 0.5
            else:
                sentiment_data['news_sentiment'] = 0.0
                sentiment_data['news_volatility'] = 0.0
                sentiment_data['news_count'] = -0.5
            
            # Fear & Greed Index proxy (for crypto assets)
            if asset in ['BTC', 'ETH']:
                fear_greed = await self._get_fear_greed_proxy()
                sentiment_data['fear_greed'] = fear_greed
            else:
                sentiment_data['fear_greed'] = 0.0
            
            # Cache results
            self.cache[cache_key] = sentiment_data
            self.last_update[cache_key] = time.time()
            
            return sentiment_data
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed for {asset}: {e}")
            return {"news_sentiment": 0.0, "news_volatility": 0.0, "news_count": -0.5, "fear_greed": 0.0}
    
    async def _get_news_headlines(self, asset: str) -> List[str]:
        """Get recent news headlines for an asset."""
        try:
            # This is a simplified example - you'd want to use a proper news API
            search_terms = {
                "BTC": "bitcoin cryptocurrency",
                "ETH": "ethereum cryptocurrency", 
                "EURUSD": "euro dollar forex",
                "GBPUSD": "pound dollar forex",
                "XAUUSD": "gold price",
                "XAGUSD": "silver price"
            }
            
            term = search_terms.get(asset, asset)
            
            # Placeholder - replace with actual news API
            # For now, return some sample headlines
            sample_headlines = [
                f"{asset} shows strong momentum in recent trading",
                f"Market analysts bullish on {asset} outlook",
                f"{asset} faces resistance at key levels"
            ]
            
            return sample_headlines[:3]  # Limit to avoid rate limits
            
        except Exception as e:
            logger.warning(f"News headline fetch failed: {e}")
            return []
    
    async def _get_fear_greed_proxy(self) -> float:
        """Get a proxy for fear & greed sentiment."""
        try:
            # This could be VIX for traditional markets, or crypto fear & greed index
            # For now, return a random walk around neutral
            return np.random.normal(0, 0.3)
        except:
            return 0.0

class MacroEconomicSource(DataSource):
    """Macroeconomic indicators and cross-asset correlations."""
    
    def __init__(self):
        super().__init__("macro")
        self.macro_cache_duration = 3600  # 1 hour
    
    async def get_data(self, asset: str) -> Dict:
        try:
            cache_key = f"{asset}_macro"
            if (cache_key in self.cache and 
                time.time() - self.last_update.get(cache_key, 0) < self.macro_cache_duration):
                return self.cache[cache_key]
            
            macro_data = {}
            
            # Get correlation with major indices
            correlations = await self._get_asset_correlations(asset)
            macro_data.update(correlations)
            
            # Time-based features
            now = datetime.now()
            macro_data['hour_of_day'] = np.sin(2 * np.pi * now.hour / 24)
            macro_data['day_of_week'] = np.sin(2 * np.pi * now.weekday() / 7)
            macro_data['day_of_month'] = np.sin(2 * np.pi * now.day / 31)
            
            # Market session indicators
            macro_data['us_session'] = 1.0 if 14 <= now.hour <= 21 else -1.0  # UTC
            macro_data['asia_session'] = 1.0 if 0 <= now.hour <= 8 else -1.0
            macro_data['europe_session'] = 1.0 if 7 <= now.hour <= 16 else -1.0
            
            # Cache results
            self.cache[cache_key] = macro_data
            self.last_update[cache_key] = time.time()
            
            return macro_data
            
        except Exception as e:
            logger.warning(f"Macro analysis failed for {asset}: {e}")
            return {"correlation_spy": 0.0, "correlation_vix": 0.0, "hour_of_day": 0.0}
    
    async def _get_asset_correlations(self, asset: str) -> Dict:
        """Calculate correlations with major market indicators."""
        try:
            # Get recent price data for the asset and major indices
            correlations = {}
            
            # Simplified correlation calculation
            # In practice, you'd calculate rolling correlations with SPY, VIX, DXY, etc.
            correlations['correlation_spy'] = np.random.normal(0, 0.3)  # Placeholder
            correlations['correlation_vix'] = np.random.normal(0, 0.3)  # Placeholder
            correlations['correlation_dxy'] = np.random.normal(0, 0.3)  # Placeholder
            
            return correlations
            
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            return {"correlation_spy": 0.0, "correlation_vix": 0.0, "correlation_dxy": 0.0}

class OnChainAnalysisSource(DataSource):
    """On-chain metrics for crypto assets."""
    
    def __init__(self):
        super().__init__("onchain")
        self.onchain_cache_duration = 1800  # 30 minutes
    
    async def get_data(self, asset: str) -> Dict:
        if asset not in ['BTC', 'ETH']:
            return {"onchain_signal": 0.0}
        
        try:
            cache_key = f"{asset}_onchain"
            if (cache_key in self.cache and 
                time.time() - self.last_update.get(cache_key, 0) < self.onchain_cache_duration):
                return self.cache[cache_key]
            
            onchain_data = {}
            
            # Placeholder for on-chain metrics
            # In practice, you'd fetch from APIs like Glassnode, CoinMetrics, etc.
            onchain_data['network_activity'] = np.random.normal(0, 0.2)
            onchain_data['whale_activity'] = np.random.normal(0, 0.3)
            onchain_data['exchange_flows'] = np.random.normal(0, 0.25)
            
            # Cache results
            self.cache[cache_key] = onchain_data
            self.last_update[cache_key] = time.time()
            
            return onchain_data
            
        except Exception as e:
            logger.warning(f"On-chain analysis failed for {asset}: {e}")
            return {"onchain_signal": 0.0}

class AdvancedMiner:
    """Advanced miner that combines multiple data sources."""
    
    def __init__(self, hotkey: str):
        self.hotkey = hotkey
        self.data_sources = [
            TechnicalAnalysisSource(),
            SentimentAnalysisSource(), 
            MacroEconomicSource(),
            OnChainAnalysisSource()
        ]
        
        # Feature engineering components
        self.scalers = {}
        self.pca_components = {}
        self.anomaly_detectors = {}
        
        # Initialize for each asset
        for asset in ASSETS:
            self.scalers[asset] = RobustScaler()
            if ASSET_EMBEDDING_DIMS[asset] > 10:
                self.pca_components[asset] = PCA(n_components=min(50, ASSET_EMBEDDING_DIMS[asset] // 2))
            self.anomaly_detectors[asset] = IsolationForest(contamination=0.1, random_state=42)
    
    async def generate_embeddings(self) -> List[List[float]]:
        """Generate embeddings for all assets."""
        embeddings = []
        
        for asset in ASSETS:
            try:
                embedding = await self._generate_asset_embedding(asset)
                embeddings.append(embedding)
                logger.info(f"Generated embedding for {asset}: dim={len(embedding)}, norm={np.linalg.norm(embedding):.3f}")
            except Exception as e:
                logger.error(f"Failed to generate embedding for {asset}: {e}")
                # Fallback to zero embedding
                embeddings.append([0.0] * ASSET_EMBEDDING_DIMS[asset])
        
        return embeddings
    
    async def _generate_asset_embedding(self, asset: str) -> List[float]:
        """Generate embedding for a single asset."""
        target_dim = ASSET_EMBEDDING_DIMS[asset]
        
        # Collect data from all sources
        all_features = {}
        
        for source in self.data_sources:
            try:
                source_data = await source.get_data(asset)
                if "error" not in source_data:
                    for key, value in source_data.items():
                        all_features[f"{source.name}_{key}"] = value
            except Exception as e:
                logger.warning(f"Data source {source.name} failed for {asset}: {e}")
        
        if not all_features:
            logger.warning(f"No features available for {asset}, using zero embedding")
            return [0.0] * target_dim
        
        # Convert to feature vector
        feature_names = sorted(all_features.keys())
        feature_vector = np.array([all_features[name] for name in feature_names])
        
        # Handle NaN values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply feature engineering
        processed_features = self._apply_feature_engineering(asset, feature_vector)
        
        # Resize to target dimension
        final_embedding = self._resize_to_target_dim(processed_features, target_dim)
        
        # Ensure values are in [-1, 1] range
        final_embedding = np.clip(final_embedding, -1.0, 1.0)
        
        return final_embedding.tolist()
    
    def _apply_feature_engineering(self, asset: str, features: np.ndarray) -> np.ndarray:
        """Apply advanced feature engineering techniques."""
        try:
            # Robust scaling
            features_scaled = features.reshape(1, -1)
            
            # Add polynomial features for non-linear relationships
            poly_features = []
            for i in range(len(features)):
                for j in range(i, len(features)):
                    poly_features.append(features[i] * features[j])
            
            # Add trigonometric transformations
            trig_features = []
            for f in features:
                trig_features.extend([np.sin(f * np.pi), np.cos(f * np.pi)])
            
            # Combine all features
            all_features = np.concatenate([
                features,
                np.array(poly_features[:len(features)]),  # Limit polynomial features
                np.array(trig_features[:len(features)])   # Limit trig features
            ])
            
            # Apply smoothing to reduce noise
            if len(all_features) > 5:
                all_features = savgol_filter(all_features, 
                                           window_length=min(5, len(all_features)), 
                                           polyorder=2)
            
            return all_features
            
        except Exception as e:
            logger.warning(f"Feature engineering failed for {asset}: {e}")
            return features
    
    def _resize_to_target_dim(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """Resize feature vector to target dimension."""
        if len(features) == target_dim:
            return features
        elif len(features) > target_dim:
            # Use PCA or truncation for dimension reduction
            if target_dim > 10 and len(features) > target_dim * 2:
                # Use PCA for significant dimension reduction
                pca = PCA(n_components=target_dim)
                return pca.fit_transform(features.reshape(1, -1)).flatten()
            else:
                # Simple truncation with importance weighting
                importance_weights = np.abs(features)
                top_indices = np.argsort(importance_weights)[-target_dim:]
                return features[top_indices]
        else:
            # Pad with engineered features
            padding_size = target_dim - len(features)
            
            # Create meaningful padding
            padding = []
            for i in range(padding_size):
                if i < len(features):
                    # Use combinations of existing features
                    idx1, idx2 = i % len(features), (i + 1) % len(features)
                    padding.append(np.tanh(features[idx1] * features[idx2]))
                else:
                    # Use statistical transformations
                    padding.append(np.tanh(np.mean(features) + np.random.normal(0, 0.1)))
            
            return np.concatenate([features, padding])

class MinerDeployer:
    """Handles deployment of miner embeddings."""
    
    def __init__(self, hotkey: str):
        self.hotkey = hotkey
        self.miner = AdvancedMiner(hotkey)
        self.deployer = R2Deployer()
        self.tlock = Timelock(DRAND_PUBLIC_KEY)
    
    async def mine_and_deploy(self) -> str:
        """Generate embeddings and deploy to public URL."""
        try:
            logger.info("🧠 Generating advanced embeddings...")
            embeddings = await self.miner.generate_embeddings()
            
            # Validate embeddings
            self._validate_embeddings(embeddings)
            
            logger.info("🔐 Encrypting payload...")
            payload = await self._encrypt_payload(embeddings)
            
            logger.info("☁️ Deploying to R2...")
            public_url = self.deployer.upload_to_r2(payload, self.hotkey)
            
            logger.info(f"✅ Successfully deployed to: {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"❌ Mining and deployment failed: {e}")
            raise
    
    def _validate_embeddings(self, embeddings: List[List[float]]):
        """Validate embedding format and constraints."""
        if len(embeddings) != len(ASSETS):
            raise ValueError(f"Expected {len(ASSETS)} embeddings, got {len(embeddings)}")
        
        for i, (asset, embedding) in enumerate(zip(ASSETS, embeddings)):
            expected_dim = ASSET_EMBEDDING_DIMS[asset]
            if len(embedding) != expected_dim:
                raise ValueError(f"Wrong dimension for {asset}: got {len(embedding)}, expected {expected_dim}")
            
            for j, val in enumerate(embedding):
                if not isinstance(val, (int, float)) or not (-1.0 <= val <= 1.0):
                    raise ValueError(f"Invalid value in {asset}[{j}]: {val}")
    
    async def _encrypt_payload(self, embeddings: List[List[float]]) -> Dict:
        """Encrypt embeddings with time-lock encryption."""
        try:
            # Get Drand beacon info
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10) as resp:
                    info = await resp.json()
            
            future_time = time.time() + 30  # 30 seconds in future
            target_round = int((future_time - info["genesis_time"]) // info["period"])
            
            # Create plaintext with hotkey verification
            plaintext = f"{str(embeddings)}:::{self.hotkey}"
            
            # Encrypt
            salt = secrets.token_bytes(32)
            ciphertext_hex = self.tlock.tle(target_round, plaintext, salt).hex()
            
            return {
                "round": target_round,
                "ciphertext": ciphertext_hex
            }
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

async def main():
    """Main entry point for the advanced miner."""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Advanced MANTIS Miner")
    parser.add_argument("--hotkey", default=os.getenv("MY_HOTKEY"), help="Your hotkey")
    parser.add_argument("--interval", type=int, default=300, help="Mining interval in seconds")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    
    args = parser.parse_args()
    
    if not args.hotkey or args.hotkey == "5D...":
        logger.error("❌ Please set your hotkey via --hotkey or MY_HOTKEY environment variable")
        return
    
    logger.info(f"🚀 Starting Advanced MANTIS Miner")
    logger.info(f"🔑 Hotkey: {args.hotkey}")
    logger.info(f"⏱️ Interval: {args.interval} seconds")
    
    deployer = MinerDeployer(args.hotkey)
    
    if args.continuous:
        # Continuous mining
        while True:
            try:
                await deployer.mine_and_deploy()
                logger.info(f"😴 Sleeping for {args.interval} seconds...")
                await asyncio.sleep(args.interval)
            except KeyboardInterrupt:
                logger.info("🛑 Stopping miner...")
                break
            except Exception as e:
                logger.error(f"❌ Error in mining loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    else:
        # Single run
        await deployer.mine_and_deploy()

if __name__ == "__main__":
    asyncio.run(main())