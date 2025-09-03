#!/usr/bin/env python3
"""
Continuous deployment script that runs your mining operation.
This script should run continuously to keep updating your predictions.
"""

import time
import schedule
import logging
from datetime import datetime
from deploy_to_r2 import R2Deployer
from dotenv import load_dotenv
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('miner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

class ContinuousMiner:
    def __init__(self):
        self.hotkey = os.getenv("MY_HOTKEY")
        if not self.hotkey or self.hotkey == "5D...":
            raise ValueError("Please set your MY_HOTKEY in .env file")
        
        self.deployer = R2Deployer()
        self.deployment_count = 0
    
    def mine_and_deploy(self):
        """Single mining and deployment cycle."""
        try:
            logger.info(f"🚀 Starting deployment cycle #{self.deployment_count + 1}")
            
            # Deploy new embeddings
            public_url = self.deployer.deploy(self.hotkey)
            
            self.deployment_count += 1
            logger.info(f"✅ Deployment #{self.deployment_count} successful")
            logger.info(f"📍 URL: {public_url}")
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
    
    def start_continuous_mining(self, interval_minutes: int = 5):
        """Start continuous mining with specified interval."""
        logger.info(f"🎯 Starting continuous mining (every {interval_minutes} minutes)")
        logger.info(f"🔑 Hotkey: {self.hotkey}")
        
        # Schedule regular deployments
        schedule.every(interval_minutes).minutes.do(self.mine_and_deploy)
        
        # Run initial deployment
        self.mine_and_deploy()
        
        # Keep running
        while True:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds
            except KeyboardInterrupt:
                logger.info("🛑 Stopping continuous mining")
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                time.sleep(60)  # Wait a minute before retrying


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continuous MANTIS mining")
    parser.add_argument(
        "--interval", 
        type=int, 
        default=5, 
        help="Deployment interval in minutes (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        miner = ContinuousMiner()
        miner.start_continuous_mining(args.interval)
    except Exception as e:
        logger.error(f"❌ Failed to start miner: {e}")


if __name__ == "__main__":
    main()