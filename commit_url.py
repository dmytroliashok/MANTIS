#!/usr/bin/env python3
"""
One-time script to commit your public URL to the Bittensor subnet.
Run this once after setting up your deployment infrastructure.
"""

import bittensor as bt
from dotenv import load_dotenv
import os

load_dotenv()

def commit_url_to_subnet():
    """Commit your public URL to the subnet."""
    
    # Load configuration
    wallet_name = os.getenv("WALLET_NAME")
    wallet_hotkey = os.getenv("WALLET_HOTKEY") 
    my_hotkey = os.getenv("MY_HOTKEY")
    bucket_name = os.getenv("R2_BUCKET_NAME")
    
    if not all([wallet_name, wallet_hotkey, my_hotkey]):
        print("❌ Missing configuration. Check your .env file.")
        return
    
    # Construct your public URL
    if bucket_name:
        # R2 URL
        public_url = f"https://{bucket_name}.r2.dev/{my_hotkey}"
    else:
        # Custom server URL (modify as needed)
        server_base = input("Enter your server base URL (e.g., https://myserver.com): ").strip()
        public_url = f"{server_base}/{my_hotkey}"
    
    print(f"🔗 Public URL: {public_url}")
    
    # Initialize Bittensor components
    try:
        wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        subtensor = bt.subtensor(network="finney")
        
        print(f"👛 Using wallet: {wallet_name}/{wallet_hotkey}")
        print(f"🔑 Hotkey: {my_hotkey}")
        
        # Commit the URL
        print("📡 Committing URL to subnet...")
        result = subtensor.commit(
            wallet=wallet,
            netuid=123,  # MANTIS netuid
            data=public_url
        )
        
        if result:
            print("✅ Successfully committed URL to subnet!")
            print(f"📍 Your committed URL: {public_url}")
            print("\n📝 Next steps:")
            print("1. Make sure your deployment script runs regularly")
            print("2. Monitor your miner's performance")
            print("3. Update your embeddings frequently")
        else:
            print("❌ Failed to commit URL to subnet")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    commit_url_to_subnet()