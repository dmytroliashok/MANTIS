#!/usr/bin/env python3
"""
Deploy encrypted embeddings to Cloudflare R2 bucket.
This is the recommended approach for MANTIS miners.
"""

import json
import os
import secrets
import time
from typing import List
import boto3
from botocore.client import Config
from dotenv import load_dotenv
import requests

from timelock import Timelock
from config import ASSETS, ASSET_EMBEDDING_DIMS

# Load environment variables
load_dotenv()

# Drand configuration (DO NOT CHANGE)
DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

class R2Deployer:
    def __init__(self):
        # Load R2 credentials from environment
        self.account_id = os.getenv("R2_ACCOUNT_ID")
        self.access_key_id = os.getenv("R2_ACCESS_KEY_ID") 
        self.secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("R2_BUCKET_NAME")
        
        if not all([self.account_id, self.access_key_id, self.secret_access_key, self.bucket_name]):
            raise ValueError("Missing R2 credentials. Check your .env file.")
        
        # Initialize R2 client
        self.endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="auto",
            config=Config(signature_version="s3v4")
        )
        
        self.tlock = Timelock(DRAND_PUBLIC_KEY)
    
    def generate_embeddings(self) -> List[List[float]]:
        """
        Generate your embeddings here. Replace this with your actual model/strategy.
        This is just a placeholder that generates random embeddings.
        """
        embeddings = []
        for asset in ASSETS:
            dim = ASSET_EMBEDDING_DIMS[asset]
            # TODO: Replace with your actual prediction logic
            embedding = [0.0] * dim  # All zeros = no signal
            # Example: embedding = your_model.predict(asset)
            embeddings.append(embedding)
        return embeddings
    
    def encrypt_payload(self, embeddings: List[List[float]], hotkey: str, lock_time_seconds: int = 30) -> dict:
        """Encrypt embeddings with time-lock encryption."""
        try:
            # Get Drand beacon info
            info = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10).json()
            future_time = time.time() + lock_time_seconds
            target_round = int((future_time - info["genesis_time"]) // info["period"])
            
            # Create plaintext with hotkey verification
            plaintext = f"{str(embeddings)}:::{hotkey}"
            
            # Encrypt
            salt = secrets.token_bytes(32)
            ciphertext_hex = self.tlock.tle(target_round, plaintext, salt).hex()
            
            return {
                "round": target_round,
                "ciphertext": ciphertext_hex
            }
            
        except Exception as e:
            print(f"❌ Encryption failed: {e}")
            raise
    
    def upload_to_r2(self, payload: dict, hotkey: str) -> str:
        """Upload encrypted payload to R2 bucket."""
        try:
            # Convert payload to JSON
            payload_json = json.dumps(payload, indent=2)
            
            # Upload to R2 (filename must match hotkey)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=hotkey,
                Body=payload_json.encode('utf-8'),
                ContentType='application/json'
            )
            
            # Return public URL
            public_url = f"https://{self.bucket_name}.r2.dev/{hotkey}"
            return public_url
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            raise
    
    def deploy(self, hotkey: str) -> str:
        """Complete deployment process."""
        print(f"🚀 Starting deployment for hotkey: {hotkey}")
        
        # Step 1: Generate embeddings
        print("📊 Generating embeddings...")
        embeddings = self.generate_embeddings()
        
        # Validate embeddings
        for i, (asset, embedding) in enumerate(zip(ASSETS, embeddings)):
            expected_dim = ASSET_EMBEDDING_DIMS[asset]
            if len(embedding) != expected_dim:
                raise ValueError(f"Wrong dimension for {asset}: got {len(embedding)}, expected {expected_dim}")
            
            # Check value ranges
            for val in embedding:
                if not (-1.0 <= val <= 1.0):
                    raise ValueError(f"Embedding values must be between -1.0 and 1.0, got {val}")
        
        print(f"✅ Generated embeddings for {len(ASSETS)} assets")
        
        # Step 2: Encrypt payload
        print("🔐 Encrypting payload...")
        payload = self.encrypt_payload(embeddings, hotkey)
        print(f"✅ Encrypted for round {payload['round']}")
        
        # Step 3: Upload to R2
        print("☁️ Uploading to R2...")
        public_url = self.upload_to_r2(payload, hotkey)
        print(f"✅ Deployed to: {public_url}")
        
        return public_url


def main():
    """Example usage of the R2 deployer."""
    
    # Your hotkey (replace with your actual hotkey)
    MY_HOTKEY = "5D..."  # TODO: Replace with your hotkey
    
    if MY_HOTKEY == "5D...":
        print("❌ Please set your actual hotkey in the script!")
        return
    
    try:
        deployer = R2Deployer()
        public_url = deployer.deploy(MY_HOTKEY)
        
        print("\n🎉 Deployment successful!")
        print(f"📍 Your payload is now available at: {public_url}")
        print("\n📝 Next steps:")
        print("1. Test the URL in your browser to make sure it's accessible")
        print("2. Commit this URL to the Bittensor subnet (one-time setup)")
        print("3. Run this script regularly to update your predictions")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")


if __name__ == "__main__":
    main()