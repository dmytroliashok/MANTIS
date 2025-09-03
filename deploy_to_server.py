#!/usr/bin/env python3
"""
Deploy encrypted embeddings to your own HTTP server.
Alternative to R2 if you prefer to host your own server.
"""

import json
import os
import secrets
import time
from typing import List
import requests
from pathlib import Path

from timelock import Timelock
from config import ASSETS, ASSET_EMBEDDING_DIMS

# Drand configuration (DO NOT CHANGE)
DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

class ServerDeployer:
    def __init__(self, server_url: str, upload_method: str = "scp"):
        """
        Initialize server deployer.
        
        Args:
            server_url: Base URL of your server (e.g., "https://myserver.com")
            upload_method: "scp", "rsync", or "http_post"
        """
        self.server_url = server_url.rstrip('/')
        self.upload_method = upload_method
        self.tlock = Timelock(DRAND_PUBLIC_KEY)
    
    def generate_embeddings(self) -> List[List[float]]:
        """
        Generate your embeddings here. Replace with your actual model.
        """
        embeddings = []
        for asset in ASSETS:
            dim = ASSET_EMBEDDING_DIMS[asset]
            # TODO: Replace with your actual prediction logic
            embedding = [0.0] * dim  # Placeholder
            embeddings.append(embedding)
        return embeddings
    
    def encrypt_payload(self, embeddings: List[List[float]], hotkey: str) -> dict:
        """Encrypt embeddings with time-lock encryption."""
        try:
            info = requests.get(f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10).json()
            future_time = time.time() + 30  # 30 seconds in future
            target_round = int((future_time - info["genesis_time"]) // info["period"])
            
            plaintext = f"{str(embeddings)}:::{hotkey}"
            salt = secrets.token_bytes(32)
            ciphertext_hex = self.tlock.tle(target_round, plaintext, salt).hex()
            
            return {
                "round": target_round,
                "ciphertext": ciphertext_hex
            }
        except Exception as e:
            print(f"❌ Encryption failed: {e}")
            raise
    
    def upload_via_scp(self, payload: dict, hotkey: str, server_path: str, ssh_key: str = None):
        """Upload using SCP."""
        import subprocess
        
        # Save payload locally first
        local_file = f"/tmp/{hotkey}"
        with open(local_file, 'w') as f:
            json.dump(payload, f, indent=2)
        
        # Build SCP command
        scp_cmd = ["scp"]
        if ssh_key:
            scp_cmd.extend(["-i", ssh_key])
        scp_cmd.extend([local_file, server_path])
        
        try:
            result = subprocess.run(scp_cmd, check=True, capture_output=True, text=True)
            print(f"✅ SCP upload successful")
            os.remove(local_file)  # Cleanup
        except subprocess.CalledProcessError as e:
            print(f"❌ SCP upload failed: {e}")
            raise
    
    def upload_via_http_post(self, payload: dict, hotkey: str, upload_endpoint: str):
        """Upload using HTTP POST."""
        try:
            files = {'file': (hotkey, json.dumps(payload, indent=2), 'application/json')}
            response = requests.post(upload_endpoint, files=files, timeout=30)
            response.raise_for_status()
            print(f"✅ HTTP upload successful")
        except Exception as e:
            print(f"❌ HTTP upload failed: {e}")
            raise
    
    def deploy(self, hotkey: str, **upload_kwargs) -> str:
        """Deploy embeddings to server."""
        print(f"🚀 Starting deployment for hotkey: {hotkey}")
        
        # Generate and validate embeddings
        print("📊 Generating embeddings...")
        embeddings = self.generate_embeddings()
        
        for i, (asset, embedding) in enumerate(zip(ASSETS, embeddings)):
            expected_dim = ASSET_EMBEDDING_DIMS[asset]
            if len(embedding) != expected_dim:
                raise ValueError(f"Wrong dimension for {asset}")
            for val in embedding:
                if not (-1.0 <= val <= 1.0):
                    raise ValueError(f"Invalid embedding value: {val}")
        
        # Encrypt
        print("🔐 Encrypting payload...")
        payload = self.encrypt_payload(embeddings, hotkey)
        
        # Upload based on method
        print(f"☁️ Uploading via {self.upload_method}...")
        if self.upload_method == "scp":
            self.upload_via_scp(payload, hotkey, **upload_kwargs)
        elif self.upload_method == "http_post":
            self.upload_via_http_post(payload, hotkey, **upload_kwargs)
        else:
            raise ValueError(f"Unsupported upload method: {self.upload_method}")
        
        public_url = f"{self.server_url}/{hotkey}"
        print(f"✅ Deployed to: {public_url}")
        return public_url


def main():
    """Example usage."""
    MY_HOTKEY = "5D..."  # TODO: Replace with your hotkey
    
    if MY_HOTKEY == "5D...":
        print("❌ Please set your actual hotkey!")
        return
    
    # Example 1: SCP deployment
    deployer = ServerDeployer("https://myserver.com", "scp")
    try:
        public_url = deployer.deploy(
            MY_HOTKEY,
            server_path=f"user@myserver.com:/var/www/html/{MY_HOTKEY}",
            ssh_key="/path/to/ssh/key"  # Optional
        )
        print(f"🎉 Deployed to: {public_url}")
    except Exception as e:
        print(f"❌ Deployment failed: {e}")


if __name__ == "__main__":
    main()