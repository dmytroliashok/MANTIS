#!/usr/bin/env python3
"""
Test script to verify your R2 configuration is working correctly.
Run this after setting up your .env file to make sure everything is configured properly.
"""

import os
import json
from dotenv import load_dotenv
import boto3
from botocore.client import Config

def test_r2_connection():
    """Test R2 connection and configuration."""
    
    print("🔧 Testing Cloudflare R2 Configuration")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Check if all required variables are present
    required_vars = ["R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"]
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            # Show partial value for security
            if "KEY" in var:
                display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else f"{value[:4]}..."
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
    
    if missing_vars:
        print(f"\n❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and make sure all variables are set.")
        return False
    
    # Test R2 connection
    try:
        account_id = os.getenv("R2_ACCOUNT_ID")
        access_key = os.getenv("R2_ACCESS_KEY_ID")
        secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
        bucket_name = os.getenv("R2_BUCKET_NAME")
        
        print(f"\n🌐 Testing connection to R2...")
        
        # Create R2 client
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
        s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(signature_version="s3v4")
        )
        
        # Test 1: List buckets
        print("📋 Listing buckets...")
        response = s3_client.list_buckets()
        available_buckets = [b['Name'] for b in response['Buckets']]
        print(f"✅ Found {len(available_buckets)} buckets: {available_buckets}")
        
        # Test 2: Check if target bucket exists
        if bucket_name in available_buckets:
            print(f"✅ Target bucket '{bucket_name}' exists")
        else:
            print(f"❌ Target bucket '{bucket_name}' not found!")
            print(f"Available buckets: {available_buckets}")
            return False
        
        # Test 3: Try to upload a test file
        print(f"\n📤 Testing upload to bucket '{bucket_name}'...")
        test_content = json.dumps({"test": "data", "timestamp": "2024-01-01"})
        test_key = "test-connection"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=test_content.encode('utf-8'),
            ContentType='application/json'
        )
        print("✅ Test upload successful")
        
        # Test 4: Try to download the test file
        print("📥 Testing download...")
        response = s3_client.get_object(Bucket=bucket_name, Key=test_key)
        downloaded_content = response['Body'].read().decode('utf-8')
        
        if downloaded_content == test_content:
            print("✅ Test download successful")
        else:
            print("❌ Downloaded content doesn't match uploaded content")
            return False
        
        # Test 5: Check public URL
        public_url = f"https://{bucket_name}.r2.dev/{test_key}"
        print(f"\n🌍 Testing public access...")
        print(f"Public URL: {public_url}")
        
        try:
            import requests
            response = requests.get(public_url, timeout=10)
            if response.status_code == 200:
                print("✅ Public access working")
            else:
                print(f"⚠️  Public access returned status {response.status_code}")
                print("You may need to enable public access in your R2 bucket settings")
        except Exception as e:
            print(f"⚠️  Could not test public access: {e}")
            print("This might be normal if public access isn't enabled yet")
        
        # Cleanup: Remove test file
        print("\n🧹 Cleaning up test file...")
        s3_client.delete_object(Bucket=bucket_name, Key=test_key)
        print("✅ Cleanup complete")
        
        print(f"\n🎉 All tests passed! Your R2 configuration is working correctly.")
        print(f"You can now use deploy_to_r2.py to upload your embeddings.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ R2 connection test failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Double-check your Account ID in the Cloudflare dashboard")
        print("2. Regenerate your API token with R2:Edit permissions")
        print("3. Ensure your bucket name is spelled correctly")
        print("4. Check that your bucket exists in the correct account")
        return False


def show_next_steps():
    """Show what to do after successful configuration."""
    print("\n📋 Next Steps:")
    print("1. Edit deploy_to_r2.py and replace 'MY_HOTKEY = \"5D...\"' with your actual hotkey")
    print("2. Implement your embedding generation logic in the generate_embeddings() function")
    print("3. Run: python commit_url.py (one-time setup)")
    print("4. Run: python continuous_deployment.py --interval 5 (start mining)")
    print("\n💡 Pro tip: Start with simple test embeddings to make sure everything works!")


if __name__ == "__main__":
    success = test_r2_connection()
    if success:
        show_next_steps()
    else:
        print("\n🔧 Please fix the configuration issues above and try again.")