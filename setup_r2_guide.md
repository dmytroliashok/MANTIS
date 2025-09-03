# Cloudflare R2 Setup Guide

This guide shows you how to get all the required environment variables for deploying your MANTIS embeddings to Cloudflare R2.

## Step 1: Create Cloudflare Account

1. Go to [https://cloudflare.com](https://cloudflare.com)
2. Sign up for a free account or log in to existing account
3. Navigate to the Cloudflare dashboard

## Step 2: Create R2 Bucket

1. In the Cloudflare dashboard, click **"R2 Object Storage"** in the left sidebar
2. Click **"Create bucket"**
3. Choose a bucket name (e.g., `mantis-embeddings-yourname`)
4. Select a location (choose closest to your location)
5. Click **"Create bucket"**

**Important**: Make note of your bucket name - this becomes your `R2_BUCKET_NAME`

## Step 3: Get Your Account ID

1. In the Cloudflare dashboard, look at the right sidebar
2. You'll see **"Account ID"** - copy this value
3. This is your `R2_ACCOUNT_ID`

## Step 4: Create API Token

1. Go to **"My Profile"** (click your profile icon in top right)
2. Click the **"API Tokens"** tab
3. Click **"Create Token"**
4. Click **"Custom token"** → **"Get started"**

Configure the token with these settings:
- **Token name**: `MANTIS R2 Access`
- **Permissions**: 
  - `Account` - `Cloudflare R2:Edit`
- **Account Resources**: 
  - `Include` - `Your Account`
- **Zone Resources**: 
  - `Include` - `All zones` (or leave default)

5. Click **"Continue to summary"**
6. Click **"Create Token"**
7. **COPY THE TOKEN** - this is your `R2_ACCESS_KEY_ID`

## Step 5: Create API Secret (Alternative Method)

If the above doesn't work, try this method:

1. Go to **"My Profile"** → **"API Tokens"**
2. Scroll down to **"Global API Key"**
3. Click **"View"** and enter your password
4. Copy the key - this can be used as `R2_SECRET_ACCESS_KEY`

**OR** create R2-specific credentials:

1. In R2 dashboard, click **"Manage R2 API tokens"**
2. Click **"Create API token"**
3. Give it a name like "MANTIS Access"
4. Set permissions to **"Admin Read & Write"**
5. Click **"Create API token"**
6. Copy both the **Access Key ID** and **Secret Access Key**

## Step 6: Configure Your .env File

Create a `.env` file in your project directory:

```bash
# Cloudflare R2 Configuration
R2_ACCOUNT_ID=your_account_id_from_step_3
R2_ACCESS_KEY_ID=your_access_key_from_step_4
R2_SECRET_ACCESS_KEY=your_secret_key_from_step_5
R2_BUCKET_NAME=your_bucket_name_from_step_2

# Bittensor Configuration
WALLET_NAME=your_wallet_name
WALLET_HOTKEY=your_hotkey_name
MY_HOTKEY=5D...your_actual_hotkey_ss58_address

# Optional: Custom server URL (if not using R2)
# SERVER_URL=https://your-server.com
```

## Step 7: Test Your Configuration

Run this test script to verify everything works:

```python
import os
from dotenv import load_dotenv
import boto3
from botocore.client import Config

load_dotenv()

# Test R2 connection
try:
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY_ID") 
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("R2_BUCKET_NAME")
    
    print(f"Account ID: {account_id}")
    print(f"Bucket: {bucket_name}")
    print(f"Access Key: {access_key[:10]}..." if access_key else "Missing")
    
    # Test connection
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4")
    )
    
    # List buckets to test connection
    response = s3_client.list_buckets()
    print("✅ R2 connection successful!")
    print(f"Available buckets: {[b['Name'] for b in response['Buckets']]}")
    
except Exception as e:
    print(f"❌ R2 connection failed: {e}")
```

## Step 8: Enable Public Access (Important!)

1. Go to your R2 bucket in the Cloudflare dashboard
2. Click **"Settings"** tab
3. Scroll to **"Public access"**
4. Click **"Allow Access"** 
5. This enables the public URL: `https://your-bucket-name.r2.dev/filename`

## Troubleshooting

**"Access Denied" errors**: 
- Check your API token permissions
- Ensure the token has R2:Edit permissions
- Verify your account ID is correct

**"Bucket not found"**:
- Double-check your bucket name spelling
- Ensure the bucket exists in your account

**"Invalid credentials"**:
- Regenerate your API token
- Make sure you copied the full token without extra spaces

**"Connection timeout"**:
- Check your internet connection
- Try a different region for your bucket

## Security Notes

- Keep your `.env` file private (add to `.gitignore`)
- Never commit API keys to version control
- Regularly rotate your API tokens
- Use minimal required permissions

## Cost Information

Cloudflare R2 pricing (as of 2024):
- **Storage**: $0.015 per GB per month
- **Class A operations** (writes): $4.50 per million requests
- **Class B operations** (reads): $0.36 per million requests
- **Egress**: Free up to 10GB per month

For MANTIS mining, costs should be minimal (few cents per month).