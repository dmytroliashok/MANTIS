# MANTIS Deployment Guide

This guide shows you how to deploy your embeddings to a public URL for the MANTIS subnet.

## Quick Start

1. **Setup Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

2. **Install Dependencies**
   ```bash
   pip install boto3 python-dotenv timelock requests schedule
   ```

3. **Choose Deployment Method**
   - **Recommended**: Cloudflare R2 (use `deploy_to_r2.py`)
   - **Alternative**: Your own server (use `deploy_to_server.py`)

4. **One-time Setup**
   ```bash
   python commit_url.py  # Commit your URL to the subnet
   ```

5. **Start Mining**
   ```bash
   python continuous_deployment.py --interval 5  # Deploy every 5 minutes
   ```

## Deployment Options

### Option 1: Cloudflare R2 (Recommended)

**Pros**: Reliable, fast, cheap, global CDN
**Cons**: Requires R2 account setup

1. Create Cloudflare R2 bucket
2. Get API credentials
3. Configure `.env` file
4. Use `deploy_to_r2.py`

### Option 2: Your Own Server

**Pros**: Full control, no external dependencies
**Cons**: Requires server maintenance, uptime responsibility

1. Set up a web server
2. Configure `deploy_to_server.py`
3. Ensure 24/7 uptime

### Option 3: Simple HTTP Server

**Pros**: Easy setup for testing
**Cons**: Not suitable for production

```bash
python simple_http_server.py --port 8000
```

## File Structure

```
your_project/
├── deploy_to_r2.py          # R2 deployment script
├── deploy_to_server.py      # Server deployment script  
├── simple_http_server.py    # Local server for testing
├── commit_url.py            # One-time URL commitment
├── continuous_deployment.py # Continuous mining script
├── .env                     # Your configuration
└── embeddings/              # Local embedding storage
```

## Environment Variables

Create a `.env` file with:

```bash
# Cloudflare R2 (if using R2)
R2_ACCOUNT_ID=your_account_id
R2_ACCESS_KEY_ID=your_access_key  
R2_SECRET_ACCESS_KEY=your_secret_key
R2_BUCKET_NAME=your_bucket_name

# Bittensor wallet
WALLET_NAME=your_wallet_name
WALLET_HOTKEY=your_hotkey_name

# Your actual hotkey (ss58 format)
MY_HOTKEY=5D...your_actual_hotkey
```

## Important Notes

1. **Filename Must Match Hotkey**: Your file must be named exactly like your hotkey
2. **Public Access Required**: Validators must be able to download your file
3. **Regular Updates**: Update your embeddings frequently (every 1-5 minutes)
4. **Embedding Format**: Must follow the exact format specified in config.py
5. **Time-lock Encryption**: Always encrypt with your hotkey embedded

## Troubleshooting

**"Upload failed"**: Check your credentials and network connection
**"Invalid embedding"**: Verify dimensions and value ranges (-1 to 1)
**"Hotkey mismatch"**: Ensure filename matches your hotkey exactly
**"URL not accessible"**: Test your public URL in a browser

## Security

- Keep your private keys secure
- Use HTTPS for all public URLs
- Regularly rotate API credentials
- Monitor access logs for unusual activity

## Monitoring

Check your deployment logs:
```bash
tail -f miner.log
```

Monitor your public URL:
```bash
curl https://your-bucket.r2.dev/your-hotkey
```

## Support

If you encounter issues:
1. Check the logs for error messages
2. Verify your .env configuration
3. Test your public URL manually
4. Ensure your embeddings follow the correct format