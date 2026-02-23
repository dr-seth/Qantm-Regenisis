# Platform Setup Guide

**Last Updated**: February 2026

## Overview

This guide covers setting up connections to supported trading platforms.

## Binance

### Account Setup

1. Create a Binance account at https://www.binance.com
2. Complete identity verification (KYC)
3. Enable 2FA for security

### API Key Creation

1. Go to API Management in your account settings
2. Create a new API key with a descriptive label
3. Configure permissions:
   - **Read**: Required for market data
   - **Spot Trading**: Required for trading
   - **Futures Trading**: Optional, for futures
4. Whitelist your IP addresses for security
5. Save the API key and secret securely

### Environment Variables

```bash
# Production
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"

# Testnet (for development)
export BINANCE_TESTNET_API_KEY="your_testnet_key"
export BINANCE_TESTNET_API_SECRET="your_testnet_secret"
```

### Testnet Setup

For development, use Binance Testnet:

1. Go to https://testnet.binance.vision
2. Create testnet API keys
3. Get testnet funds from the faucet

### Testing Connection

```bash
python scripts/test_api_connections.py --platform binance
```

## Interactive Brokers

### Account Setup

1. Open an account at https://www.interactivebrokers.com
2. Fund your account
3. Apply for appropriate trading permissions

### TWS/Gateway Installation

1. Download TWS or IB Gateway from the IB website
2. Install and configure:
   - Enable API connections
   - Set socket port (7496 for live, 7497 for paper)
   - Configure trusted IPs

### Paper Trading

1. Log into TWS with paper trading credentials
2. Ensure paper trading port (7497) is configured
3. Enable "Allow connections from localhost only"

### Environment Variables

```bash
export IB_HOST="127.0.0.1"
export IB_PORT="7497"  # 7496 for live, 7497 for paper
export IB_CLIENT_ID="1"
```

### Testing Connection

```bash
# Ensure TWS/Gateway is running first
python scripts/test_api_connections.py --platform interactive_brokers
```

## QuantConnect

### Account Setup

1. Create an account at https://www.quantconnect.com
2. Subscribe to a plan with live trading access

### API Credentials

1. Go to Account Settings > API
2. Generate API credentials
3. Note your User ID and API Token

### Environment Variables

```bash
export QC_USER_ID="your_user_id"
export QC_API_TOKEN="your_api_token"
```

### Testing Connection

```bash
python scripts/test_api_connections.py --platform quantconnect
```

## Security Best Practices

### API Key Security

- **Never commit API keys** to version control
- Use environment variables or secrets management
- Rotate keys periodically
- Use IP whitelisting where available
- Enable 2FA on all accounts

### Permission Minimization

- Only enable required permissions
- Use read-only keys for data collection
- Use separate keys for paper vs live trading

### Monitoring

- Monitor API usage and alerts
- Set up withdrawal address whitelisting
- Enable login notifications

## Troubleshooting

### Binance

| Issue | Solution |
|-------|----------|
| Invalid API key | Check key is correct and not expired |
| IP not whitelisted | Add your IP to the whitelist |
| Permission denied | Enable required permissions |
| Rate limited | Reduce request frequency |

### Interactive Brokers

| Issue | Solution |
|-------|----------|
| Connection refused | Ensure TWS/Gateway is running |
| Not logged in | Log into TWS/Gateway |
| API not enabled | Enable API in TWS settings |
| Wrong port | Check port configuration |

### QuantConnect

| Issue | Solution |
|-------|----------|
| Authentication failed | Verify User ID and Token |
| Subscription required | Upgrade to paid plan |
| Project not found | Check project ID |
