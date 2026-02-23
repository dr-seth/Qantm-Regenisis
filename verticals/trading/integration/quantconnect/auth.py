"""
QuantConnect API Authentication Module

Implements SHA-256 hashed token authentication for QuantConnect API v2.
Uses timestamped tokens as nonce to ensure each request has a unique signature.

Reference: https://www.quantconnect.com/docs/v2/our-platform/api-reference/authentication
"""

import os
from base64 import b64encode
from hashlib import sha256
from time import time
from typing import Dict, Optional


class QuantConnectAuth:
    """
    QuantConnect API Authentication Handler.
    
    Generates timestamped SHA-256 hashed tokens for secure API authentication.
    The API token is never sent directly - only a hash of token:timestamp.
    
    Attributes:
        user_id: QuantConnect user ID
        api_token: QuantConnect API token (kept secure, never transmitted)
        organization_id: Optional organization ID for org-specific requests
    """
    
    def __init__(
        self,
        user_id: Optional[int] = None,
        api_token: Optional[str] = None,
        organization_id: Optional[str] = None
    ):
        """
        Initialize QuantConnect authentication.
        
        Args:
            user_id: QuantConnect user ID. If None, reads from QC_USER_ID env var.
            api_token: QuantConnect API token. If None, reads from QC_API_TOKEN env var.
            organization_id: Organization ID. If None, reads from QC_ORGANIZATION_ID env var.
        
        Raises:
            ValueError: If user_id or api_token are not provided and not in environment.
        """
        self.user_id = user_id or self._get_env_int('QC_USER_ID')
        self.api_token = api_token or os.getenv('QC_API_TOKEN')
        self.organization_id = organization_id or os.getenv('QC_ORGANIZATION_ID')
        
        if not self.user_id:
            raise ValueError(
                "QuantConnect user_id required. "
                "Provide as argument or set QC_USER_ID environment variable."
            )
        if not self.api_token:
            raise ValueError(
                "QuantConnect api_token required. "
                "Provide as argument or set QC_API_TOKEN environment variable."
            )
    
    def _get_env_int(self, key: str) -> Optional[int]:
        """Get integer value from environment variable."""
        value = os.getenv(key)
        if value:
            try:
                return int(value)
            except ValueError:
                return None
        return None
    
    def get_timestamp(self) -> str:
        """
        Get current Unix timestamp as string.
        
        Returns:
            Current Unix timestamp (seconds since epoch) as string.
        """
        return str(int(time()))
    
    def hash_token(self, timestamp: str) -> str:
        """
        Create SHA-256 hash of api_token:timestamp.
        
        Args:
            timestamp: Unix timestamp string to include in hash.
        
        Returns:
            Hexadecimal SHA-256 hash of token:timestamp.
        """
        time_stamped_token = f'{self.api_token}:{timestamp}'.encode('utf-8')
        return sha256(time_stamped_token).hexdigest()
    
    def get_basic_auth(self, timestamp: str) -> str:
        """
        Generate Base64-encoded Basic authentication string.
        
        Args:
            timestamp: Unix timestamp string used in hash.
        
        Returns:
            Base64-encoded string of user_id:hashed_token.
        """
        hashed_token = self.hash_token(timestamp)
        authentication = f'{self.user_id}:{hashed_token}'.encode('utf-8')
        return b64encode(authentication).decode('ascii')
    
    def get_headers(self) -> Dict[str, str]:
        """
        Generate authentication headers for QuantConnect API request.
        
        Creates timestamped SHA-256 hashed token for secure authentication.
        Each call generates a new timestamp ensuring unique request signatures.
        
        Returns:
            Dictionary with Authorization and Timestamp headers.
        
        Example:
            >>> auth = QuantConnectAuth(user_id=12345, api_token='my_token')
            >>> headers = auth.get_headers()
            >>> # {'Authorization': 'Basic ...', 'Timestamp': '1234567890'}
        """
        timestamp = self.get_timestamp()
        authentication = self.get_basic_auth(timestamp)
        
        return {
            'Authorization': f'Basic {authentication}',
            'Timestamp': timestamp
        }
    
    def validate_credentials(self) -> bool:
        """
        Check if credentials are properly configured.
        
        Returns:
            True if user_id and api_token are set.
        """
        return bool(self.user_id and self.api_token)
    
    def __repr__(self) -> str:
        """String representation (hides sensitive token)."""
        return (
            f"QuantConnectAuth(user_id={self.user_id}, "
            f"organization_id={self.organization_id}, "
            f"token_configured={bool(self.api_token)})"
        )
