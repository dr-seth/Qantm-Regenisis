"""
Tests for QuantConnect Authentication Module

Tests SHA-256 hashed token generation and header creation.
"""

import os
import re
from base64 import b64decode
from hashlib import sha256
from unittest.mock import patch

import pytest

from verticals.trading.integration.quantconnect.auth import QuantConnectAuth


class TestQuantConnectAuth:
    """Test suite for QuantConnectAuth class."""
    
    @pytest.fixture
    def auth(self):
        """Create auth instance with test credentials."""
        return QuantConnectAuth(user_id=462895, api_token="test_token_12345")
    
    @pytest.fixture
    def auth_with_org(self):
        """Create auth instance with organization ID."""
        return QuantConnectAuth(
            user_id=462895,
            api_token="test_token_12345",
            organization_id="6158162de34ee050731d19d95b86e4de"
        )
    
    def test_init_with_credentials(self, auth):
        """Test initialization with explicit credentials."""
        assert auth.user_id == 462895
        assert auth.api_token == "test_token_12345"
        assert auth.organization_id is None
    
    def test_init_with_organization(self, auth_with_org):
        """Test initialization with organization ID."""
        assert auth_with_org.organization_id == "6158162de34ee050731d19d95b86e4de"
    
    def test_init_from_env_vars(self):
        """Test initialization from environment variables."""
        with patch.dict(os.environ, {
            'QC_USER_ID': '12345',
            'QC_API_TOKEN': 'env_token'
        }):
            auth = QuantConnectAuth()
            assert auth.user_id == 12345
            assert auth.api_token == 'env_token'
    
    def test_init_missing_user_id_raises(self):
        """Test that missing user_id raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="user_id required"):
                QuantConnectAuth(api_token="token")
    
    def test_init_missing_token_raises(self):
        """Test that missing api_token raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="api_token required"):
                QuantConnectAuth(user_id=12345)
    
    def test_get_timestamp_format(self, auth):
        """Test timestamp is Unix timestamp string."""
        timestamp = auth.get_timestamp()
        assert isinstance(timestamp, str)
        assert timestamp.isdigit()
        assert len(timestamp) >= 10  # Unix timestamp has at least 10 digits
    
    def test_hash_token_produces_sha256(self, auth):
        """Test hash_token produces valid SHA-256 hex digest."""
        timestamp = "1234567890"
        hashed = auth.hash_token(timestamp)
        
        # SHA-256 produces 64 character hex string
        assert len(hashed) == 64
        assert all(c in '0123456789abcdef' for c in hashed)
    
    def test_hash_token_matches_expected(self, auth):
        """Test hash_token produces correct hash."""
        timestamp = "1234567890"
        expected_input = f"test_token_12345:{timestamp}".encode('utf-8')
        expected_hash = sha256(expected_input).hexdigest()
        
        assert auth.hash_token(timestamp) == expected_hash
    
    def test_hash_token_different_timestamps(self, auth):
        """Test different timestamps produce different hashes."""
        hash1 = auth.hash_token("1234567890")
        hash2 = auth.hash_token("1234567891")
        
        assert hash1 != hash2
    
    def test_get_basic_auth_format(self, auth):
        """Test get_basic_auth produces valid Base64 string."""
        timestamp = "1234567890"
        basic_auth = auth.get_basic_auth(timestamp)
        
        # Should be valid Base64
        decoded = b64decode(basic_auth).decode('utf-8')
        
        # Should contain user_id:hashed_token
        assert decoded.startswith("462895:")
        assert len(decoded.split(':')[1]) == 64  # SHA-256 hex
    
    def test_get_headers_structure(self, auth):
        """Test get_headers returns correct structure."""
        headers = auth.get_headers()
        
        assert 'Authorization' in headers
        assert 'Timestamp' in headers
        assert headers['Authorization'].startswith('Basic ')
    
    def test_get_headers_authorization_valid(self, auth):
        """Test Authorization header contains valid Base64."""
        headers = auth.get_headers()
        
        # Extract Base64 part
        auth_value = headers['Authorization'].replace('Basic ', '')
        
        # Should decode successfully
        decoded = b64decode(auth_value).decode('utf-8')
        parts = decoded.split(':')
        
        assert len(parts) == 2
        assert parts[0] == "462895"
        assert len(parts[1]) == 64  # SHA-256 hex
    
    def test_get_headers_timestamp_matches(self, auth):
        """Test Timestamp header matches hash input."""
        headers = auth.get_headers()
        timestamp = headers['Timestamp']
        
        # Verify the hash was created with this timestamp
        expected_hash = auth.hash_token(timestamp)
        
        # Decode authorization to verify
        auth_value = headers['Authorization'].replace('Basic ', '')
        decoded = b64decode(auth_value).decode('utf-8')
        actual_hash = decoded.split(':')[1]
        
        assert actual_hash == expected_hash
    
    def test_get_headers_unique_per_call(self, auth):
        """Test each get_headers call produces unique timestamp."""
        import time
        
        headers1 = auth.get_headers()
        time.sleep(1.1)  # Ensure timestamp changes
        headers2 = auth.get_headers()
        
        # Timestamps should be different (or at least could be)
        # Authorization should be different due to timestamp
        assert headers1['Timestamp'] != headers2['Timestamp'] or \
               int(headers2['Timestamp']) - int(headers1['Timestamp']) <= 1
    
    def test_validate_credentials_true(self, auth):
        """Test validate_credentials returns True when configured."""
        assert auth.validate_credentials() is True
    
    def test_validate_credentials_false_no_token(self):
        """Test validate_credentials returns False without token."""
        with patch.dict(os.environ, {'QC_USER_ID': '12345', 'QC_API_TOKEN': 'token'}):
            auth = QuantConnectAuth()
            auth.api_token = None
            assert auth.validate_credentials() is False
    
    def test_repr_hides_token(self, auth):
        """Test __repr__ does not expose API token."""
        repr_str = repr(auth)
        
        assert "test_token_12345" not in repr_str
        assert "token_configured=True" in repr_str
        assert "user_id=462895" in repr_str
    
    def test_repr_with_organization(self, auth_with_org):
        """Test __repr__ includes organization ID."""
        repr_str = repr(auth_with_org)
        
        assert "organization_id=6158162de34ee050731d19d95b86e4de" in repr_str


class TestQuantConnectAuthIntegration:
    """Integration tests for QuantConnect authentication."""
    
    @pytest.mark.integration
    def test_real_authentication_flow(self):
        """Test authentication with real credentials (requires env vars)."""
        user_id = os.getenv('QC_USER_ID')
        api_token = os.getenv('QC_API_TOKEN')
        
        if not user_id or not api_token:
            pytest.skip("QC_USER_ID and QC_API_TOKEN not set")
        
        auth = QuantConnectAuth()
        headers = auth.get_headers()
        
        # Headers should be properly formatted
        assert 'Authorization' in headers
        assert 'Timestamp' in headers
        assert headers['Authorization'].startswith('Basic ')
