"""
Tests for QuantConnect API Client

Tests API client methods with mocked responses.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
import requests

from verticals.trading.integration.quantconnect.client import (
    QuantConnectClient,
    QCResponse,
    QCEndpoint,
    Project,
    Backtest,
)


class TestQCResponse:
    """Test suite for QCResponse dataclass."""
    
    def test_from_response_success(self):
        """Test creating QCResponse from successful response."""
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'success': True,
            'projects': [{'projectId': 1, 'name': 'Test'}]
        }
        
        qc_response = QCResponse.from_response(mock_response)
        
        assert qc_response.success is True
        assert qc_response.status_code == 200
        assert qc_response.data['projects'][0]['name'] == 'Test'
    
    def test_from_response_failure(self):
        """Test creating QCResponse from failed response."""
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 401
        mock_response.json.return_value = {
            'success': False,
            'errors': ['Unauthorized']
        }
        
        qc_response = QCResponse.from_response(mock_response)
        
        assert qc_response.success is False
        assert qc_response.status_code == 401
        assert 'Unauthorized' in qc_response.errors
    
    def test_from_response_invalid_json(self):
        """Test handling invalid JSON response."""
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.side_effect = json.JSONDecodeError("", "", 0)
        
        qc_response = QCResponse.from_response(mock_response)
        
        assert qc_response.success is False
        assert qc_response.status_code == 500
        assert "Failed to parse response" in qc_response.errors[0]


class TestProject:
    """Test suite for Project dataclass."""
    
    def test_from_dict(self):
        """Test creating Project from API response dict."""
        data = {
            'projectId': 12345,
            'name': 'RSI Strategy',
            'created': '2024-01-15T10:30:00Z',
            'modified': '2024-01-20T15:45:00Z',
            'language': 'Python'
        }
        
        project = Project.from_dict(data)
        
        assert project.project_id == 12345
        assert project.name == 'RSI Strategy'
        assert project.language == 'Python'
        assert isinstance(project.created, datetime)
    
    def test_from_dict_missing_fields(self):
        """Test Project creation with missing fields uses defaults."""
        data = {}
        
        project = Project.from_dict(data)
        
        assert project.project_id == 0
        assert project.name == ''
        assert project.language == 'Python'


class TestBacktest:
    """Test suite for Backtest dataclass."""
    
    def test_from_dict(self):
        """Test creating Backtest from API response dict."""
        data = {
            'backtestId': 'bt-12345',
            'projectId': 100,
            'name': 'Test Backtest',
            'status': 'Completed',
            'created': '2024-01-15T10:30:00Z',
            'completed': '2024-01-15T10:35:00Z',
            'progress': 1.0,
            'result': {'Statistics': {'Sharpe Ratio': '1.5'}}
        }
        
        backtest = Backtest.from_dict(data)
        
        assert backtest.backtest_id == 'bt-12345'
        assert backtest.project_id == 100
        assert backtest.status == 'Completed'
        assert backtest.progress == 1.0
        assert backtest.result is not None


class TestQuantConnectClient:
    """Test suite for QuantConnectClient."""
    
    @pytest.fixture
    def client(self):
        """Create client with test credentials."""
        return QuantConnectClient(user_id=462895, api_token="test_token")
    
    @pytest.fixture
    def mock_session(self):
        """Create mock requests session."""
        with patch('requests.Session') as mock:
            yield mock.return_value
    
    def test_init(self, client):
        """Test client initialization."""
        assert client.auth.user_id == 462895
        assert client.base_url == "https://www.quantconnect.com/api/v2"
        assert client.timeout == 30
    
    def test_authenticate_success(self, client):
        """Test successful authentication."""
        with patch.object(client._session, 'request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {'success': True}
            mock_response.status_code = 200
            mock_request.return_value = mock_response
            
            result = client.authenticate()
            
            assert result is True
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert 'authenticate' in call_args.kwargs['url']
    
    def test_authenticate_failure(self, client):
        """Test failed authentication."""
        with patch.object(client._session, 'request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {'success': False, 'errors': ['Invalid token']}
            mock_response.status_code = 401
            mock_request.return_value = mock_response
            
            result = client.authenticate()
            
            assert result is False
    
    def test_list_projects(self, client):
        """Test listing projects."""
        with patch.object(client._session, 'request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                'success': True,
                'projects': [
                    {
                        'projectId': 1,
                        'name': 'Project 1',
                        'created': '2024-01-01T00:00:00Z',
                        'modified': '2024-01-01T00:00:00Z',
                        'language': 'Python'
                    },
                    {
                        'projectId': 2,
                        'name': 'Project 2',
                        'created': '2024-01-02T00:00:00Z',
                        'modified': '2024-01-02T00:00:00Z',
                        'language': 'C#'
                    }
                ]
            }
            mock_response.status_code = 200
            mock_request.return_value = mock_response
            
            projects = client.list_projects()
            
            assert len(projects) == 2
            assert projects[0].name == 'Project 1'
            assert projects[1].language == 'C#'
    
    def test_create_project(self, client):
        """Test creating a project."""
        with patch.object(client._session, 'request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                'success': True,
                'projects': [{'projectId': 123, 'name': 'New Project'}]
            }
            mock_response.status_code = 200
            mock_request.return_value = mock_response
            
            response = client.create_project("New Project", "Python")
            
            assert response.success is True
            call_args = mock_request.call_args
            assert call_args.kwargs['json'] == {'name': 'New Project', 'language': 'Python'}
    
    def test_compile_project(self, client):
        """Test compiling a project."""
        with patch.object(client._session, 'request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                'success': True,
                'compileId': 'compile-123',
                'state': 'InQueue'
            }
            mock_response.status_code = 200
            mock_request.return_value = mock_response
            
            response = client.compile_project(12345)
            
            assert response.success is True
            assert response.data['compileId'] == 'compile-123'
    
    def test_create_backtest(self, client):
        """Test creating a backtest."""
        with patch.object(client._session, 'request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                'success': True,
                'backtestId': 'bt-456'
            }
            mock_response.status_code = 200
            mock_request.return_value = mock_response
            
            response = client.create_backtest(
                project_id=123,
                compile_id='compile-123',
                name='Test Backtest'
            )
            
            assert response.success is True
            assert response.data['backtestId'] == 'bt-456'
    
    def test_get_backtest(self, client):
        """Test getting backtest status."""
        with patch.object(client._session, 'request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                'success': True,
                'backtestId': 'bt-456',
                'projectId': 123,
                'name': 'Test',
                'status': 'Completed',
                'created': '2024-01-01T00:00:00Z',
                'progress': 1.0
            }
            mock_response.status_code = 200
            mock_request.return_value = mock_response
            
            backtest = client.get_backtest(123, 'bt-456')
            
            assert backtest is not None
            assert backtest.status == 'Completed'
            assert backtest.progress == 1.0
    
    def test_request_timeout(self, client):
        """Test handling request timeout."""
        with patch.object(client._session, 'request') as mock_request:
            mock_request.side_effect = requests.exceptions.Timeout()
            
            response = client._request('GET', 'test')
            
            assert response.success is False
            assert response.status_code == 408
            assert 'timed out' in response.errors[0].lower()
    
    def test_request_connection_error(self, client):
        """Test handling connection error."""
        with patch.object(client._session, 'request') as mock_request:
            mock_request.side_effect = requests.exceptions.ConnectionError("Connection refused")
            
            response = client._request('GET', 'test')
            
            assert response.success is False
            assert response.status_code == 503
            assert 'Connection error' in response.errors[0]
    
    def test_headers_included(self, client):
        """Test that authentication headers are included."""
        with patch.object(client._session, 'request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {'success': True}
            mock_response.status_code = 200
            mock_request.return_value = mock_response
            
            client.authenticate()
            
            call_args = mock_request.call_args
            headers = call_args.kwargs['headers']
            
            assert 'Authorization' in headers
            assert 'Timestamp' in headers
            assert headers['Authorization'].startswith('Basic ')


class TestQuantConnectClientIntegration:
    """Integration tests for QuantConnect client."""
    
    @pytest.fixture
    def live_client(self):
        """Create client with real credentials from env."""
        import os
        user_id = os.getenv('QC_USER_ID')
        api_token = os.getenv('QC_API_TOKEN')
        
        if not user_id or not api_token:
            pytest.skip("QC_USER_ID and QC_API_TOKEN not set")
        
        return QuantConnectClient()
    
    @pytest.mark.integration
    def test_real_authentication(self, live_client):
        """Test real authentication against QuantConnect API."""
        result = live_client.authenticate()
        assert result is True
    
    @pytest.mark.integration
    def test_real_list_projects(self, live_client):
        """Test listing real projects."""
        projects = live_client.list_projects()
        # Should return a list (may be empty)
        assert isinstance(projects, list)
    
    @pytest.mark.integration
    def test_real_get_account(self, live_client):
        """Test getting real account info."""
        response = live_client.get_account()
        assert response.success is True
