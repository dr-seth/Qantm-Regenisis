"""
QuantConnect API Client

Full-featured client for QuantConnect API v2 with support for:
- Authentication verification
- Project management
- Algorithm deployment
- Backtesting
- Live trading
- Data access

Reference: https://www.quantconnect.com/docs/v2/our-platform/api-reference
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import requests

from .auth import QuantConnectAuth


class QCEndpoint(Enum):
    """QuantConnect API endpoints."""
    AUTHENTICATE = "authenticate"
    PROJECTS = "projects"
    FILES = "files"
    COMPILE = "compile"
    BACKTEST = "backtests"
    LIVE = "live"
    DATA = "data"
    ACCOUNT = "account"
    ORDERS = "orders"
    HOLDINGS = "holdings"


@dataclass
class QCResponse:
    """
    Standardized response from QuantConnect API.
    
    Attributes:
        success: Whether the API request was successful
        data: Response data (varies by endpoint)
        errors: List of error messages if request failed
        status_code: HTTP status code
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    status_code: int = 200
    
    @classmethod
    def from_response(cls, response: requests.Response) -> 'QCResponse':
        """Create QCResponse from requests.Response object."""
        try:
            json_data = response.json()
            return cls(
                success=json_data.get('success', False),
                data=json_data,
                errors=json_data.get('errors', []),
                status_code=response.status_code
            )
        except json.JSONDecodeError:
            return cls(
                success=False,
                data=None,
                errors=[f"Failed to parse response: {response.text[:500]}"],
                status_code=response.status_code
            )


@dataclass
class Project:
    """QuantConnect project representation."""
    project_id: int
    name: str
    created: datetime
    modified: datetime
    language: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Create Project from API response dictionary."""
        return cls(
            project_id=data.get('projectId', 0),
            name=data.get('name', ''),
            created=datetime.fromisoformat(data.get('created', '').replace('Z', '+00:00')) if data.get('created') else datetime.now(),
            modified=datetime.fromisoformat(data.get('modified', '').replace('Z', '+00:00')) if data.get('modified') else datetime.now(),
            language=data.get('language', 'Python')
        )


@dataclass
class Backtest:
    """QuantConnect backtest representation."""
    backtest_id: str
    project_id: int
    name: str
    status: str
    created: datetime
    completed: Optional[datetime]
    progress: float
    result: Optional[Dict[str, Any]]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Backtest':
        """Create Backtest from API response dictionary."""
        return cls(
            backtest_id=data.get('backtestId', ''),
            project_id=data.get('projectId', 0),
            name=data.get('name', ''),
            status=data.get('status', 'Unknown'),
            created=datetime.fromisoformat(data.get('created', '').replace('Z', '+00:00')) if data.get('created') else datetime.now(),
            completed=datetime.fromisoformat(data.get('completed', '').replace('Z', '+00:00')) if data.get('completed') else None,
            progress=data.get('progress', 0.0),
            result=data.get('result')
        )


class QuantConnectClient:
    """
    QuantConnect API v2 Client.
    
    Provides methods for interacting with all QuantConnect API endpoints
    including authentication, project management, backtesting, and live trading.
    
    Attributes:
        auth: QuantConnectAuth instance for request authentication
        base_url: QuantConnect API base URL
        timeout: Request timeout in seconds
    
    Example:
        >>> client = QuantConnectClient(user_id=12345, api_token='my_token')
        >>> if client.authenticate():
        ...     projects = client.list_projects()
        ...     print(f"Found {len(projects)} projects")
    """
    
    BASE_URL = "https://www.quantconnect.com/api/v2"
    
    def __init__(
        self,
        user_id: Optional[int] = None,
        api_token: Optional[str] = None,
        organization_id: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize QuantConnect API client.
        
        Args:
            user_id: QuantConnect user ID. If None, reads from QC_USER_ID env var.
            api_token: QuantConnect API token. If None, reads from QC_API_TOKEN env var.
            organization_id: Organization ID for org-specific requests.
            timeout: Request timeout in seconds (default: 30).
        """
        self.auth = QuantConnectAuth(
            user_id=user_id,
            api_token=api_token,
            organization_id=organization_id
        )
        self.base_url = self.BASE_URL
        self.timeout = timeout
        self._session = requests.Session()
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> QCResponse:
        """
        Make authenticated request to QuantConnect API.
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint (without base URL)
            data: JSON body for POST requests
            params: Query parameters
        
        Returns:
            QCResponse with success status and data/errors
        """
        url = f"{self.base_url}/{endpoint}"
        headers = self.auth.get_headers()
        headers['Content-Type'] = 'application/json'
        
        try:
            response = self._session.request(
                method=method,
                url=url,
                headers=headers,
                json=data,
                params=params,
                timeout=self.timeout
            )
            return QCResponse.from_response(response)
        except requests.exceptions.Timeout:
            return QCResponse(
                success=False,
                errors=["Request timed out"],
                status_code=408
            )
        except requests.exceptions.ConnectionError as e:
            return QCResponse(
                success=False,
                errors=[f"Connection error: {str(e)}"],
                status_code=503
            )
        except requests.exceptions.RequestException as e:
            return QCResponse(
                success=False,
                errors=[f"Request failed: {str(e)}"],
                status_code=500
            )
    
    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> QCResponse:
        """Make authenticated GET request."""
        return self._request('GET', endpoint, params=params)
    
    def _post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> QCResponse:
        """Make authenticated POST request."""
        return self._request('POST', endpoint, data=data)
    
    def _delete(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> QCResponse:
        """Make authenticated DELETE request."""
        return self._request('DELETE', endpoint, data=data)
    
    # ==================== Authentication ====================
    
    def authenticate(self) -> bool:
        """
        Verify authentication credentials with QuantConnect API.
        
        Returns:
            True if authentication is successful, False otherwise.
        
        Example:
            >>> client = QuantConnectClient()
            >>> if client.authenticate():
            ...     print("Authentication successful!")
        """
        response = self._get(QCEndpoint.AUTHENTICATE.value)
        return response.success
    
    def get_auth_status(self) -> QCResponse:
        """
        Get detailed authentication status.
        
        Returns:
            QCResponse with authentication details.
        """
        return self._get(QCEndpoint.AUTHENTICATE.value)
    
    # ==================== Account ====================
    
    def get_account(self) -> QCResponse:
        """
        Get account information.
        
        Returns:
            QCResponse with account details including subscription info.
        """
        return self._get(QCEndpoint.ACCOUNT.value)
    
    # ==================== Projects ====================
    
    def list_projects(self) -> List[Project]:
        """
        List all projects in the account.
        
        Returns:
            List of Project objects.
        """
        response = self._get(QCEndpoint.PROJECTS.value)
        if response.success and response.data:
            projects_data = response.data.get('projects', [])
            return [Project.from_dict(p) for p in projects_data]
        return []
    
    def get_project(self, project_id: int) -> Optional[Project]:
        """
        Get a specific project by ID.
        
        Args:
            project_id: The project ID to retrieve.
        
        Returns:
            Project object if found, None otherwise.
        """
        response = self._get(f"{QCEndpoint.PROJECTS.value}/{project_id}")
        if response.success and response.data:
            projects = response.data.get('projects', [])
            if projects:
                return Project.from_dict(projects[0])
        return None
    
    def create_project(self, name: str, language: str = "Python") -> QCResponse:
        """
        Create a new project.
        
        Args:
            name: Project name.
            language: Programming language (Python or C#).
        
        Returns:
            QCResponse with created project details.
        """
        return self._post(
            QCEndpoint.PROJECTS.value,
            data={"name": name, "language": language}
        )
    
    def delete_project(self, project_id: int) -> QCResponse:
        """
        Delete a project.
        
        Args:
            project_id: The project ID to delete.
        
        Returns:
            QCResponse with deletion status.
        """
        return self._delete(
            QCEndpoint.PROJECTS.value,
            data={"projectId": project_id}
        )
    
    # ==================== Files ====================
    
    def list_files(self, project_id: int) -> QCResponse:
        """
        List all files in a project.
        
        Args:
            project_id: The project ID.
        
        Returns:
            QCResponse with list of files.
        """
        return self._get(
            QCEndpoint.FILES.value,
            params={"projectId": project_id}
        )
    
    def get_file(self, project_id: int, file_name: str) -> QCResponse:
        """
        Get contents of a specific file.
        
        Args:
            project_id: The project ID.
            file_name: Name of the file to retrieve.
        
        Returns:
            QCResponse with file contents.
        """
        return self._get(
            QCEndpoint.FILES.value,
            params={"projectId": project_id, "name": file_name}
        )
    
    def create_file(self, project_id: int, name: str, content: str) -> QCResponse:
        """
        Create a new file in a project.
        
        Args:
            project_id: The project ID.
            name: File name.
            content: File contents.
        
        Returns:
            QCResponse with creation status.
        """
        return self._post(
            QCEndpoint.FILES.value,
            data={"projectId": project_id, "name": name, "content": content}
        )
    
    def update_file(self, project_id: int, name: str, content: str) -> QCResponse:
        """
        Update an existing file in a project.
        
        Args:
            project_id: The project ID.
            name: File name.
            content: New file contents.
        
        Returns:
            QCResponse with update status.
        """
        return self._post(
            f"{QCEndpoint.FILES.value}/update",
            data={"projectId": project_id, "name": name, "content": content}
        )
    
    def delete_file(self, project_id: int, name: str) -> QCResponse:
        """
        Delete a file from a project.
        
        Args:
            project_id: The project ID.
            name: File name to delete.
        
        Returns:
            QCResponse with deletion status.
        """
        return self._delete(
            QCEndpoint.FILES.value,
            data={"projectId": project_id, "name": name}
        )
    
    # ==================== Compile ====================
    
    def compile_project(self, project_id: int) -> QCResponse:
        """
        Compile a project.
        
        Args:
            project_id: The project ID to compile.
        
        Returns:
            QCResponse with compile ID and status.
        """
        return self._post(
            QCEndpoint.COMPILE.value,
            data={"projectId": project_id}
        )
    
    def get_compile_status(self, project_id: int, compile_id: str) -> QCResponse:
        """
        Get compilation status.
        
        Args:
            project_id: The project ID.
            compile_id: The compile ID from compile_project().
        
        Returns:
            QCResponse with compilation status.
        """
        return self._get(
            QCEndpoint.COMPILE.value,
            params={"projectId": project_id, "compileId": compile_id}
        )
    
    # ==================== Backtesting ====================
    
    def create_backtest(
        self,
        project_id: int,
        compile_id: str,
        name: str
    ) -> QCResponse:
        """
        Create and run a backtest.
        
        Args:
            project_id: The project ID.
            compile_id: Compile ID from successful compilation.
            name: Name for the backtest.
        
        Returns:
            QCResponse with backtest ID and initial status.
        """
        return self._post(
            QCEndpoint.BACKTEST.value,
            data={
                "projectId": project_id,
                "compileId": compile_id,
                "backtestName": name
            }
        )
    
    def get_backtest(self, project_id: int, backtest_id: str) -> Optional[Backtest]:
        """
        Get backtest status and results.
        
        Args:
            project_id: The project ID.
            backtest_id: The backtest ID.
        
        Returns:
            Backtest object if found, None otherwise.
        """
        response = self._get(
            QCEndpoint.BACKTEST.value,
            params={"projectId": project_id, "backtestId": backtest_id}
        )
        if response.success and response.data:
            return Backtest.from_dict(response.data)
        return None
    
    def list_backtests(self, project_id: int) -> List[Backtest]:
        """
        List all backtests for a project.
        
        Args:
            project_id: The project ID.
        
        Returns:
            List of Backtest objects.
        """
        response = self._get(
            QCEndpoint.BACKTEST.value,
            params={"projectId": project_id}
        )
        if response.success and response.data:
            backtests_data = response.data.get('backtests', [])
            return [Backtest.from_dict(b) for b in backtests_data]
        return []
    
    def delete_backtest(self, project_id: int, backtest_id: str) -> QCResponse:
        """
        Delete a backtest.
        
        Args:
            project_id: The project ID.
            backtest_id: The backtest ID to delete.
        
        Returns:
            QCResponse with deletion status.
        """
        return self._delete(
            QCEndpoint.BACKTEST.value,
            data={"projectId": project_id, "backtestId": backtest_id}
        )
    
    # ==================== Live Trading ====================
    
    def create_live_algorithm(
        self,
        project_id: int,
        compile_id: str,
        node_id: str,
        brokerage: str,
        brokerage_data: Dict[str, Any]
    ) -> QCResponse:
        """
        Deploy a live trading algorithm.
        
        Args:
            project_id: The project ID.
            compile_id: Compile ID from successful compilation.
            node_id: Node ID for deployment.
            brokerage: Brokerage name (e.g., "InteractiveBrokers").
            brokerage_data: Brokerage-specific configuration.
        
        Returns:
            QCResponse with deployment status.
        """
        return self._post(
            QCEndpoint.LIVE.value,
            data={
                "projectId": project_id,
                "compileId": compile_id,
                "nodeId": node_id,
                "brokerage": brokerage,
                "brokerageData": brokerage_data
            }
        )
    
    def get_live_algorithm(self, project_id: int) -> QCResponse:
        """
        Get live algorithm status.
        
        Args:
            project_id: The project ID.
        
        Returns:
            QCResponse with live algorithm status and details.
        """
        return self._get(
            QCEndpoint.LIVE.value,
            params={"projectId": project_id}
        )
    
    def list_live_algorithms(self, status: str = "Running") -> QCResponse:
        """
        List all live algorithms.
        
        Args:
            status: Filter by status (Running, Stopped, etc.).
        
        Returns:
            QCResponse with list of live algorithms.
        """
        return self._get(
            f"{QCEndpoint.LIVE.value}/list",
            params={"status": status}
        )
    
    def stop_live_algorithm(self, project_id: int) -> QCResponse:
        """
        Stop a live algorithm.
        
        Args:
            project_id: The project ID.
        
        Returns:
            QCResponse with stop status.
        """
        return self._post(
            f"{QCEndpoint.LIVE.value}/stop",
            data={"projectId": project_id}
        )
    
    def liquidate_live_algorithm(self, project_id: int) -> QCResponse:
        """
        Liquidate all positions and stop live algorithm.
        
        Args:
            project_id: The project ID.
        
        Returns:
            QCResponse with liquidation status.
        """
        return self._post(
            f"{QCEndpoint.LIVE.value}/liquidate",
            data={"projectId": project_id}
        )
    
    # ==================== Orders ====================
    
    def get_orders(self, project_id: int, start: int = 0, end: int = 100) -> QCResponse:
        """
        Get orders for a live algorithm.
        
        Args:
            project_id: The project ID.
            start: Start index for pagination.
            end: End index for pagination.
        
        Returns:
            QCResponse with list of orders.
        """
        return self._get(
            QCEndpoint.ORDERS.value,
            params={"projectId": project_id, "start": start, "end": end}
        )
    
    # ==================== Holdings ====================
    
    def get_holdings(self, project_id: int) -> QCResponse:
        """
        Get current holdings for a live algorithm.
        
        Args:
            project_id: The project ID.
        
        Returns:
            QCResponse with current holdings.
        """
        return self._get(
            QCEndpoint.HOLDINGS.value,
            params={"projectId": project_id}
        )
    
    # ==================== Data ====================
    
    def get_data_prices(self) -> QCResponse:
        """
        Get data pricing information.
        
        Returns:
            QCResponse with data pricing details.
        """
        return self._get(f"{QCEndpoint.DATA.value}/prices")
    
    def download_data(
        self,
        symbol: str,
        resolution: str,
        date: str,
        market: str = "usa"
    ) -> QCResponse:
        """
        Download historical data.
        
        Args:
            symbol: Ticker symbol.
            resolution: Data resolution (minute, hour, daily).
            date: Date in YYYYMMDD format.
            market: Market (usa, crypto, etc.).
        
        Returns:
            QCResponse with data download link or data.
        """
        return self._get(
            f"{QCEndpoint.DATA.value}/read",
            params={
                "symbol": symbol,
                "resolution": resolution,
                "date": date,
                "market": market
            }
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"QuantConnectClient(user_id={self.auth.user_id}, authenticated={self.authenticate()})"
