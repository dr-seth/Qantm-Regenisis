"""
QuantConnect Connector

High-level connector for AARA-QuantConnect integration.
Provides simplified interface for common trading operations.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .auth import QuantConnectAuth
from .client import QuantConnectClient, QCResponse, Project, Backtest


class ConnectionState(Enum):
    """Connector connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class AlgorithmDeployment:
    """Represents a deployed algorithm."""
    project_id: int
    deploy_id: str
    name: str
    status: str
    launched: datetime
    brokerage: str
    
    
@dataclass
class BacktestResult:
    """Backtest execution result."""
    backtest_id: str
    name: str
    status: str
    sharpe_ratio: Optional[float] = None
    total_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    total_trades: int = 0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    @classmethod
    def from_backtest(cls, backtest: Backtest) -> 'BacktestResult':
        """Create BacktestResult from Backtest object."""
        result = backtest.result or {}
        statistics = result.get('Statistics', {})
        
        return cls(
            backtest_id=backtest.backtest_id,
            name=backtest.name,
            status=backtest.status,
            sharpe_ratio=cls._parse_float(statistics.get('Sharpe Ratio')),
            total_return=cls._parse_percent(statistics.get('Total Net Profit')),
            max_drawdown=cls._parse_percent(statistics.get('Drawdown')),
            win_rate=cls._parse_percent(statistics.get('Win Rate')),
            total_trades=int(statistics.get('Total Trades', 0)),
            start_date=backtest.created,
            end_date=backtest.completed
        )
    
    @staticmethod
    def _parse_float(value: Optional[str]) -> Optional[float]:
        """Parse float from string."""
        if value is None:
            return None
        try:
            return float(value.replace('%', '').replace(',', ''))
        except (ValueError, AttributeError):
            return None
    
    @staticmethod
    def _parse_percent(value: Optional[str]) -> Optional[float]:
        """Parse percentage from string."""
        if value is None:
            return None
        try:
            clean = value.replace('%', '').replace(',', '')
            return float(clean) / 100.0
        except (ValueError, AttributeError):
            return None


class QuantConnectConnector:
    """
    High-level connector for QuantConnect integration with AARA.
    
    Provides simplified interface for:
    - Algorithm deployment and management
    - Backtest execution and monitoring
    - Live trading control
    - Position and order tracking
    
    Example:
        >>> connector = QuantConnectConnector()
        >>> await connector.connect()
        >>> 
        >>> # Run a backtest
        >>> result = await connector.run_backtest(
        ...     project_id=12345,
        ...     name="RSI Strategy Test"
        ... )
        >>> print(f"Sharpe: {result.sharpe_ratio}")
        >>> 
        >>> # Deploy live
        >>> deployment = await connector.deploy_live(
        ...     project_id=12345,
        ...     brokerage="InteractiveBrokers"
        ... )
    """
    
    def __init__(
        self,
        user_id: Optional[int] = None,
        api_token: Optional[str] = None,
        organization_id: Optional[str] = None
    ):
        """
        Initialize QuantConnect connector.
        
        Args:
            user_id: QuantConnect user ID (or QC_USER_ID env var)
            api_token: QuantConnect API token (or QC_API_TOKEN env var)
            organization_id: Organization ID (or QC_ORGANIZATION_ID env var)
        """
        self.client = QuantConnectClient(
            user_id=user_id,
            api_token=api_token,
            organization_id=organization_id
        )
        self.state = ConnectionState.DISCONNECTED
        self._callbacks: Dict[str, List[Callable]] = {
            'backtest_complete': [],
            'live_update': [],
            'order_filled': [],
            'error': []
        }
        self._active_backtests: Dict[str, asyncio.Task] = {}
    
    async def connect(self) -> bool:
        """
        Establish connection and verify authentication.
        
        Returns:
            True if connection successful.
        """
        self.state = ConnectionState.CONNECTING
        
        try:
            if self.client.authenticate():
                self.state = ConnectionState.CONNECTED
                return True
            else:
                self.state = ConnectionState.ERROR
                return False
        except Exception as e:
            self.state = ConnectionState.ERROR
            self._emit('error', str(e))
            return False
    
    def disconnect(self):
        """Disconnect and cleanup."""
        # Cancel any active backtest monitors
        for task in self._active_backtests.values():
            task.cancel()
        self._active_backtests.clear()
        self.state = ConnectionState.DISCONNECTED
    
    def on(self, event: str, callback: Callable):
        """
        Register event callback.
        
        Args:
            event: Event name (backtest_complete, live_update, order_filled, error)
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit(self, event: str, *args, **kwargs):
        """Emit event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                pass  # Don't let callback errors break the connector
    
    # ==================== Project Management ====================
    
    def get_projects(self) -> List[Project]:
        """Get all projects."""
        return self.client.list_projects()
    
    def get_project(self, project_id: int) -> Optional[Project]:
        """Get a specific project."""
        return self.client.get_project(project_id)
    
    def create_project(self, name: str, language: str = "Python") -> QCResponse:
        """Create a new project."""
        return self.client.create_project(name, language)
    
    # ==================== Backtesting ====================
    
    async def run_backtest(
        self,
        project_id: int,
        name: str,
        wait_for_completion: bool = True,
        poll_interval: float = 5.0
    ) -> BacktestResult:
        """
        Compile project and run backtest.
        
        Args:
            project_id: Project ID to backtest
            name: Name for the backtest
            wait_for_completion: Whether to wait for backtest to complete
            poll_interval: Seconds between status checks
        
        Returns:
            BacktestResult with performance metrics
        
        Raises:
            RuntimeError: If compilation or backtest creation fails
        """
        # Compile project
        compile_response = self.client.compile_project(project_id)
        if not compile_response.success:
            raise RuntimeError(f"Compilation failed: {compile_response.errors}")
        
        compile_id = compile_response.data.get('compileId')
        if not compile_id:
            raise RuntimeError("No compile ID returned")
        
        # Wait for compilation to complete
        while True:
            status = self.client.get_compile_status(project_id, compile_id)
            if status.data.get('state') == 'BuildSuccess':
                break
            elif status.data.get('state') == 'BuildError':
                raise RuntimeError(f"Build failed: {status.data.get('logs', [])}")
            await asyncio.sleep(1)
        
        # Create backtest
        backtest_response = self.client.create_backtest(project_id, compile_id, name)
        if not backtest_response.success:
            raise RuntimeError(f"Backtest creation failed: {backtest_response.errors}")
        
        backtest_id = backtest_response.data.get('backtestId')
        if not backtest_id:
            raise RuntimeError("No backtest ID returned")
        
        if not wait_for_completion:
            # Start background monitor
            task = asyncio.create_task(
                self._monitor_backtest(project_id, backtest_id, poll_interval)
            )
            self._active_backtests[backtest_id] = task
            
            # Return initial result
            backtest = self.client.get_backtest(project_id, backtest_id)
            return BacktestResult.from_backtest(backtest) if backtest else BacktestResult(
                backtest_id=backtest_id,
                name=name,
                status="Running"
            )
        
        # Wait for completion
        while True:
            backtest = self.client.get_backtest(project_id, backtest_id)
            if backtest and backtest.status in ('Completed', 'Failed', 'Cancelled'):
                result = BacktestResult.from_backtest(backtest)
                self._emit('backtest_complete', result)
                return result
            await asyncio.sleep(poll_interval)
    
    async def _monitor_backtest(
        self,
        project_id: int,
        backtest_id: str,
        poll_interval: float
    ):
        """Background monitor for backtest completion."""
        try:
            while True:
                backtest = self.client.get_backtest(project_id, backtest_id)
                if backtest and backtest.status in ('Completed', 'Failed', 'Cancelled'):
                    result = BacktestResult.from_backtest(backtest)
                    self._emit('backtest_complete', result)
                    break
                await asyncio.sleep(poll_interval)
        finally:
            self._active_backtests.pop(backtest_id, None)
    
    def get_backtest_results(self, project_id: int) -> List[BacktestResult]:
        """Get all backtest results for a project."""
        backtests = self.client.list_backtests(project_id)
        return [BacktestResult.from_backtest(b) for b in backtests]
    
    # ==================== Live Trading ====================
    
    async def deploy_live(
        self,
        project_id: int,
        brokerage: str,
        brokerage_config: Dict[str, Any],
        node_id: Optional[str] = None
    ) -> AlgorithmDeployment:
        """
        Deploy algorithm for live trading.
        
        Args:
            project_id: Project ID to deploy
            brokerage: Brokerage name (e.g., "InteractiveBrokers", "Alpaca")
            brokerage_config: Brokerage-specific configuration
            node_id: Node ID for deployment (auto-selected if None)
        
        Returns:
            AlgorithmDeployment with deployment details
        
        Raises:
            RuntimeError: If compilation or deployment fails
        """
        # Compile project
        compile_response = self.client.compile_project(project_id)
        if not compile_response.success:
            raise RuntimeError(f"Compilation failed: {compile_response.errors}")
        
        compile_id = compile_response.data.get('compileId')
        
        # Wait for compilation
        while True:
            status = self.client.get_compile_status(project_id, compile_id)
            if status.data.get('state') == 'BuildSuccess':
                break
            elif status.data.get('state') == 'BuildError':
                raise RuntimeError(f"Build failed: {status.data.get('logs', [])}")
            await asyncio.sleep(1)
        
        # Deploy live
        if not node_id:
            node_id = "L-MICRO"  # Default to micro node
        
        response = self.client.create_live_algorithm(
            project_id=project_id,
            compile_id=compile_id,
            node_id=node_id,
            brokerage=brokerage,
            brokerage_data=brokerage_config
        )
        
        if not response.success:
            raise RuntimeError(f"Deployment failed: {response.errors}")
        
        return AlgorithmDeployment(
            project_id=project_id,
            deploy_id=response.data.get('deployId', ''),
            name=response.data.get('name', ''),
            status='Deploying',
            launched=datetime.now(),
            brokerage=brokerage
        )
    
    def stop_live(self, project_id: int) -> QCResponse:
        """Stop a live algorithm."""
        return self.client.stop_live_algorithm(project_id)
    
    def liquidate_live(self, project_id: int) -> QCResponse:
        """Liquidate all positions and stop live algorithm."""
        return self.client.liquidate_live_algorithm(project_id)
    
    def get_live_status(self, project_id: int) -> QCResponse:
        """Get live algorithm status."""
        return self.client.get_live_algorithm(project_id)
    
    # ==================== Orders & Holdings ====================
    
    def get_orders(self, project_id: int) -> QCResponse:
        """Get orders for a live algorithm."""
        return self.client.get_orders(project_id)
    
    def get_holdings(self, project_id: int) -> QCResponse:
        """Get current holdings for a live algorithm."""
        return self.client.get_holdings(project_id)
    
    # ==================== Data ====================
    
    def download_data(
        self,
        symbol: str,
        resolution: str,
        date: str,
        market: str = "usa"
    ) -> QCResponse:
        """Download historical data."""
        return self.client.download_data(symbol, resolution, date, market)
    
    # ==================== Algorithm Code Management ====================
    
    def upload_algorithm(
        self,
        project_id: int,
        filename: str,
        code: str
    ) -> QCResponse:
        """
        Upload algorithm code to a project.
        
        Args:
            project_id: Project ID
            filename: File name (e.g., "main.py")
            code: Algorithm source code
        
        Returns:
            QCResponse with upload status
        """
        # Check if file exists
        files_response = self.client.list_files(project_id)
        if files_response.success:
            existing_files = [f.get('name') for f in files_response.data.get('files', [])]
            if filename in existing_files:
                return self.client.update_file(project_id, filename, code)
        
        return self.client.create_file(project_id, filename, code)
    
    def get_algorithm_code(self, project_id: int, filename: str) -> Optional[str]:
        """
        Get algorithm code from a project.
        
        Args:
            project_id: Project ID
            filename: File name to retrieve
        
        Returns:
            File contents or None if not found
        """
        response = self.client.get_file(project_id, filename)
        if response.success and response.data:
            files = response.data.get('files', [])
            if files:
                return files[0].get('content')
        return None
    
    @property
    def is_connected(self) -> bool:
        """Check if connector is connected."""
        return self.state == ConnectionState.CONNECTED
    
    def __repr__(self) -> str:
        """String representation."""
        return f"QuantConnectConnector(state={self.state.value})"
