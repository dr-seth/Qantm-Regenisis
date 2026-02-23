"""
QuantConnect Data Models

Pydantic models for QuantConnect API request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "Python"
    CSHARP = "C#"


class BacktestStatus(str, Enum):
    """Backtest execution status."""
    QUEUED = "Queued"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class LiveAlgorithmStatus(str, Enum):
    """Live algorithm status."""
    DEPLOYING = "Deploying"
    RUNNING = "Running"
    STOPPED = "Stopped"
    LIQUIDATED = "Liquidated"
    ERROR = "Error"


class Resolution(str, Enum):
    """Data resolution options."""
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAILY = "daily"


class OrderType(str, Enum):
    """Order types."""
    MARKET = "Market"
    LIMIT = "Limit"
    STOP_MARKET = "StopMarket"
    STOP_LIMIT = "StopLimit"
    MARKET_ON_OPEN = "MarketOnOpen"
    MARKET_ON_CLOSE = "MarketOnClose"


class OrderStatus(str, Enum):
    """Order status."""
    NEW = "New"
    SUBMITTED = "Submitted"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    INVALID = "Invalid"


# ==================== Request Models ====================

class CreateProjectRequest(BaseModel):
    """Request to create a new project."""
    name: str = Field(..., description="Project name")
    language: Language = Field(default=Language.PYTHON, description="Programming language")


class CreateFileRequest(BaseModel):
    """Request to create a file in a project."""
    project_id: int = Field(..., alias="projectId", description="Project ID")
    name: str = Field(..., description="File name")
    content: str = Field(..., description="File contents")


class UpdateFileRequest(BaseModel):
    """Request to update a file in a project."""
    project_id: int = Field(..., alias="projectId", description="Project ID")
    name: str = Field(..., description="File name")
    content: str = Field(..., description="New file contents")


class CompileRequest(BaseModel):
    """Request to compile a project."""
    project_id: int = Field(..., alias="projectId", description="Project ID")


class CreateBacktestRequest(BaseModel):
    """Request to create a backtest."""
    project_id: int = Field(..., alias="projectId", description="Project ID")
    compile_id: str = Field(..., alias="compileId", description="Compile ID")
    backtest_name: str = Field(..., alias="backtestName", description="Backtest name")


class CreateLiveRequest(BaseModel):
    """Request to deploy a live algorithm."""
    project_id: int = Field(..., alias="projectId", description="Project ID")
    compile_id: str = Field(..., alias="compileId", description="Compile ID")
    node_id: str = Field(..., alias="nodeId", description="Node ID for deployment")
    brokerage: str = Field(..., description="Brokerage name")
    brokerage_data: Dict[str, Any] = Field(..., alias="brokerageData", description="Brokerage config")


class DataDownloadRequest(BaseModel):
    """Request to download historical data."""
    symbol: str = Field(..., description="Ticker symbol")
    resolution: Resolution = Field(..., description="Data resolution")
    date: str = Field(..., description="Date in YYYYMMDD format")
    market: str = Field(default="usa", description="Market (usa, crypto, etc.)")


# ==================== Response Models ====================

class BaseResponse(BaseModel):
    """Base API response."""
    success: bool = Field(..., description="Whether request was successful")
    errors: Optional[List[str]] = Field(default=None, description="Error messages")


class AuthenticateResponse(BaseResponse):
    """Authentication response."""
    pass


class ProjectModel(BaseModel):
    """Project data model."""
    project_id: int = Field(..., alias="projectId")
    name: str
    created: datetime
    modified: datetime
    language: Language


class ProjectsResponse(BaseResponse):
    """Projects list response."""
    projects: List[ProjectModel] = Field(default_factory=list)


class FileModel(BaseModel):
    """File data model."""
    name: str
    content: str
    modified: datetime


class FilesResponse(BaseResponse):
    """Files list response."""
    files: List[FileModel] = Field(default_factory=list)


class CompileModel(BaseModel):
    """Compile data model."""
    compile_id: str = Field(..., alias="compileId")
    state: str
    logs: List[str] = Field(default_factory=list)


class CompileResponse(BaseResponse):
    """Compile response."""
    compile_id: Optional[str] = Field(default=None, alias="compileId")
    state: Optional[str] = None
    logs: List[str] = Field(default_factory=list)


class BacktestModel(BaseModel):
    """Backtest data model."""
    backtest_id: str = Field(..., alias="backtestId")
    project_id: int = Field(..., alias="projectId")
    name: str
    status: BacktestStatus
    created: datetime
    completed: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None


class BacktestsResponse(BaseResponse):
    """Backtests list response."""
    backtests: List[BacktestModel] = Field(default_factory=list)


class OrderModel(BaseModel):
    """Order data model."""
    order_id: int = Field(..., alias="orderId")
    symbol: str
    quantity: float
    order_type: OrderType = Field(..., alias="type")
    status: OrderStatus
    limit_price: Optional[float] = Field(default=None, alias="limitPrice")
    stop_price: Optional[float] = Field(default=None, alias="stopPrice")
    created: datetime
    filled: Optional[datetime] = None
    fill_price: Optional[float] = Field(default=None, alias="fillPrice")


class OrdersResponse(BaseResponse):
    """Orders list response."""
    orders: List[OrderModel] = Field(default_factory=list)


class HoldingModel(BaseModel):
    """Holding data model."""
    symbol: str
    quantity: float
    average_price: float = Field(..., alias="averagePrice")
    market_price: float = Field(..., alias="marketPrice")
    market_value: float = Field(..., alias="marketValue")
    unrealized_pnl: float = Field(..., alias="unrealizedPnl")


class HoldingsResponse(BaseResponse):
    """Holdings response."""
    holdings: List[HoldingModel] = Field(default_factory=list)
    cash: float = 0.0
    equity: float = 0.0


class LiveAlgorithmModel(BaseModel):
    """Live algorithm data model."""
    project_id: int = Field(..., alias="projectId")
    deploy_id: str = Field(..., alias="deployId")
    status: LiveAlgorithmStatus
    launched: datetime
    stopped: Optional[datetime] = None
    brokerage: str
    subscription: str


class LiveAlgorithmsResponse(BaseResponse):
    """Live algorithms list response."""
    live: List[LiveAlgorithmModel] = Field(default_factory=list)


class AccountModel(BaseModel):
    """Account data model."""
    organization_id: str = Field(..., alias="organizationId")
    subscription: str
    credit: float
    balance: float


class AccountResponse(BaseResponse):
    """Account response."""
    account: Optional[AccountModel] = None
