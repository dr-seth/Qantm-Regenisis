"""ARY-1083: OpenAPI Specification for QuantConnect Integration.

This module provides OpenAPI schema definitions for the QuantConnect
API integration endpoints, enabling integration with the API documentation.

Author: ARYA Labs QantmOrchstrtr-RSI
Version: 1.0.0
Issue: ARY-1083
"""

from typing import Any, Dict

# OpenAPI Schema for QuantConnect Integration
QUANTCONNECT_OPENAPI_SCHEMA: Dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {
        "title": "QuantConnect Integration API",
        "description": (
            "API for integrating with QuantConnect algorithmic trading platform. "
            "Supports project management, backtesting, live trading, and data access."
        ),
        "version": "1.0.0",
        "contact": {
            "name": "ARYA Labs",
            "email": "engineering@aryalabs.io",
        },
    },
    "servers": [
        {
            "url": "https://www.quantconnect.com/api/v2",
            "description": "QuantConnect API v2",
        }
    ],
    "paths": {
        "/api/v1/trading/quantconnect/authenticate": {
            "post": {
                "tags": ["QuantConnect Authentication"],
                "summary": "Verify QuantConnect authentication",
                "description": "Verify that the configured QuantConnect credentials are valid.",
                "operationId": "authenticateQuantConnect",
                "responses": {
                    "200": {
                        "description": "Authentication successful",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AuthResponse"},
                            },
                        },
                    },
                    "401": {
                        "description": "Authentication failed",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                            },
                        },
                    },
                },
            },
        },
        "/api/v1/trading/quantconnect/projects": {
            "get": {
                "tags": ["QuantConnect Projects"],
                "summary": "List all projects",
                "description": "Get a list of all QuantConnect projects in the account.",
                "operationId": "listProjects",
                "responses": {
                    "200": {
                        "description": "List of projects",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Project"},
                                },
                            },
                        },
                    },
                },
            },
            "post": {
                "tags": ["QuantConnect Projects"],
                "summary": "Create a new project",
                "description": "Create a new algorithm project in QuantConnect.",
                "operationId": "createProject",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/CreateProjectRequest"},
                        },
                    },
                },
                "responses": {
                    "200": {
                        "description": "Project created",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Project"},
                            },
                        },
                    },
                },
            },
        },
        "/api/v1/trading/quantconnect/projects/{projectId}": {
            "get": {
                "tags": ["QuantConnect Projects"],
                "summary": "Get project details",
                "description": "Get details of a specific project.",
                "operationId": "getProject",
                "parameters": [
                    {
                        "name": "projectId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Project details",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Project"},
                            },
                        },
                    },
                    "404": {
                        "description": "Project not found",
                    },
                },
            },
            "delete": {
                "tags": ["QuantConnect Projects"],
                "summary": "Delete a project",
                "description": "Delete a project from QuantConnect.",
                "operationId": "deleteProject",
                "parameters": [
                    {
                        "name": "projectId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Project deleted",
                    },
                },
            },
        },
        "/api/v1/trading/quantconnect/backtests": {
            "post": {
                "tags": ["QuantConnect Backtesting"],
                "summary": "Run a backtest",
                "description": "Compile and run a backtest for a project.",
                "operationId": "runBacktest",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/RunBacktestRequest"},
                        },
                    },
                },
                "responses": {
                    "200": {
                        "description": "Backtest started",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/BacktestResult"},
                            },
                        },
                    },
                },
            },
        },
        "/api/v1/trading/quantconnect/backtests/{projectId}/{backtestId}": {
            "get": {
                "tags": ["QuantConnect Backtesting"],
                "summary": "Get backtest results",
                "description": "Get the status and results of a backtest.",
                "operationId": "getBacktest",
                "parameters": [
                    {
                        "name": "projectId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    },
                    {
                        "name": "backtestId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                ],
                "responses": {
                    "200": {
                        "description": "Backtest results",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/BacktestResult"},
                            },
                        },
                    },
                },
            },
        },
        "/api/v1/trading/quantconnect/live": {
            "post": {
                "tags": ["QuantConnect Live Trading"],
                "summary": "Deploy live algorithm",
                "description": "Deploy an algorithm for live trading.",
                "operationId": "deployLive",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/DeployLiveRequest"},
                        },
                    },
                },
                "responses": {
                    "200": {
                        "description": "Algorithm deployed",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/LiveDeployment"},
                            },
                        },
                    },
                },
            },
        },
        "/api/v1/trading/quantconnect/live/{projectId}": {
            "get": {
                "tags": ["QuantConnect Live Trading"],
                "summary": "Get live algorithm status",
                "description": "Get the status of a live trading algorithm.",
                "operationId": "getLiveStatus",
                "parameters": [
                    {
                        "name": "projectId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Live algorithm status",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/LiveDeployment"},
                            },
                        },
                    },
                },
            },
        },
        "/api/v1/trading/quantconnect/live/{projectId}/stop": {
            "post": {
                "tags": ["QuantConnect Live Trading"],
                "summary": "Stop live algorithm",
                "description": "Stop a running live algorithm.",
                "operationId": "stopLive",
                "parameters": [
                    {
                        "name": "projectId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Algorithm stopped",
                    },
                },
            },
        },
        "/api/v1/trading/quantconnect/live/{projectId}/liquidate": {
            "post": {
                "tags": ["QuantConnect Live Trading"],
                "summary": "Liquidate and stop",
                "description": "Liquidate all positions and stop the live algorithm.",
                "operationId": "liquidateLive",
                "parameters": [
                    {
                        "name": "projectId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Positions liquidated and algorithm stopped",
                    },
                },
            },
        },
        "/api/v1/trading/quantconnect/orders/{projectId}": {
            "get": {
                "tags": ["QuantConnect Orders"],
                "summary": "Get orders",
                "description": "Get orders for a live algorithm.",
                "operationId": "getOrders",
                "parameters": [
                    {
                        "name": "projectId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "List of orders",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "array",
                                    "items": {"$ref": "#/components/schemas/Order"},
                                },
                            },
                        },
                    },
                },
            },
        },
        "/api/v1/trading/quantconnect/holdings/{projectId}": {
            "get": {
                "tags": ["QuantConnect Holdings"],
                "summary": "Get holdings",
                "description": "Get current holdings for a live algorithm.",
                "operationId": "getHoldings",
                "parameters": [
                    {
                        "name": "projectId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Current holdings",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Holdings"},
                            },
                        },
                    },
                },
            },
        },
    },
    "components": {
        "schemas": {
            "AuthResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "user_id": {"type": "integer"},
                    "organization_id": {"type": "string"},
                },
            },
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean", "example": False},
                    "errors": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
            "Project": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "name": {"type": "string"},
                    "created": {"type": "string", "format": "date-time"},
                    "modified": {"type": "string", "format": "date-time"},
                    "language": {
                        "type": "string",
                        "enum": ["Python", "C#"],
                    },
                },
            },
            "CreateProjectRequest": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "language": {
                        "type": "string",
                        "enum": ["Python", "C#"],
                        "default": "Python",
                    },
                },
            },
            "RunBacktestRequest": {
                "type": "object",
                "required": ["project_id", "name"],
                "properties": {
                    "project_id": {"type": "integer"},
                    "name": {"type": "string"},
                    "wait_for_completion": {
                        "type": "boolean",
                        "default": True,
                    },
                },
            },
            "BacktestResult": {
                "type": "object",
                "properties": {
                    "backtest_id": {"type": "string"},
                    "name": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["Queued", "Running", "Completed", "Failed", "Cancelled"],
                    },
                    "sharpe_ratio": {"type": "number", "nullable": True},
                    "total_return": {"type": "number", "nullable": True},
                    "max_drawdown": {"type": "number", "nullable": True},
                    "win_rate": {"type": "number", "nullable": True},
                    "total_trades": {"type": "integer"},
                    "start_date": {"type": "string", "format": "date-time"},
                    "end_date": {"type": "string", "format": "date-time"},
                },
            },
            "DeployLiveRequest": {
                "type": "object",
                "required": ["project_id", "brokerage", "brokerage_config"],
                "properties": {
                    "project_id": {"type": "integer"},
                    "brokerage": {
                        "type": "string",
                        "description": "Brokerage name (e.g., InteractiveBrokers, Alpaca)",
                    },
                    "brokerage_config": {
                        "type": "object",
                        "description": "Brokerage-specific configuration",
                    },
                    "node_id": {
                        "type": "string",
                        "description": "Node ID for deployment",
                        "default": "L-MICRO",
                    },
                },
            },
            "LiveDeployment": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "integer"},
                    "deploy_id": {"type": "string"},
                    "name": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["Deploying", "Running", "Stopped", "Liquidated", "Error"],
                    },
                    "launched": {"type": "string", "format": "date-time"},
                    "brokerage": {"type": "string"},
                },
            },
            "Order": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer"},
                    "symbol": {"type": "string"},
                    "quantity": {"type": "number"},
                    "type": {
                        "type": "string",
                        "enum": ["Market", "Limit", "StopMarket", "StopLimit"],
                    },
                    "status": {
                        "type": "string",
                        "enum": ["New", "Submitted", "PartiallyFilled", "Filled", "Cancelled"],
                    },
                    "limit_price": {"type": "number", "nullable": True},
                    "stop_price": {"type": "number", "nullable": True},
                    "created": {"type": "string", "format": "date-time"},
                    "filled": {"type": "string", "format": "date-time", "nullable": True},
                    "fill_price": {"type": "number", "nullable": True},
                },
            },
            "Holding": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "quantity": {"type": "number"},
                    "average_price": {"type": "number"},
                    "market_price": {"type": "number"},
                    "market_value": {"type": "number"},
                    "unrealized_pnl": {"type": "number"},
                },
            },
            "Holdings": {
                "type": "object",
                "properties": {
                    "holdings": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/Holding"},
                    },
                    "cash": {"type": "number"},
                    "equity": {"type": "number"},
                },
            },
        },
        "securitySchemes": {
            "quantconnect_auth": {
                "type": "http",
                "scheme": "basic",
                "description": "QuantConnect API authentication using hashed token",
            },
        },
    },
    "security": [{"quantconnect_auth": []}],
}


def get_openapi_schema() -> Dict[str, Any]:
    """Get the OpenAPI schema for QuantConnect endpoints."""
    return QUANTCONNECT_OPENAPI_SCHEMA


def get_openapi_paths() -> Dict[str, Any]:
    """Get just the paths portion for integration with main OpenAPI spec."""
    return QUANTCONNECT_OPENAPI_SCHEMA["paths"]


def get_openapi_components() -> Dict[str, Any]:
    """Get just the components portion for integration with main OpenAPI spec."""
    return QUANTCONNECT_OPENAPI_SCHEMA["components"]


__all__ = [
    "QUANTCONNECT_OPENAPI_SCHEMA",
    "get_openapi_schema",
    "get_openapi_paths",
    "get_openapi_components",
]
