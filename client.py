"""
ML Training Optimizer Environment Client.

Provides an MCP tool client for interacting with the ML Training Optimizer
environment server.
"""

from openenv.core.mcp_client import MCPToolClient


class MLTrainerEnv(MCPToolClient):
    """
    Client for the ML Training Optimizer Environment.

    Inherits all MCP functionality from MCPToolClient:
    - `list_tools()`: Discover available tools
    - `call_tool(name, **kwargs)`: Call a tool by name
    - `reset(**kwargs)`: Reset the environment
    - `step(action)`: Execute an action

    Example:
        >>> with MLTrainerEnv(base_url="http://localhost:8000") as env:
        ...     env.reset(task_id="easy_mnist")
        ...     tools = env.list_tools()
        ...     result = env.call_tool("configure_training",
        ...         optimizer="adam", learning_rate=0.001)
        ...     result = env.call_tool("run_epochs", num_epochs=10)
        ...     result = env.call_tool("submit_model")
    """

    pass  # MCPToolClient provides all needed functionality
