"""
FastAPI application for the ML Training Optimizer Environment.

This module creates an HTTP server that exposes the MLTrainerEnvironment
over HTTP and WebSocket endpoints, compatible with MCPToolClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    # Or run directly:
    uv run --project . server
"""

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from .ml_trainer_environment import MLTrainerEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.ml_trainer_environment import MLTrainerEnvironment

app = create_app(
    MLTrainerEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="ml_trainer_env",
)


@app.get("/health")
def health() -> dict:
    """Simple container health endpoint."""
    return {"status": "ok"}


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
