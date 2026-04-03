"""
Data models for the ML Training Optimizer Environment.

Not used directly when running in MCP mode (tools define their own schemas),
but provided for compatibility with the OpenEnv spec.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Dict, List, Optional


class TrainerAction(Action):
    """Action for the ML Trainer environment — wraps MCP tool calls."""
    tool_name: str = Field(..., description="Name of the MCP tool to call")
    arguments: Dict = Field(default_factory=dict, description="Tool arguments")


class TrainerObservation(Observation):
    """Observation from the ML Trainer environment."""
    task_id: str = Field(default="", description="Current task ID")
    current_epoch: int = Field(default=0, description="Current training epoch")
    max_epochs: int = Field(default=0, description="Maximum allowed epochs")
    train_loss: float = Field(default=0.0, description="Current training loss")
    val_loss: float = Field(default=0.0, description="Current validation loss")
    train_accuracy: float = Field(default=0.0, description="Current training accuracy")
    val_accuracy: float = Field(default=0.0, description="Current validation accuracy")
    best_val_accuracy: float = Field(default=0.0, description="Best validation accuracy so far")
    convergence_signal: str = Field(default="not_started", description="Training convergence status")
