"""
Real PyTorch Training Engine for the ML Training Optimizer environment.

Manages the full training lifecycle:
- Model creation with configurable architecture and dropout
- Optimizer creation (SGD, Adam, AdamW) with configurable hyperparameters
- Learning rate scheduling (constant, step, cosine, warmup_cosine)
- Data augmentation control
- Training loop execution
- Evaluation and metric tracking
- Best model checkpointing
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR

from .models_nn import create_model
from .datasets import create_dataloaders


@dataclass
class TrainingConfig:
    """Configuration for a training run, set by the agent."""
    optimizer: str = "adam"          # sgd, adam, adamw
    learning_rate: float = 0.001
    batch_size: int = 64
    weight_decay: float = 0.0
    dropout: float = 0.0
    lr_schedule: str = "constant"   # constant, step, cosine, warmup_cosine
    warmup_epochs: int = 5
    augmentation: bool = False
    augmentation_strength: float = 0.5


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    current_lr: float = 0.0
    epoch_time_seconds: float = 0.0


@dataclass
class TrainingState:
    """Full internal state of the trainer."""
    current_epoch: int = 0
    total_epochs_run: int = 0
    best_val_accuracy: float = 0.0
    best_val_epoch: int = 0
    train_loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    train_acc_history: List[float] = field(default_factory=list)
    val_acc_history: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    is_diverged: bool = False
    config: Optional[TrainingConfig] = None


class Trainer:
    """
    Manages real PyTorch model training on CPU.

    Designed to be used by the environment: the agent configures hyperparameters,
    then calls train_epochs() to actually train the model. The trainer tracks
    all metrics and maintains the best model checkpoint.
    """

    def __init__(
        self,
        model_type: str,
        dataset_name: str,
        max_epochs: int,
        seed: int = 42,
    ):
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.max_epochs = max_epochs
        self.seed = seed

        # Set deterministic behavior
        torch.manual_seed(seed)
        torch.set_num_threads(2)  # Match 2 vCPU constraint
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass  # Some ops don't have deterministic impl

        self.state = TrainingState()
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.best_model_state: Optional[dict] = None
        self.device = torch.device("cpu")
        self.criterion = nn.CrossEntropyLoss()

        self._initialized = False

    def configure(self, config: TrainingConfig) -> str:
        """
        Configure (or reconfigure) the training setup.

        Creates/rebuilds model, optimizer, scheduler, and data loaders
        based on the provided configuration.

        Returns:
            Status message describing what was configured.
        """
        old_config = self.state.config
        self.state.config = config

        # Rebuild model if dropout changed or first time
        rebuild_model = (
            not self._initialized
            or old_config is None
            or old_config.dropout != config.dropout
        )

        if rebuild_model:
            torch.manual_seed(self.seed)
            self.model = create_model(self.model_type, dropout=config.dropout)
            self.model.to(self.device)
            self.state.current_epoch = 0
            self.state.total_epochs_run = 0
            self.state.best_val_accuracy = 0.0
            self.state.best_val_epoch = 0
            self.state.train_loss_history = []
            self.state.val_loss_history = []
            self.state.train_acc_history = []
            self.state.val_acc_history = []
            self.state.lr_history = []
            self.state.is_diverged = False
            self.best_model_state = None

        # Rebuild data loaders if batch size or augmentation changed
        rebuild_data = (
            not self._initialized
            or old_config is None
            or old_config.batch_size != config.batch_size
            or old_config.augmentation != config.augmentation
            or old_config.augmentation_strength != config.augmentation_strength
        )

        if rebuild_data:
            self.train_loader, self.val_loader, _, _ = create_dataloaders(
                self.dataset_name,
                batch_size=config.batch_size,
                seed=self.seed,
                augment=config.augmentation,
                aug_strength=config.augmentation_strength,
            )

        # Always rebuild optimizer (LR, weight decay, or optimizer type may change)
        self._build_optimizer(config)

        # Always rebuild scheduler
        remaining = max(1, self.max_epochs - self.state.current_epoch)
        self._build_scheduler(config, remaining)

        self._initialized = True

        status_parts = []
        if rebuild_model:
            param_count = sum(p.numel() for p in self.model.parameters())
            status_parts.append(f"Model created: {self.model_type} ({param_count:,} params, dropout={config.dropout})")
        if rebuild_data:
            status_parts.append(f"Data loaded: {self.dataset_name} (batch_size={config.batch_size}, aug={'on' if config.augmentation else 'off'})")
        status_parts.append(f"Optimizer: {config.optimizer} (lr={config.learning_rate}, wd={config.weight_decay})")
        status_parts.append(f"Schedule: {config.lr_schedule}")

        return " | ".join(status_parts)

    def _build_optimizer(self, config: TrainingConfig):
        """Create optimizer from config."""
        params = self.model.parameters()
        if config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=config.learning_rate,
                weight_decay=config.weight_decay, momentum=0.9,
            )
        elif config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params, lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                params, lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

    def _build_scheduler(self, config: TrainingConfig, total_epochs: int):
        """Create learning rate scheduler from config."""
        if config.lr_schedule == "constant":
            self.scheduler = None
        elif config.lr_schedule == "step":
            self.scheduler = StepLR(self.optimizer, step_size=max(1, total_epochs // 3), gamma=0.1)
        elif config.lr_schedule == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_epochs)
        elif config.lr_schedule == "warmup_cosine":
            warmup = config.warmup_epochs

            def lr_lambda(epoch):
                if epoch < warmup:
                    return (epoch + 1) / warmup
                progress = (epoch - warmup) / max(1, total_epochs - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown LR schedule: {config.lr_schedule}")

    def adjust_learning_rate(self, new_lr: float) -> str:
        """Adjust the learning rate on the live optimizer."""
        if self.optimizer is None:
            return "Error: No optimizer configured. Call configure_training first."
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        if self.state.config:
            self.state.config.learning_rate = new_lr
        return f"Learning rate adjusted to {new_lr}"

    def train_epochs(self, num_epochs: int) -> List[TrainingMetrics]:
        """
        Train for num_epochs epochs. Returns metrics for each epoch.

        This runs REAL PyTorch training — forward pass, loss computation,
        backward pass, optimizer step — on actual data.
        """
        if not self._initialized or self.model is None:
            raise RuntimeError("Trainer not configured. Call configure() first.")

        if self.state.is_diverged:
            return [TrainingMetrics(
                train_loss=float('inf'), val_loss=float('inf'),
                train_accuracy=0.0, val_accuracy=0.0,
                current_lr=0.0, epoch_time_seconds=0.0,
            )]

        # Clamp to budget
        remaining = self.max_epochs - self.state.current_epoch
        actual_epochs = min(num_epochs, remaining)
        if actual_epochs <= 0:
            return []

        all_metrics = []

        for _ in range(actual_epochs):
            epoch_start = time.time()

            # --- Training phase ---
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_loss = running_loss / total
            train_acc = correct / total

            # Check for divergence
            if math.isnan(train_loss) or math.isinf(train_loss) or train_loss > 100:
                self.state.is_diverged = True
                metrics = TrainingMetrics(
                    train_loss=float('inf'), val_loss=float('inf'),
                    train_accuracy=0.0, val_accuracy=0.0,
                    current_lr=self.optimizer.param_groups[0]['lr'],
                    epoch_time_seconds=time.time() - epoch_start,
                )
                all_metrics.append(metrics)
                break

            # --- Validation phase ---
            val_loss, val_acc = self._evaluate()

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']

            # Update state
            self.state.current_epoch += 1
            self.state.total_epochs_run += 1
            self.state.train_loss_history.append(train_loss)
            self.state.val_loss_history.append(val_loss)
            self.state.train_acc_history.append(train_acc)
            self.state.val_acc_history.append(val_acc)
            self.state.lr_history.append(current_lr)

            # Track best model
            if val_acc > self.state.best_val_accuracy:
                self.state.best_val_accuracy = val_acc
                self.state.best_val_epoch = self.state.current_epoch
                self.best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }

            epoch_time = time.time() - epoch_start
            metrics = TrainingMetrics(
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                current_lr=current_lr,
                epoch_time_seconds=epoch_time,
            )
            all_metrics.append(metrics)

        return all_metrics

    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate model on validation set. Returns (val_loss, val_accuracy)."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return running_loss / total, correct / total

    def get_metrics_summary(self) -> Dict:
        """Get current training metrics as a dictionary."""
        if not self.state.train_loss_history:
            return {
                "current_epoch": 0,
                "max_epochs": self.max_epochs,
                "remaining_budget": self.max_epochs,
                "train_loss": 0.0,
                "val_loss": 0.0,
                "train_accuracy": 0.0,
                "val_accuracy": 0.0,
                "best_val_accuracy": 0.0,
                "best_val_epoch": 0,
                "loss_history_last_10": [],
                "val_loss_history_last_10": [],
                "convergence_signal": "not_started",
                "is_diverged": False,
            }

        # Determine convergence signal
        signal = self._get_convergence_signal()

        return {
            "current_epoch": self.state.current_epoch,
            "max_epochs": self.max_epochs,
            "remaining_budget": self.max_epochs - self.state.current_epoch,
            "train_loss": round(self.state.train_loss_history[-1], 6),
            "val_loss": round(self.state.val_loss_history[-1], 6),
            "train_accuracy": round(self.state.train_acc_history[-1], 4),
            "val_accuracy": round(self.state.val_acc_history[-1], 4),
            "best_val_accuracy": round(self.state.best_val_accuracy, 4),
            "best_val_epoch": self.state.best_val_epoch,
            "loss_history_last_10": [round(x, 4) for x in self.state.train_loss_history[-10:]],
            "val_loss_history_last_10": [round(x, 4) for x in self.state.val_loss_history[-10:]],
            "convergence_signal": signal,
            "is_diverged": self.state.is_diverged,
        }

    def _get_convergence_signal(self) -> str:
        """Analyze recent training dynamics to produce a human-readable signal."""
        if self.state.is_diverged:
            return "diverged"

        if len(self.state.val_loss_history) < 3:
            return "warming_up"

        recent_val = self.state.val_loss_history[-5:]
        recent_train = self.state.train_loss_history[-5:]

        # Check overfitting: train improving but val getting worse
        if len(recent_val) >= 3:
            val_trend = recent_val[-1] - recent_val[0]
            train_trend = recent_train[-1] - recent_train[0]
            if val_trend > 0 and train_trend < 0:
                return "overfitting"

        # Check plateau: val loss barely changing
        if len(recent_val) >= 3:
            val_range = max(recent_val) - min(recent_val)
            if val_range < 0.005:
                return "plateaued"

        # Check if improving
        if recent_val[-1] < recent_val[0]:
            return "improving"

        return "stalling"

    def get_overfitting_gap(self) -> float:
        """Returns train_accuracy - val_accuracy (overfitting indicator)."""
        if not self.state.train_acc_history:
            return 0.0
        return self.state.train_acc_history[-1] - self.state.val_acc_history[-1]

    def get_wasted_epochs(self) -> int:
        """Epochs trained after best val accuracy was achieved."""
        if self.state.best_val_epoch == 0:
            return 0
        return max(0, self.state.current_epoch - self.state.best_val_epoch)

    def get_loss_variance(self, window: int = 10) -> float:
        """Variance of recent validation losses."""
        if len(self.state.val_loss_history) < 2:
            return 0.0
        recent = self.state.val_loss_history[-window:]
        mean = sum(recent) / len(recent)
        return sum((x - mean) ** 2 for x in recent) / len(recent)
