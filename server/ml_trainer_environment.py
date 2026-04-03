"""
ML Training Optimizer Environment Implementation.

An MCP environment where an AI agent optimizes ML model training by:
- Configuring hyperparameters (optimizer, LR, batch size, etc.)
- Running real PyTorch training epochs
- Observing training curves and metrics
- Adjusting strategy based on convergence signals
- Submitting the best model for grading

All training is REAL — actual forward/backward passes with PyTorch on CPU.
"""

from typing import Any, Optional
from uuid import uuid4

from fastmcp import FastMCP

try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from .trainer import Trainer, TrainingConfig
from .tasks import TASKS, grade_task


class MLTrainerEnvironment(MCPEnvironment):
    """
    An OpenEnv environment for ML training optimization.

    The agent interacts via MCP tools to configure and run real PyTorch
    model training, observing actual training dynamics and making decisions
    to optimize performance.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the ML trainer environment with MCP tools."""
        mcp = FastMCP("ml_trainer_env")
        self._setup_tools(mcp)
        super().__init__(mcp)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id: Optional[str] = None
        self._trainer: Optional[Trainer] = None
        self._max_steps = 15
        self._submitted = False
        self._prev_best_val_acc = 0.0
        self._cumulative_reward = 0.0

    def _setup_tools(self, mcp: FastMCP):
        """Register all MCP tools for the environment."""

        @mcp.tool
        def configure_training(
            optimizer: str = "adam",
            learning_rate: float = 0.001,
            batch_size: int = 64,
            weight_decay: float = 0.0,
            dropout: float = 0.0,
            lr_schedule: str = "constant",
            warmup_epochs: int = 5,
            augmentation: bool = False,
            augmentation_strength: float = 0.5,
        ) -> dict:
            """
            Configure the training hyperparameters. Call this before running
            training or to change the configuration mid-training.

            Note: Changing dropout will reset the model and training progress.
            Other changes (optimizer, LR, batch size, etc.) are applied without
            resetting the model.

            Args:
                optimizer: Optimizer to use. One of: 'sgd', 'adam', 'adamw'
                learning_rate: Learning rate (e.g., 0.001, 0.01, 0.1)
                batch_size: Batch size for training (e.g., 32, 64, 128)
                weight_decay: L2 regularization strength (e.g., 0.0, 0.0001, 0.001)
                dropout: Dropout rate for the model (e.g., 0.0, 0.2, 0.5).
                    WARNING: changing dropout resets the model!
                lr_schedule: Learning rate schedule. One of: 'constant', 'step',
                    'cosine', 'warmup_cosine'
                warmup_epochs: Number of warmup epochs (only for 'warmup_cosine')
                augmentation: Whether to enable data augmentation
                augmentation_strength: Augmentation intensity (0.0 to 1.0)

            Returns:
                Status message and current training metrics
            """
            if self._trainer is None:
                return {"error": "No task loaded. Call reset() first."}
            if self._submitted:
                return {"error": "Model already submitted. Call reset() to start a new episode."}

            config = TrainingConfig(
                optimizer=optimizer,
                learning_rate=learning_rate,
                batch_size=batch_size,
                weight_decay=weight_decay,
                dropout=dropout,
                lr_schedule=lr_schedule,
                warmup_epochs=warmup_epochs,
                augmentation=augmentation,
                augmentation_strength=augmentation_strength,
            )
            status = self._trainer.configure(config)
            metrics = self._trainer.get_metrics_summary()
            return {"status": status, "metrics": metrics}

        @mcp.tool
        def run_epochs(num_epochs: int = 10) -> dict:
            """
            Run N epochs of real PyTorch training with the current configuration.
            Each epoch performs actual forward/backward passes on the training data
            and evaluates on the validation set.

            Training time per epoch varies by task:
            - Easy (MNIST MLP): ~0.5-1 second
            - Medium (FashionMNIST CNN): ~1-2 seconds
            - Hard (CIFAR-10 CNN): ~2-3 seconds

            Args:
                num_epochs: Number of epochs to train (1-20, will be clamped
                    to remaining budget)

            Returns:
                Training metrics after running, including loss, accuracy,
                convergence signal, and remaining budget
            """
            if self._trainer is None:
                return {"error": "No task loaded. Call reset() first."}
            if self._submitted:
                return {"error": "Model already submitted."}
            if not self._trainer._initialized:
                return {"error": "Training not configured. Call configure_training first."}

            num_epochs = max(1, min(20, num_epochs))
            epoch_metrics = self._trainer.train_epochs(num_epochs)

            metrics = self._trainer.get_metrics_summary()
            metrics["epochs_just_trained"] = len(epoch_metrics)
            if epoch_metrics:
                metrics["epoch_time_seconds"] = round(epoch_metrics[-1].epoch_time_seconds, 2)

            return metrics

        @mcp.tool
        def adjust_learning_rate(new_lr: float) -> dict:
            """
            Adjust the learning rate without changing other settings.
            Useful for manual LR scheduling or responding to plateaus.

            Args:
                new_lr: New learning rate value (e.g., 0.0001)

            Returns:
                Status message and current metrics
            """
            if self._trainer is None:
                return {"error": "No task loaded. Call reset() first."}
            if self._submitted:
                return {"error": "Model already submitted."}

            status = self._trainer.adjust_learning_rate(new_lr)
            metrics = self._trainer.get_metrics_summary()
            return {"status": status, "metrics": metrics}

        @mcp.tool
        def toggle_augmentation(enabled: bool = True, strength: float = 0.5) -> dict:
            """
            Toggle data augmentation on/off and set its strength.
            Augmentation adds random transformations to training data to
            reduce overfitting (random rotations, flips, crops, color jitter).

            Args:
                enabled: Whether to enable augmentation
                strength: Augmentation intensity from 0.0 (minimal) to 1.0 (strong)

            Returns:
                Status message and current metrics
            """
            if self._trainer is None:
                return {"error": "No task loaded. Call reset() first."}
            if self._submitted:
                return {"error": "Model already submitted."}
            if self._trainer.state.config is None:
                return {"error": "Call configure_training first."}

            config = self._trainer.state.config
            config.augmentation = enabled
            config.augmentation_strength = max(0.0, min(1.0, strength))
            self._trainer.configure(config)
            metrics = self._trainer.get_metrics_summary()
            return {
                "status": f"Augmentation {'enabled' if enabled else 'disabled'} (strength={strength})",
                "metrics": metrics,
            }

        @mcp.tool
        def get_training_status() -> dict:
            """
            Get the current training status without advancing training.
            Returns all metrics, convergence signal, and remaining budget.

            Returns:
                Current training metrics and configuration
            """
            if self._trainer is None:
                return {"error": "No task loaded. Call reset() first."}

            metrics = self._trainer.get_metrics_summary()
            config = self._trainer.state.config
            if config:
                metrics["current_config"] = {
                    "optimizer": config.optimizer,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "weight_decay": config.weight_decay,
                    "dropout": config.dropout,
                    "lr_schedule": config.lr_schedule,
                    "augmentation": config.augmentation,
                    "augmentation_strength": config.augmentation_strength,
                }
            return metrics

        @mcp.tool
        def submit_model() -> dict:
            """
            Submit the current best model for final grading.
            This ends the episode and returns the final score (0.0 to 1.0).
            The score is based on the best validation accuracy achieved
            during the entire training run, not just the current epoch.

            Returns:
                Final grade with score and detailed breakdown
            """
            if self._trainer is None:
                return {"error": "No task loaded. Call reset() first."}
            if self._submitted:
                return {"error": "Model already submitted."}

            self._submitted = True
            grade = grade_task(self._task_id, self._trainer)
            metrics = self._trainer.get_metrics_summary()

            return {
                "status": "Model submitted for grading",
                "grade": grade,
                "final_metrics": metrics,
            }

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "easy_mnist",
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment for a new training task.

        Args:
            seed: Optional random seed override
            episode_id: Optional episode ID
            task_id: Task to load: 'easy_mnist', 'medium_fashion', or 'hard_cifar'

        Returns:
            Observation with task description and initial state
        """
        if task_id not in TASKS:
            task_id = "easy_mnist"

        task = TASKS[task_id]
        self._task_id = task_id
        self._submitted = False
        self._prev_best_val_acc = 0.0
        self._cumulative_reward = 0.0

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        task_seed = seed if seed is not None else task.seed
        self._trainer = Trainer(
            model_type=task.model_type,
            dataset_name=task.dataset_name,
            max_epochs=task.max_epochs,
            seed=task_seed,
        )

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "ready",
                "task_id": task_id,
                "task_name": task.name,
                "task_description": task.description,
                "difficulty": task.difficulty,
                "model_type": task.model_type,
                "dataset": task.dataset_name,
                "max_epochs": task.max_epochs,
                "max_steps": self._max_steps,
                "target_metric": task.target_metric,
                "target_value": task.target_value,
                "available_tools": [
                    "configure_training", "run_epochs", "adjust_learning_rate",
                    "toggle_augmentation", "get_training_status", "submit_model",
                ],
                "hint": (
                    "Start by calling configure_training to set hyperparameters, "
                    "then call run_epochs to train. Monitor metrics and adjust as needed. "
                    "Call submit_model when satisfied with performance."
                ),
            },
        )

    def _compute_step_reward(self) -> float:
        """Compute reward signal for the current step."""
        if self._trainer is None or not self._trainer.state.val_acc_history:
            return 0.0

        reward = 0.0
        current_best = self._trainer.state.best_val_accuracy
        current_val_acc = self._trainer.state.val_acc_history[-1]

        # 1. Progress reward: val_accuracy improvement
        if current_best > self._prev_best_val_acc:
            improvement = current_best - self._prev_best_val_acc
            reward += 0.3 * improvement
            self._prev_best_val_acc = current_best

        # 2. Loss decrease reward
        if len(self._trainer.state.val_loss_history) >= 2:
            prev_loss = self._trainer.state.val_loss_history[-2]
            curr_loss = self._trainer.state.val_loss_history[-1]
            if curr_loss < prev_loss:
                reward += 0.05

        # 3. Divergence penalty
        if self._trainer.state.is_diverged:
            reward -= 0.2

        # 4. Overfitting penalty
        gap = self._trainer.get_overfitting_gap()
        if gap > 0.08:
            reward -= 0.05 * (gap - 0.08)

        # 5. Wasted compute penalty (for hard task)
        if self._task_id == "hard_cifar":
            wasted = self._trainer.get_wasted_epochs()
            if wasted > 10:
                reward -= 0.02

        return round(reward, 4)

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions (return error)."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step. Routes MCP actions through base class."""
        self._state.step_count += 1

        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Compute reward and check termination
        reward = self._compute_step_reward()
        done = self._submitted or self._state.step_count >= self._max_steps

        # If max steps reached without submission, auto-submit
        if self._state.step_count >= self._max_steps and not self._submitted:
            self._submitted = True
            if self._trainer and self._trainer._initialized:
                grade = grade_task(self._task_id, self._trainer)
                reward += grade["score"]
                obs.metadata = obs.metadata or {}
                obs.metadata["auto_submitted"] = True
                obs.metadata["grade"] = grade

        # Add submission bonus to reward
        if self._submitted and self._trainer and self._trainer._initialized:
            # Check if this is the submission step
            obs_metadata = obs.metadata or {}
            if "grade" in obs_metadata:
                reward += obs_metadata["grade"].get("score", 0.0)

        obs.reward = reward
        obs.done = done
        self._cumulative_reward += reward

        return obs

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Async step for WebSocket handler."""
        self._state.step_count += 1
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)

        reward = self._compute_step_reward()
        done = self._submitted or self._state.step_count >= self._max_steps

        if self._state.step_count >= self._max_steps and not self._submitted:
            self._submitted = True
            if self._trainer and self._trainer._initialized:
                grade = grade_task(self._task_id, self._trainer)
                reward += grade["score"]
                obs.metadata = obs.metadata or {}
                obs.metadata["auto_submitted"] = True
                obs.metadata["grade"] = grade

        obs.reward = reward
        obs.done = done
        self._cumulative_reward += reward

        return obs

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
