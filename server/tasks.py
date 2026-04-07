"""
Task definitions and graders for the ML Training Optimizer environment.

Three tasks with increasing difficulty:
1. Easy: MNIST Digit Classifier
2. Medium: Fashion Item Classifier
3. Hard: CIFAR-10 Under Budget

Each task defines the model, dataset, budget, and grading rubric.
Graders produce deterministic scores in [0.0, 1.0].
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class TaskDefinition:
    """Defines a training task."""
    task_id: str
    name: str
    description: str
    difficulty: str
    model_type: str
    dataset_name: str
    max_epochs: int
    seed: int
    target_metric: str
    target_value: float

    # Grading thresholds
    min_score_accuracy: float    # accuracy below this → score = 0
    max_score_accuracy: float    # accuracy above this → score = 1


TASKS: Dict[str, TaskDefinition] = {
    "easy_mnist": TaskDefinition(
        task_id="easy_mnist",
        name="MNIST Digit Classifier",
        description=(
            "Train a 2-layer MLP to classify handwritten digits from MNIST. "
            "You have a subset of 5,000 samples (4k train / 1k validation) and a budget of 100 epochs. "
            "Goal: maximize validation accuracy. Default Adam + LR=0.001 should get ~93%. "
            "With good hyperparameter choices you can reach 97%+."
        ),
        difficulty="easy",
        model_type="simple_mlp",
        dataset_name="mnist",
        max_epochs=100,
        seed=42,
        target_metric="val_accuracy",
        target_value=0.96,
        min_score_accuracy=0.88,
        max_score_accuracy=0.975,
    ),
    "medium_fashion": TaskDefinition(
        task_id="medium_fashion",
        name="Fashion Item Classifier",
        description=(
            "Train a small CNN to classify fashion items from FashionMNIST. "
            "You have 8,000 samples (6.5k train / 1.5k val) and a budget of 80 epochs. "
            "FashionMNIST is harder than MNIST — the small training set means overfitting is a real threat. "
            "Goal: maximize validation accuracy while keeping the overfitting gap (train_acc - val_acc) below 5%. "
            "You'll need proper optimizer selection, learning rate scheduling, and regularization."
        ),
        difficulty="medium",
        model_type="small_cnn",
        dataset_name="fashion_mnist",
        max_epochs=80,
        seed=123,
        target_metric="val_accuracy",
        target_value=0.87,
        min_score_accuracy=0.75,
        max_score_accuracy=0.90,
    ),
    "hard_cifar": TaskDefinition(
        task_id="hard_cifar",
        name="CIFAR-10 Under Budget",
        description=(
            "Train a deeper CNN on CIFAR-10 color images with a tight budget. "
            "You have 10,000 samples (8k train / 2k val) and only 60 epochs. "
            "CIFAR-10 is genuinely challenging for small models. The tight budget means you must "
            "make smart decisions fast — balance exploration (trying configs) vs exploitation "
            "(training with a good config). Overfitting on 8k images is severe without regularization. "
            "Score is based on accuracy (50%), compute efficiency (30%), and training stability (20%)."
        ),
        difficulty="hard",
        model_type="deeper_cnn",
        dataset_name="cifar10",
        max_epochs=60,
        seed=456,
        target_metric="val_accuracy",
        target_value=0.65,
        min_score_accuracy=0.40,
        max_score_accuracy=0.68,
    ),
}


def _clamp(value: float, low: float = 0.0001, high: float = 0.9999) -> float:
    """Clamp a value to [low, high]."""
    return max(low, min(high, value))


def grade_easy(
    val_accuracy: float,
    task: TaskDefinition,
    **kwargs,
) -> Dict:
    """
    Grade the easy MNIST task.

    Score is linear from min_score_accuracy to max_score_accuracy.
    Simple and forgiving — rewards any accuracy above 88%.
    """
    acc_score = _clamp(
        (val_accuracy - task.min_score_accuracy)
        / (task.max_score_accuracy - task.min_score_accuracy)
    )

    return {
        "score": round(acc_score, 4),
        "details": {
            "val_accuracy": round(val_accuracy, 4),
            "accuracy_score": round(acc_score, 4),
        },
    }


def grade_medium(
    val_accuracy: float,
    overfitting_gap: float,
    task: TaskDefinition,
    **kwargs,
) -> Dict:
    """
    Grade the medium FashionMNIST task.

    Score = 60% accuracy + 40% generalization.
    Penalizes overfitting (train_acc - val_acc > 12%).
    """
    acc_score = _clamp(
        (val_accuracy - task.min_score_accuracy)
        / (task.max_score_accuracy - task.min_score_accuracy)
    )

    gen_score = _clamp(1.0 - overfitting_gap / 0.12)

    combined = 0.6 * acc_score + 0.4 * gen_score

    return {
        "score": round(combined, 4),
        "details": {
            "val_accuracy": round(val_accuracy, 4),
            "overfitting_gap": round(overfitting_gap, 4),
            "accuracy_score": round(acc_score, 4),
            "generalization_score": round(gen_score, 4),
        },
    }


def grade_hard(
    val_accuracy: float,
    wasted_epochs: int,
    budget: int,
    loss_variance: float,
    task: TaskDefinition,
    **kwargs,
) -> Dict:
    """
    Grade the hard CIFAR-10 task.

    Score = 50% accuracy + 30% efficiency + 20% stability.
    Rewards good accuracy, penalizes wasted compute and unstable training.
    """
    acc_score = _clamp(
        (val_accuracy - task.min_score_accuracy)
        / (task.max_score_accuracy - task.min_score_accuracy)
    )

    efficiency_score = _clamp(1.0 - wasted_epochs / max(1, budget))

    stability_threshold = 0.01
    stability_score = _clamp(1.0 - loss_variance / stability_threshold)

    combined = 0.5 * acc_score + 0.3 * efficiency_score + 0.2 * stability_score

    return {
        "score": round(combined, 4),
        "details": {
            "val_accuracy": round(val_accuracy, 4),
            "wasted_epochs": wasted_epochs,
            "loss_variance": round(loss_variance, 6),
            "accuracy_score": round(acc_score, 4),
            "efficiency_score": round(efficiency_score, 4),
            "stability_score": round(stability_score, 4),
        },
    }


def grade_task(task_id: str, trainer) -> Dict:
    """
    Grade a completed task using the trainer's final state.

    Args:
        task_id: Task identifier
        trainer: Trainer instance with completed training

    Returns:
        Dict with 'score' (0.0–1.0) and 'details'
    """
    task = TASKS[task_id]
    best_val_acc = trainer.state.best_val_accuracy

    if task_id == "easy_mnist":
        return grade_easy(best_val_acc, task)

    elif task_id == "medium_fashion":
        gap = trainer.get_overfitting_gap()
        return grade_medium(best_val_acc, gap, task)

    elif task_id == "hard_cifar":
        wasted = trainer.get_wasted_epochs()
        variance = trainer.get_loss_variance()
        return grade_hard(
            best_val_acc, wasted, task.max_epochs, variance, task,
        )

    else:
        raise ValueError(f"Unknown task: {task_id}")
