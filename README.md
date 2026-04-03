---
title: ML Training Optimizer
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---

# ML Training Optimizer — OpenEnv Environment

An OpenEnv environment where AI agents learn to optimize the training of real ML/DL models by tuning hyperparameters, selecting optimizers, managing learning rate schedules, and applying regularization techniques.

**This environment trains REAL PyTorch models on CPU** — not simulations. The agent observes actual training curves, loss values, and validation metrics from real forward/backward passes, then decides what to change next.

This models a real workflow ML practitioners perform every day: hyperparameter tuning under limited compute, noisy validation curves, and real overfitting risk.

## Motivation

ML practitioners spend enormous time on hyperparameter tuning. This environment recreates that workflow:
- Agent observes training metrics (loss, accuracy, convergence signals)
- Agent decides what to change (optimizer, LR, regularization, etc.)
- Agent runs more training and iterates

The small dataset subsets (5k–10k samples) make overfitting a **real, tangible problem** the agent must address — exactly like real low-data regimes practitioners face daily.

## Vision & Scalability

The long-term vision for this environment is to **teach AI agents to monitor and optimize the training of large-scale models on distributed systems** — multi-GPU clusters, sharded data pipelines, and fault-tolerant training loops. In production ML, human engineers spend significant time babysitting training runs: watching for loss spikes, adjusting learning rates, restarting from checkpoints, and rebalancing resources across nodes. An agent that masters these skills could dramatically accelerate the development cycle of foundation models.

To fit within current compute constraints (and the OpenEnv specification), the environment currently operates on small models trainable on standard CPUs. However, the core abstractions — observing training curves, adjusting hyperparameters mid-run, detecting convergence/divergence, and deciding when to stop — are **identical to those required at scale**. An agent that learns effective optimization strategies here can transfer those skills to larger, distributed settings as the environment scales up.

## Tasks

### Task 1: MNIST Digit Classifier (Easy)
- **Model**: 2-layer MLP (~100k params)
- **Dataset**: MNIST 5k subset (4k train / 1k val)
- **Budget**: 100 epochs
- **Goal**: Maximize validation accuracy (target ≥ 96%)
- **Grading**: Linear scale 88%→97.5% → score 0.0→1.0

### Task 2: Fashion Item Classifier (Medium)
- **Model**: Small CNN (~200k params)
- **Dataset**: FashionMNIST 8k subset (6.5k train / 1.5k val)
- **Budget**: 80 epochs
- **Goal**: Maximize accuracy while keeping overfitting gap < 5%
- **Grading**: 60% accuracy score + 40% generalization score

### Task 3: CIFAR-10 Under Budget (Hard)
- **Model**: Deeper CNN (~500k params)
- **Dataset**: CIFAR-10 10k subset (8k train / 2k val)
- **Budget**: 60 epochs
- **Goal**: Maximize accuracy under tight budget
- **Grading**: 50% accuracy + 30% efficiency + 20% stability

## Action Space (MCP Tools)

| Tool | Parameters | Description |
|---|---|---|
| `configure_training` | optimizer, learning_rate, batch_size, weight_decay, dropout, lr_schedule, warmup_epochs, augmentation, augmentation_strength | Set/update training config |
| `run_epochs` | num_epochs (1–20) | Run N epochs of real PyTorch training |
| `adjust_learning_rate` | new_lr | Change LR mid-training |
| `toggle_augmentation` | enabled, strength | Toggle data augmentation |
| `get_training_status` | — | Query current metrics |
| `submit_model` | — | Submit for final grading |

### Configuration Options

**Optimizers**: `sgd` (with momentum=0.9), `adam`, `adamw`

**LR Schedules**: `constant`, `step` (decay by 0.1 every T/3 epochs), `cosine` (cosine annealing), `warmup_cosine` (linear warmup + cosine)

**Regularization**: `weight_decay` (L2), `dropout` (0.0–0.5), `augmentation` (random transforms)

**Batch Sizes**: 32, 64, 128, 256

## Observation Space

After each action, the agent receives:
```json
{
  "current_epoch": 30,
  "max_epochs": 100,
  "remaining_budget": 70,
  "train_loss": 0.342,
  "val_loss": 0.401,
  "train_accuracy": 0.891,
  "val_accuracy": 0.864,
  "best_val_accuracy": 0.871,
  "best_val_epoch": 25,
  "loss_history_last_10": [0.45, 0.43, ...],
  "val_loss_history_last_10": [0.52, 0.49, ...],
  "convergence_signal": "improving",
  "is_diverged": false
}
```

**Convergence signals**: `not_started`, `warming_up`, `improving`, `plateaued`, `overfitting`, `stalling`, `diverged`

## Reward Function

Rewards per step (not just at the end):
- **Progress reward**: +0.3 × accuracy improvement above previous best
- **Convergence reward**: +0.05 for decreasing validation loss
- **Divergence penalty**: −0.2 if training diverges
- **Overfitting penalty**: −0.05 × excess when gap > 8%
- **Submission bonus**: Final grader score (0.0–1.0) added on submit

## Setup & Usage

### Install
```bash
uv sync
```

### Run the server locally
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run with Docker
```bash
docker build -f server/Dockerfile -t ml-trainer-env .
docker run -p 8000:8000 ml-trainer-env
```

### Run the baseline inference
This baseline uses the OpenAI API by default. With the default `LLM_RPM_LIMIT=5`, it spaces requests to stay under free-tier quotas and uses a small, quota-aware decision budget per task.

Recommended `.env`:
```bash
OPENAI_API_KEY=sk-proj-...
MODEL_NAME=gpt-4o-mini
ENV_URL=http://localhost:8000
LLM_RPM_LIMIT=5
LLM_MAX_RETRIES=3
LLM_REASONING_EFFORT=minimal
LLM_MAX_STEPS_EASY=5
LLM_MAX_STEPS_MEDIUM=6
LLM_MAX_STEPS_HARD=7
```

Then run:
```bash
export ENV_URL=http://localhost:8000
uv run inference.py
```
The script uses the OpenAI Python client against the official OpenAI API by default. You can also point it at other OpenAI-compatible providers (like OpenRouter or Gemini) by setting corresponding `API_BASE_URL`, `OPENROUTER_API_KEY`, or `GEMINI_API_KEY` in your `.env`.

### Required environment variables

| `API_BASE_URL` | no (defaults to OpenAI) | LLM API endpoint |
| `MODEL_NAME` | yes | Model identifier (default: gpt-4o-mini) |
| `ENV_URL` | yes | URL of the running OpenEnv environment |
| `OPENAI_API_KEY` | yes | Auth for OpenAI (preferred) |
| `OPENROUTER_API_KEY` or `GEMINI_API_KEY` | yes | Fallback Auth for alternative providers |
| `HF_TOKEN` | needed for HF deployment workflows | Hugging Face auth token |

### Optional inference tuning variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_RPM_LIMIT` | `5` | Hard request cap used by the scheduler |
| `LLM_MAX_RETRIES` | `3` | Rate-limit retries per model request |
| `LLM_REASONING_EFFORT` | `minimal` | Gemini reasoning effort |
| `LLM_MAX_STEPS_EASY` | `5` | Max model decisions for `easy_mnist` |
| `LLM_MAX_STEPS_MEDIUM` | `6` | Max model decisions for `medium_fashion` |
| `LLM_MAX_STEPS_HARD` | `7` | Max model decisions for `hard_cifar` |

### Interact via Python client
```python
from ml_trainer_env import MLTrainerEnv

with MLTrainerEnv(base_url="http://localhost:8000") as env:
    env.reset(task_id="easy_mnist")
    tools = env.list_tools()
    
    result = env.call_tool("configure_training",
        optimizer="adam", learning_rate=0.001, batch_size=64)
    
    result = env.call_tool("run_epochs", num_epochs=10)
    print(result)  # Real training metrics!
    
    result = env.call_tool("submit_model")
    print(result)  # Final score
```

## Baseline Scores

| Task | Expected Score Range | Notes |
|---|---|---|
| easy_mnist | 0.6 – 0.9 | Most models solve this well |
| medium_fashion | 0.4 – 0.7 | Requires regularization awareness |
| hard_cifar | 0.2 – 0.5 | Genuinely challenging under budget |

Scores are reported as expected ranges rather than exact fixed values because training remains real, even though seeds and data subsets are deterministic.

## Architecture

```
openenv-hack/
├── __init__.py           # Package exports
├── models.py             # Pydantic Action/Observation models
├── client.py             # MCPToolClient subclass
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml        # Dependencies
├── inference.py          # Baseline inference script
├── README.md
├── server/
│   ├── app.py            # FastAPI server
│   ├── ml_trainer_environment.py  # MCPEnvironment with tools
│   ├── trainer.py        # Real PyTorch training engine
│   ├── models_nn.py      # Neural network architectures
│   ├── datasets.py       # Dataset loading & subsetting
│   ├── tasks.py          # Task definitions & graders
│   └── Dockerfile
└── outputs/
    ├── logs/
    └── evals/
```

## Technical Details

- **Real training**: Actual PyTorch forward/backward passes on CPU
- **Deterministic**: `torch.manual_seed()` ensures reproducible results
- **Constrained**: `torch.set_num_threads(2)` matches 2 vCPU limit
- **Fast**: ~0.5–3s per epoch depending on task
- **Pre-cached**: Datasets downloaded at Docker build time
- **Quota-aware baseline**: `inference.py` is optimized for low-RPM Gemini quotas and uses function calling with compact state summaries
