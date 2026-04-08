"""
Quota-aware baseline inference script for the ML Training Optimizer environment.

This baseline uses the OpenAI Python client against Gemini's OpenAI-compatible
endpoint and keeps request pacing under a hard RPM ceiling.

Environment Variables:
    API_BASE_URL           - LLM API endpoint
    MODEL_NAME             - model identifier
    HF_TOKEN               - API key for auth
    ENV_URL                - environment server URL (default: http://localhost:8000)
    LLM_RPM_LIMIT          - max model requests per minute (default: 5)
    LLM_MAX_RETRIES        - max rate-limit retries per request (default: 3)
    LLM_REASONING_EFFORT   - reasoning effort sent to the model (default: minimal)
    LLM_MAX_STEPS_EASY     - max model decisions for easy task (default: 5)
    LLM_MAX_STEPS_MEDIUM   - max model decisions for medium task (default: 6)
    LLM_MAX_STEPS_HARD     - max model decisions for hard task (default: 7)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from openenv.core.env_server.mcp_types import CallToolAction

from client import MLTrainerEnv
from server.tasks import TASKS as SERVER_TASKS

load_dotenv()

API_BASE_URL = os.environ.get(
    "API_BASE_URL",
    "https://api.openai.com/v1",
)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
)

ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

LLM_RPM_LIMIT = max(1, int(os.environ.get("LLM_RPM_LIMIT", "5")))
LLM_MAX_RETRIES = max(0, int(os.environ.get("LLM_MAX_RETRIES", "3")))
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "minimal")
LLM_FORCE_TOOL_CHOICE = os.environ.get("LLM_FORCE_TOOL_CHOICE", "true").lower() == "true"
LLM_REASONING_MODELS = ["gemini", "o1", "deepseek-r1"]

LLM_MAX_STEPS = {
    "easy_mnist": max(1, int(os.environ.get("LLM_MAX_STEPS_EASY", "5"))),
    "medium_fashion": max(1, int(os.environ.get("LLM_MAX_STEPS_MEDIUM", "6"))),
    "hard_cifar": max(1, int(os.environ.get("LLM_MAX_STEPS_HARD", "7"))),
}
MIN_REQUEST_GAP_SECONDS = (60.0 / LLM_RPM_LIMIT) + 0.5

TASKS = ["easy_mnist", "medium_fashion", "hard_cifar"]
VALID_TOOLS = {
    "configure_training",
    "run_epochs",
    "adjust_learning_rate",
    "toggle_augmentation",
    "get_training_status",
    "submit_model",
}

RETRY_BACKOFF_SECONDS = [15, 30, 60]

LOCAL_TASK_METADATA = {
    task_id: {
        "task_id": task.task_id,
        "task_name": task.name,
        "task_description": task.description,
        "difficulty": task.difficulty,
        "model_type": task.model_type,
        "dataset": task.dataset_name,
        "max_epochs": task.max_epochs,
        "target_metric": task.target_metric,
        "target_value": task.target_value,
    }
    for task_id, task in SERVER_TASKS.items()
}

SYSTEM_PROMPT = """You are an ML training optimization agent.

You are controlling a real CPU PyTorch training environment through tool calls.
Your objective is to maximize validation quality with as few actions as possible.

Rules:
- Always respond by calling exactly one tool.
- First decision must configure training.
- After run_epochs, use the returned metrics instead of calling get_training_status immediately.
- Prefer larger epoch chunks early to conserve request budget.
- Only adjust learning rate or augmentation when metrics justify it.
- Submit once progress has stalled or the decision budget is nearly exhausted.

Task-specific epoch guidance:
- easy_mnist: prefer run_epochs(10-15) early
- medium_fashion: prefer run_epochs(8-12) early
- hard_cifar: prefer run_epochs(5-8) early
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "configure_training",
            "description": "Configure training hyperparameters before training starts or when retuning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "optimizer": {"type": "string", "enum": ["sgd", "adam", "adamw"]},
                    "learning_rate": {"type": "number"},
                    "batch_size": {"type": "integer", "enum": [32, 64, 128, 256]},
                    "weight_decay": {"type": "number"},
                    "dropout": {"type": "number"},
                    "lr_schedule": {
                        "type": "string",
                        "enum": ["constant", "step", "cosine", "warmup_cosine"],
                    },
                    "warmup_epochs": {"type": "integer"},
                    "augmentation": {"type": "boolean"},
                    "augmentation_strength": {"type": "number"},
                },
                "required": [
                    "optimizer",
                    "learning_rate",
                    "batch_size",
                    "weight_decay",
                    "dropout",
                    "lr_schedule",
                    "warmup_epochs",
                    "augmentation",
                    "augmentation_strength",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_epochs",
            "description": "Run real training for a chunk of epochs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "num_epochs": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["num_epochs"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_learning_rate",
            "description": "Update the live optimizer learning rate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_lr": {"type": "number"},
                },
                "required": ["new_lr"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_augmentation",
            "description": "Enable or disable data augmentation and set strength.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "strength": {"type": "number"},
                },
                "required": ["enabled", "strength"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_training_status",
            "description": "Read the current metrics without advancing training.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_model",
            "description": "Submit the current best model for grading.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


class InferenceError(RuntimeError):
    """Raised when the baseline cannot continue safely."""


@dataclass
class TaskStats:
    """Tracks quota and runtime stats for a single task."""

    requests: int = 0
    retries: int = 0
    decisions: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class RequestScheduler:
    """Ensures model requests stay under the configured RPM limit."""

    min_gap_seconds: float
    rpm_limit: int
    time_fn: Callable[[], float] = time.monotonic
    sleep_fn: Callable[[float], None] = time.sleep
    last_request_started_at: Optional[float] = None
    request_timestamps: deque[float] = None

    def __post_init__(self) -> None:
        if self.request_timestamps is None:
            self.request_timestamps = deque()

    def wait_for_turn(self) -> None:
        """Sleep until the next request is allowed."""
        now = self.time_fn()
        while self.request_timestamps and now - self.request_timestamps[0] >= 60.0:
            self.request_timestamps.popleft()

        waits = []
        if self.last_request_started_at is not None:
            waits.append(self.min_gap_seconds - (now - self.last_request_started_at))
        if len(self.request_timestamps) >= self.rpm_limit:
            waits.append(60.0 - (now - self.request_timestamps[0]) + 0.5)

        wait_seconds = max([w for w in waits if w > 0], default=0.0)
        if wait_seconds > 0:
            self.sleep_fn(wait_seconds)
            now = self.time_fn()
            while self.request_timestamps and now - self.request_timestamps[0] >= 60.0:
                self.request_timestamps.popleft()

        self.last_request_started_at = now
        self.request_timestamps.append(now)


def create_client() -> OpenAI:
    """Create an OpenAI client configured for the selected LLM provider."""
    # OpenRouter specific headers for better visibility and ranking
    extra_headers = {}
    if "openrouter.ai" in API_BASE_URL.lower():
        extra_headers = {
            "HTTP-Referer": "https://openenv.google.github.com",
            "X-Title": "OpenEnv ML Trainer",
        }

    return OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
        default_headers=extra_headers,
    )


def extract_reset_metadata(reset_result: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize reset payloads across OpenEnv HTTP server variants."""
    if not isinstance(reset_result, dict):
        return {}

    observation = reset_result.get("observation")
    if isinstance(observation, dict) and isinstance(observation.get("metadata"), dict):
        metadata = observation["metadata"]
        if metadata:
            return metadata

    if isinstance(observation, dict):
        parsed = extract_result_dict(observation.get("result", {}))
        if parsed:
            return parsed

    if isinstance(reset_result.get("metadata"), dict):
        metadata = reset_result["metadata"]
        if metadata:
            return metadata

    result = reset_result.get("result", {})
    parsed = extract_result_dict(result)
    if parsed:
        return parsed

    return {}


def extract_result_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tool results across OpenEnv HTTP payload variants."""
    if not isinstance(payload, dict):
        return {}

    observation = payload.get("observation", payload)
    if not isinstance(observation, dict):
        return {}

    tool_result = observation.get("result", {})
    parsed = extract_result_dict(tool_result)
    if parsed:
        return parsed

    metadata = observation.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def extract_result_dict(tool_result: Any) -> Dict[str, Any]:
    """Extract structured JSON-ish payloads from OpenEnv tool results."""
    if not isinstance(tool_result, dict):
        return {}

    for key in ("data", "structured_content"):
        value = tool_result.get(key)
        if isinstance(value, dict) and value:
            return value

    content = tool_result.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and parsed:
                return parsed

    return {}


def extract_observation_metadata(observation: Any) -> Dict[str, Any]:
    """Read metadata from either object or dict observations."""
    if isinstance(observation, dict):
        metadata = observation.get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}

    metadata = getattr(observation, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def merge_task_metadata(task_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing task fields from local task definitions."""
    merged = dict(LOCAL_TASK_METADATA.get(task_id, {}))
    merged.update({k: v for k, v in metadata.items() if v not in (None, "", [], {})})
    return merged


def extract_tool_result_from_observation(observation: Any) -> Dict[str, Any]:
    """Read tool result content from either object or dict observations."""
    if isinstance(observation, dict):
        return extract_result_data({"observation": observation})

    # Try the structured result first
    result = getattr(observation, "result", None)
    parsed = extract_result_dict(result)
    if parsed:
        return parsed

    # Fallback: if result itself is already a dict with metrics keys, use it directly
    if isinstance(result, dict) and result:
        return result

    # Fallback: check metadata for embedded tool results
    metadata = getattr(observation, "metadata", None)
    if isinstance(metadata, dict) and metadata:
        # Check if metadata itself contains training metrics
        if any(k in metadata for k in ("current_epoch", "train_loss", "val_accuracy", "best_val_accuracy")):
            return metadata

    return {}


def normalize_tool_result(result_data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten tool payloads so prompts receive a consistent metrics view."""
    if not isinstance(result_data, dict):
        return {}

    final_metrics = result_data.get("final_metrics")
    if isinstance(final_metrics, dict):
        merged = dict(final_metrics)
        if "status" in result_data:
            merged["status"] = result_data["status"]
        if "grade" in result_data:
            merged["grade"] = result_data["grade"]
        return merged

    metrics = result_data.get("metrics")
    if isinstance(metrics, dict):
        merged = dict(metrics)
        if "status" in result_data:
            merged["status"] = result_data["status"]
        if "grade" in result_data:
            merged["grade"] = result_data["grade"]
        if "final_metrics" in result_data:
            merged["final_metrics"] = result_data["final_metrics"]
        return merged

    return result_data


def apply_tool_context(
    tool_name: str,
    arguments: Dict[str, Any],
    latest_data: Dict[str, Any],
    normalized_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Carry forward config state that the environment does not echo on every tool."""
    merged = dict(latest_data)
    merged.update(normalized_data)

    if tool_name == "configure_training":
        merged["current_config"] = dict(arguments)
    elif tool_name == "adjust_learning_rate":
        current_config = dict(merged.get("current_config") or {})
        current_config["learning_rate"] = arguments.get("new_lr")
        merged["current_config"] = current_config
    elif tool_name == "toggle_augmentation":
        current_config = dict(merged.get("current_config") or {})
        current_config["augmentation"] = arguments.get("enabled")
        current_config["augmentation_strength"] = arguments.get("strength")
        merged["current_config"] = current_config

    return merged


def compact_config_summary(data: Dict[str, Any]) -> str:
    """Render the active config into a short deterministic string."""
    config = data.get("current_config")
    if not isinstance(config, dict):
        return "none"

    ordered_keys = [
        "optimizer",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "dropout",
        "lr_schedule",
        "augmentation",
        "augmentation_strength",
    ]
    parts = []
    for key in ordered_keys:
        if key in config:
            parts.append(f"{key}={config[key]}")
    return ", ".join(parts) if parts else "none"


def compact_state_summary(metadata: Dict[str, Any], latest_data: Dict[str, Any]) -> str:
    """Create a compact prompt-friendly environment summary."""
    if latest_data:
        fields = {
            "epoch": latest_data.get("current_epoch", 0),
            "remaining_budget": latest_data.get("remaining_budget", metadata.get("max_epochs", "?")),
            "train_loss": latest_data.get("train_loss", 0.0),
            "val_loss": latest_data.get("val_loss", 0.0),
            "train_accuracy": latest_data.get("train_accuracy", 0.0),
            "val_accuracy": latest_data.get("val_accuracy", 0.0),
            "best_val_accuracy": latest_data.get("best_val_accuracy", 0.0),
            "convergence_signal": latest_data.get("convergence_signal", "not_started"),
            "is_diverged": latest_data.get("is_diverged", False),
            "status": latest_data.get("status"),
            "error": latest_data.get("error"),
        }
        summary = (
            "State: "
            f"epoch={fields['epoch']}, remaining_budget={fields['remaining_budget']}, "
            f"train_loss={fields['train_loss']}, val_loss={fields['val_loss']}, "
            f"train_accuracy={fields['train_accuracy']}, val_accuracy={fields['val_accuracy']}, "
            f"best_val_accuracy={fields['best_val_accuracy']}, "
            f"convergence_signal={fields['convergence_signal']}, "
            f"is_diverged={fields['is_diverged']}, "
            f"current_config={compact_config_summary(latest_data)}"
        )
        if fields["status"]:
            summary += f", status={fields['status']}"
        if fields["error"]:
            summary += f", error={fields['error']}"
        return summary

    return (
        "State: not_started, "
        f"max_epochs={metadata.get('max_epochs', '?')}, "
        f"target={metadata.get('target_metric', 'val_accuracy')} >= {metadata.get('target_value', '?')}, "
        "current_config=none"
    )


def history_block(action_summaries: List[str]) -> str:
    """Render the last two action summaries for the prompt."""
    if not action_summaries:
        return "None"
    return "\n".join(action_summaries[-2:])


def task_brief(metadata: Dict[str, Any]) -> str:
    """Create the stable task brief sent on every request."""
    return (
        f"Task: {metadata.get('task_name', metadata.get('task_id', 'unknown'))}\n"
        f"Description: {metadata.get('task_description', '')}\n"
        f"Difficulty: {metadata.get('difficulty', '')}\n"
        f"Model: {metadata.get('model_type', '')}\n"
        f"Dataset: {metadata.get('dataset', '')}\n"
        f"Max epochs: {metadata.get('max_epochs', '')}\n"
        f"Target: {metadata.get('target_metric', '')} >= {metadata.get('target_value', '')}"
    )


def build_messages(
    metadata: Dict[str, Any],
    latest_data: Dict[str, Any],
    action_summaries: List[str],
    decision_index: int,
    max_decisions: int,
) -> List[Dict[str, str]]:
    """Build the compact chat payload for a single model decision."""
    remaining_decisions = max(0, max_decisions - decision_index)
    user_prompt = (
        f"{task_brief(metadata)}\n\n"
        f"{compact_state_summary(metadata, latest_data)}\n"
        f"Recent actions:\n{history_block(action_summaries)}\n\n"
        f"Decision budget remaining: {remaining_decisions}\n"
        "You are under a hard request limit. Minimize tool calls, prefer larger run_epochs chunks early, "
        "and reserve the final decision for submit_model if needed."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_retry_after_seconds(error: RateLimitError) -> Optional[float]:
    """Read Retry-After from the SDK error response when available."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    raw_value = headers.get("retry-after")
    if raw_value is None:
        return None

    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        return None


def parse_tool_call(message: Any) -> Dict[str, Any]:
    """Extract a single valid tool call from a chat completion message."""
    tool_calls = getattr(message, "tool_calls", None) or []
    if len(tool_calls) != 1:
        raise InferenceError("Expected exactly one tool call from the model.")

    tool_call = tool_calls[0]
    function = getattr(tool_call, "function", None)
    if function is None:
        raise InferenceError("Tool call did not include function details.")

    name = getattr(function, "name", "")
    if name not in VALID_TOOLS:
        raise InferenceError(f"Unsupported tool selected by model: {name}")

    raw_arguments = getattr(function, "arguments", "") or "{}"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise InferenceError(f"Tool arguments were not valid JSON: {raw_arguments}") from exc

    if not isinstance(arguments, dict):
        raise InferenceError("Tool arguments must decode to a JSON object.")

    return {"tool_name": name, "arguments": arguments}


def action_summary(tool_name: str, arguments: Dict[str, Any], result_data: Dict[str, Any], reward: float) -> str:
    """Create a compact history line for prompt reuse."""
    best_val = result_data.get("best_val_accuracy", "-")
    val_acc = result_data.get("val_accuracy", "-")
    signal = result_data.get("convergence_signal", "-")
    remaining_budget = result_data.get("remaining_budget", "-")
    return (
        f"- {tool_name}({json.dumps(arguments, sort_keys=True)}) -> "
        f"best_val_accuracy={best_val}, val_accuracy={val_acc}, "
        f"signal={signal}, remaining_budget={remaining_budget}, reward={reward}"
    )


def model_tool_choice(decision_index: int, max_decisions: int) -> Any:
    """Choose the tool policy for the current decision."""
    if decision_index == 0:
        return {"type": "function", "function": {"name": "configure_training"}}
    if decision_index == max_decisions - 1:
        return {"type": "function", "function": {"name": "submit_model"}}
    return "required"


def request_action(
    client: OpenAI,
    scheduler: RequestScheduler,
    messages: List[Dict[str, str]],
    stats: TaskStats,
    decision_index: int,
    max_decisions: int,
) -> Dict[str, Any]:
    """Request a single model-directed tool call with strict rate-limit handling."""
    for attempt in range(LLM_MAX_RETRIES + 1):
        scheduler.wait_for_turn()
        stats.requests += 1

        # Only use reasoning_effort for known reasoning models to avoid 400s
        call_kwargs = {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "parallel_tool_calls": False,
            "temperature": 0,
            "max_completion_tokens": 200,
        }
        
        # Check if the model likely supports reasoning effort
        if any(rm in MODEL_NAME.lower() for rm in LLM_REASONING_MODELS):
            call_kwargs["reasoning_effort"] = LLM_REASONING_EFFORT

        # Determine tool_choice policy
        preferred_choice = model_tool_choice(decision_index, max_decisions)
        if not LLM_FORCE_TOOL_CHOICE and preferred_choice != "auto":
             preferred_choice = "auto"
        
        try:
            # Attempt with preferred choice
            response = client.chat.completions.create(
                tool_choice=preferred_choice,
                **call_kwargs
            )
            return parse_tool_call(response.choices[0].message)
        except Exception as exc:
            # Check if error is related to tool_choice compatibility
            err_msg = str(exc).lower()
            if ("tool_choice" in err_msg or "endpoints found" in err_msg) and preferred_choice != "auto":
                print(f"  [Model/Provider does not support forced tool choice, falling back to 'auto'...]")
                stats.retries += 1
                try:
                    response = client.chat.completions.create(
                        tool_choice="auto",
                        **call_kwargs
                    )
                    return parse_tool_call(response.choices[0].message)
                except Exception as nested_exc:
                    raise InferenceError(f"Model failed even with tool_choice='auto': {nested_exc}") from nested_exc
            
            if isinstance(exc, RateLimitError):
                if attempt >= LLM_MAX_RETRIES:
                    raise InferenceError(
                        f"Rate limit persisted after {LLM_MAX_RETRIES} retries."
                    ) from exc

                stats.retries += 1
                retry_after = parse_retry_after_seconds(exc)
                backoff = retry_after if retry_after is not None else RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
                print(f"  [Rate limited, waiting {backoff:.1f} seconds before retry...]")
                time.sleep(backoff)
            else:
                raise exc


def run_task(client: OpenAI, scheduler: RequestScheduler, task_id: str) -> Dict[str, Any]:
    """Run one environment task under the quota-aware baseline."""
    max_decisions = LLM_MAX_STEPS[task_id]
    task_stats = TaskStats()
    task_started_at = time.time()

    print(f"[START] task={task_id} env=ml_training_optimizer model={MODEL_NAME}", flush=True)

    rewards: List[float] = []
    total_reward = 0.0
    final_score = 0.0
    steps_taken = 0
    success = False

    try:
        env_instance = MLTrainerEnv(base_url=ENV_URL).sync()
        with env_instance as env:
            try:
                reset_result = env.reset(task_id=task_id)
            except Exception as e:
                # Need to swallow to trigger finally block properly
                print(f"[DEBUG] env.reset() failed: {e}", flush=True)
                raise e

            metadata = merge_task_metadata(task_id, extract_observation_metadata(reset_result.observation))
            latest_data: Dict[str, Any] = {}
            action_summaries: List[str] = []

            for decision_index in range(max_decisions):
                try:
                    messages = build_messages(
                        metadata=metadata,
                        latest_data=latest_data,
                        action_summaries=action_summaries,
                        decision_index=decision_index,
                        max_decisions=max_decisions,
                    )
                    tool_call = request_action(
                        client=client,
                        scheduler=scheduler,
                        messages=messages,
                        stats=task_stats,
                        decision_index=decision_index,
                        max_decisions=max_decisions,
                    )
                except Exception as e:
                    print(f"[DEBUG] model request failed: {e}", flush=True)
                    break
                    
                task_stats.decisions += 1
                tool_name = tool_call["tool_name"]
                arguments = tool_call["arguments"]
                
                args_str = json.dumps(arguments, separators=(',', ':'))
                action_str = f"{tool_name}({args_str})"
                
                error = None
                reward = 0.0
                done = False

                try:
                    step_result = env.step(CallToolAction(tool_name=tool_name, arguments=arguments))
                    observation = step_result.observation
                    reward = step_result.reward or 0.0
                    done = step_result.done
                    
                    result_data = normalize_tool_result(extract_tool_result_from_observation(observation))
                    if result_data:
                        latest_data = apply_tool_context(tool_name, arguments, latest_data, result_data)
                    else:
                        latest_data = apply_tool_context(
                            tool_name,
                            arguments,
                            latest_data,
                            extract_observation_metadata(observation),
                        )
                    
                    # Update score
                    metadata_grade = extract_observation_metadata(observation).get("grade", {})
                    result_grade = latest_data.get("grade", {}) if isinstance(latest_data, dict) else {}
                    if result_grade:
                        final_score = result_grade.get("score") or 0.0
                    elif metadata_grade:
                        final_score = metadata_grade.get("score") or 0.0
                        
                except Exception as exc:
                    error = str(exc)
                    done = True

                rewards.append(reward)
                total_reward += reward
                steps_taken = decision_index + 1
                
                error_val = error if error else "null"
                error_val = error_val.replace('\n', ' ')
                done_val = str(done).lower()
                
                print(f"[STEP] step={steps_taken} action={action_str} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

                action_summaries.append(action_summary(tool_name, arguments, latest_data, reward))

                if done:
                    break
            else:
                # auto-submit
                try:
                    action_str = "submit_model({})"
                    step_result = env.step(CallToolAction(tool_name="submit_model", arguments={}))
                    observation = step_result.observation
                    reward = step_result.reward or 0.0
                    total_reward += reward
                    done = step_result.done
                    
                    result_data = normalize_tool_result(extract_tool_result_from_observation(observation))
                    if result_data:
                        latest_data = apply_tool_context("submit_model", {}, latest_data, result_data)

                    metadata_grade = extract_observation_metadata(observation).get("grade", {})
                    result_grade = latest_data.get("grade", {}) if isinstance(latest_data, dict) else {}
                    if result_grade:
                        final_score = result_grade.get("score") or 0.0
                    elif metadata_grade:
                        final_score = metadata_grade.get("score") or 0.0
                        
                    steps_taken += 1
                    rewards.append(reward)
                    print(f"[STEP] step={steps_taken} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

                except Exception as exc:
                    pass

    except Exception as e:
        print(f"[DEBUG] task container failed: {e}", flush=True)

    finally:
        task_stats.elapsed_seconds = round(time.time() - task_started_at, 1)
        if final_score is None:
            final_score = 0.001
        final_score = min(max(float(final_score), 0.001), 0.999)
        success = final_score >= 0.1
        
        rewards_str = ",".join(f"{float(r):.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={final_score:.3f} rewards={rewards_str}", flush=True)

    return {
        "task_id": task_id,
        "final_score": final_score,
        "total_reward": round(total_reward, 4),
        "llm_decisions": task_stats.decisions,
        "requests": task_stats.requests,
        "retries": task_stats.retries,
        "elapsed_seconds": task_stats.elapsed_seconds,
    }


def main() -> None:
    """Run the baseline across all tasks."""
    if not HF_TOKEN:
        print("ERROR: Set HF_TOKEN, OPENAI_API_KEY, OPENROUTER_API_KEY or GEMINI_API_KEY environment variable")
        sys.exit(1)

    print(f"API Base: {API_BASE_URL}", file=sys.stderr)
    print(f"Model: {MODEL_NAME}", file=sys.stderr)
    print(f"Environment: {ENV_URL}", file=sys.stderr)
    print(f"Tasks: {TASKS}", file=sys.stderr)
    print(f"RPM limit: {LLM_RPM_LIMIT} (min gap {MIN_REQUEST_GAP_SECONDS:.1f}s)", file=sys.stderr)
    print(f"Retries: {LLM_MAX_RETRIES}", file=sys.stderr)

    client = create_client()
    scheduler = RequestScheduler(
        min_gap_seconds=MIN_REQUEST_GAP_SECONDS,
        rpm_limit=LLM_RPM_LIMIT,
    )
    started_at = time.time()
    results = []

    try:
        for task_id in TASKS:
            results.append(run_task(client, scheduler, task_id))
    except InferenceError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    elapsed = round(time.time() - started_at, 1)
    total_requests = sum(result["requests"] for result in results)
    total_retries = sum(result["retries"] for result in results)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print("BASELINE RESULTS", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    for result in results:
        print(
            "  "
            f"{result['task_id']:20s} score={result['final_score']} "
            f"decisions={result['llm_decisions']} requests={result['requests']} "
            f"retries={result['retries']} elapsed={result['elapsed_seconds']}s",
            file=sys.stderr
        )
    print(f"  Total requests: {total_requests}", file=sys.stderr)
    print(f"  Total retries: {total_retries}", file=sys.stderr)
    print(f"  Total elapsed: {elapsed}s ({elapsed / 60:.1f} min)", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    output = {
        "results": results,
        "total_requests": total_requests,
        "total_retries": total_retries,
        "total_time_seconds": elapsed,
        "model": MODEL_NAME,
        "rpm_limit": LLM_RPM_LIMIT,
        "min_request_gap_seconds": round(MIN_REQUEST_GAP_SECONDS, 1),
    }
    print(f"\n{json.dumps(output, indent=2, default=str)}", file=sys.stderr)


if __name__ == "__main__":
    main()
