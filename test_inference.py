from types import SimpleNamespace

import pytest

import inference


def make_tool_message(name: str, arguments: str):
    return SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(name=name, arguments=arguments)
            )
        ]
    )


def test_request_scheduler_enforces_min_gap():
    current_time = {"value": 100.0}
    sleeps = []

    def fake_time():
        return current_time["value"]

    def fake_sleep(seconds: float):
        sleeps.append(seconds)
        current_time["value"] += seconds

    scheduler = inference.RequestScheduler(
        min_gap_seconds=12.5,
        rpm_limit=5,
        time_fn=fake_time,
        sleep_fn=fake_sleep,
    )

    scheduler.wait_for_turn()
    current_time["value"] += 2.0
    scheduler.wait_for_turn()

    assert sleeps == [pytest.approx(10.5)]


def test_parse_tool_call_accepts_valid_tool_call():
    message = make_tool_message("run_epochs", '{"num_epochs": 8}')

    tool_call = inference.parse_tool_call(message)

    assert tool_call == {"tool_name": "run_epochs", "arguments": {"num_epochs": 8}}


def test_parse_tool_call_rejects_text_only_response():
    message = SimpleNamespace(tool_calls=[])

    with pytest.raises(inference.InferenceError, match="exactly one tool call"):
        inference.parse_tool_call(message)


def test_parse_tool_call_rejects_malformed_arguments():
    message = make_tool_message("run_epochs", "{bad json")

    with pytest.raises(inference.InferenceError, match="valid JSON"):
        inference.parse_tool_call(message)


def test_request_action_retries_rate_limit_with_retry_after(monkeypatch):
    sleeps = []
    monkeypatch.setattr(inference.time, "sleep", lambda seconds: sleeps.append(seconds))

    class DummyRateLimitError(Exception):
        def __init__(self, retry_after: str):
            self.response = SimpleNamespace(headers={"retry-after": retry_after})

    monkeypatch.setattr(inference, "RateLimitError", DummyRateLimitError)

    responses = [
        DummyRateLimitError("7"),
        SimpleNamespace(
            choices=[SimpleNamespace(message=make_tool_message("submit_model", "{}"))]
        ),
    ]

    class FakeCompletions:
        def create(self, **kwargs):
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    scheduler = inference.RequestScheduler(
        min_gap_seconds=0.0,
        rpm_limit=5,
        time_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    )
    stats = inference.TaskStats()

    tool_call = inference.request_action(
        client=client,
        scheduler=scheduler,
        messages=[{"role": "user", "content": "test"}],
        stats=stats,
        decision_index=1,
        max_decisions=3,
    )

    assert tool_call["tool_name"] == "submit_model"
    assert stats.requests == 2
    assert stats.retries == 1
    assert sleeps == [7.0]


def test_extract_reset_metadata_reads_wrapped_result_payload():
    payload = {
        "observation": {
            "metadata": {},
            "result": {
                "data": {
                    "task_name": "MNIST Digit Classifier",
                    "difficulty": "easy",
                    "dataset": "mnist",
                    "max_epochs": 100,
                }
            },
        }
    }

    metadata = inference.extract_reset_metadata(payload)

    assert metadata["task_name"] == "MNIST Digit Classifier"
    assert metadata["difficulty"] == "easy"


def test_extract_result_data_reads_json_content_text():
    payload = {
        "observation": {
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": '{"status":"configured","metrics":{"current_epoch":0,"remaining_budget":100,"train_loss":0.0,"val_loss":0.0,"train_accuracy":0.0,"val_accuracy":0.0,"best_val_accuracy":0.0,"convergence_signal":"not_started","is_diverged":false}}',
                    }
                ]
            }
        }
    }

    result = inference.extract_result_data(payload)
    normalized = inference.normalize_tool_result(result)

    assert normalized["current_epoch"] == 0
    assert normalized["remaining_budget"] == 100
    assert normalized["status"] == "configured"


def test_run_task_forces_configure_then_submit(monkeypatch):
    monkeypatch.setitem(inference.LLM_MAX_STEPS, "easy_mnist", 2)

    tool_choices = []
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=make_tool_message(
                        "configure_training",
                        '{"optimizer":"adam","learning_rate":0.001,"batch_size":64,"weight_decay":0.0,"dropout":0.0,"lr_schedule":"cosine","warmup_epochs":3,"augmentation":false,"augmentation_strength":0.0}',
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(message=make_tool_message("submit_model", "{}"))]
        ),
    ]

    class FakeCompletions:
        def create(self, **kwargs):
            tool_choices.append(kwargs["tool_choice"])
            return responses.pop(0)

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    scheduler = inference.RequestScheduler(
        min_gap_seconds=0.0,
        rpm_limit=5,
        time_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    )

    class FakeEnv:
        def sync(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def reset(self, **kwargs):
            return SimpleNamespace(
                observation=SimpleNamespace(
                    metadata={
                        "task_id": kwargs["task_id"],
                        "task_name": "MNIST Digit Classifier",
                        "task_description": "desc",
                        "difficulty": "easy",
                        "model_type": "simple_mlp",
                        "dataset": "mnist",
                        "max_epochs": 100,
                        "target_metric": "val_accuracy",
                        "target_value": 0.96,
                    }
                )
            )

        def step(self, action):
            if action.tool_name == "configure_training":
                return SimpleNamespace(
                    observation=SimpleNamespace(
                        metadata={},
                        result={
                            "data": {
                                "status": "configured",
                                "metrics": {
                                    "current_epoch": 0,
                                    "remaining_budget": 100,
                                    "train_loss": 0.0,
                                    "val_loss": 0.0,
                                    "train_accuracy": 0.0,
                                    "val_accuracy": 0.0,
                                    "best_val_accuracy": 0.0,
                                    "convergence_signal": "not_started",
                                    "is_diverged": False,
                                    "current_config": {
                                        "optimizer": "adam",
                                        "learning_rate": 0.001,
                                    },
                                },
                            }
                        },
                    ),
                    reward=0.0,
                    done=False,
                )

            return SimpleNamespace(
                observation=SimpleNamespace(
                    metadata={},
                    result={"data": {"grade": {"score": 0.8}}},
                ),
                reward=0.8,
                done=True,
            )

    monkeypatch.setattr(inference, "MLTrainerEnv", lambda base_url: FakeEnv())

    result = inference.run_task(client, scheduler, "easy_mnist")

    assert tool_choices[0] == {"type": "function", "function": {"name": "configure_training"}}
    assert tool_choices[-1] == {"type": "function", "function": {"name": "submit_model"}}
    assert result["llm_decisions"] == 2
    assert result["final_score"] == 0.8


def test_run_task_reuses_same_env_session(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=make_tool_message("submit_model", "{}"))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    scheduler = inference.RequestScheduler(
        min_gap_seconds=0.0,
        rpm_limit=5,
        time_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    )

    monkeypatch.setitem(inference.LLM_MAX_STEPS, "easy_mnist", 1)

    class FakeEnv:
        def sync(self):
            return self

        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append("exit")
            return None

        def reset(self, **kwargs):
            calls.append(("reset", kwargs["task_id"]))
            return SimpleNamespace(
                observation=SimpleNamespace(
                    metadata={
                        "task_id": kwargs["task_id"],
                        "task_name": "MNIST Digit Classifier",
                        "difficulty": "easy",
                        "dataset": "mnist",
                        "max_epochs": 100,
                        "target_metric": "val_accuracy",
                        "target_value": 0.96,
                    }
                )
            )

        def step(self, action):
            calls.append(("step", action.tool_name))
            return SimpleNamespace(
                observation=SimpleNamespace(
                    metadata={},
                    result={"data": {"grade": {"score": 0.7}}},
                ),
                reward=0.7,
                done=True,
            )

    monkeypatch.setattr(inference, "MLTrainerEnv", lambda base_url: FakeEnv())

    result = inference.run_task(client, scheduler, "easy_mnist")

    assert result["final_score"] == 0.7
    assert calls == ["enter", ("reset", "easy_mnist"), ("step", "submit_model"), "exit"]


def test_request_action_does_not_send_seed(monkeypatch):
    captured_kwargs = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=make_tool_message("submit_model", "{}"))]
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    scheduler = inference.RequestScheduler(
        min_gap_seconds=0.0,
        rpm_limit=5,
        time_fn=lambda: 0.0,
        sleep_fn=lambda seconds: None,
    )
    stats = inference.TaskStats()

    inference.request_action(
        client=client,
        scheduler=scheduler,
        messages=[{"role": "user", "content": "test"}],
        stats=stats,
        decision_index=1,
        max_decisions=3,
    )

    assert "seed" not in captured_kwargs


def test_apply_tool_context_preserves_current_config():
    configured = inference.apply_tool_context(
        "configure_training",
        {"optimizer": "adam", "learning_rate": 0.001, "batch_size": 64},
        {},
        {"current_epoch": 0, "remaining_budget": 100},
    )

    assert configured["current_config"]["optimizer"] == "adam"

    updated = inference.apply_tool_context(
        "run_epochs",
        {"num_epochs": 10},
        configured,
        {"current_epoch": 10, "best_val_accuracy": 0.95},
    )

    assert updated["current_config"]["batch_size"] == 64
    assert updated["best_val_accuracy"] == 0.95


def test_merge_task_metadata_fills_missing_fields():
    merged = inference.merge_task_metadata("easy_mnist", {"task_id": "easy_mnist"})

    assert merged["difficulty"] == "easy"
    assert merged["dataset"] == "mnist"
    assert merged["max_epochs"] == 100
