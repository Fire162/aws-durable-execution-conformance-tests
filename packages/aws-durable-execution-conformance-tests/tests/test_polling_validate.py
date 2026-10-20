# SPDX-FileCopyrightText: 2026-present Amazon.com, Inc. or its affiliates.
#
# SPDX-License-Identifier: Apache-2.0
"""Exercise callback scheduling through the polling validator and sender."""

from __future__ import annotations

import json
from typing import Any

import pytest
from botocore.exceptions import ClientError

import aws_durable_execution_conformance_tests.validate as validation
from aws_durable_execution_conformance_tests.callback import CallbackAction


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _CallbackService:
    """Model service deadlines while a history request spends time retrying."""

    def __init__(
        self,
        clock: _Clock,
        *,
        history_retry_delay: float = 0,
        names: tuple[str, ...] = ("callback",),
        reject_heartbeat: bool = False,
        heartbeat_timeouts: dict[str, float] | None = None,
    ) -> None:
        self.clock = clock
        self.history_retry_delay = history_retry_delay
        self.reject_heartbeat = reject_heartbeat
        self.history_reads = 0
        self.attempts: list[tuple[str, str, float, Any]] = []
        self.heartbeat_timeouts = heartbeat_timeouts or dict.fromkeys(names, 10.0)
        self.deadlines = self.heartbeat_timeouts.copy()
        self.outcomes: dict[str, str] = {}

    def _expire(self) -> None:
        for name, deadline in self.deadlines.items():
            if name not in self.outcomes and self.clock.now >= deadline:
                self.outcomes[name] = "CallbackTimedOut"

    def _status(self) -> str:
        self._expire()
        if len(self.outcomes) < len(self.deadlines):
            return "RUNNING"
        if all(outcome == "CallbackSucceeded" for outcome in self.outcomes.values()):
            return "SUCCEEDED"
        return "FAILED"

    def get_durable_execution_history(self, **_kwargs: Any) -> dict[str, Any]:
        self.history_reads += 1
        retries = 0
        if self.history_reads == 2:
            self.clock.sleep(self.history_retry_delay)
            retries = 3 if self.history_retry_delay else 0
        status = self._status()
        events: list[dict[str, Any]] = [
            {
                "EventType": "CallbackStarted",
                "Name": name,
                "CallbackStartedDetails": {"CallbackId": name, "HeartbeatTimeout": self.heartbeat_timeouts[name]},
            }
            for name in self.deadlines
        ]
        events.extend({"EventType": outcome, "Name": name} for name, outcome in self.outcomes.items())
        return {
            "Events": events,
            "ExecutionStatus": status,
            "ResponseMetadata": {"RetryAttempts": retries},
        }

    def get_durable_execution(self, **_kwargs: Any) -> dict[str, Any]:
        return {"Status": self._status()}

    def _send(self, operation: str, callback_id: str, payload: Any = None) -> dict[str, Any]:
        self.attempts.append((operation, callback_id, self.clock.now, payload))
        self._expire()
        if callback_id in self.outcomes or (operation == "heartbeat" and self.reject_heartbeat):
            raise ClientError(
                {"Error": {"Code": "InvalidParameterValueException", "Message": "callback unavailable"}},
                "SendDurableExecutionCallback" + operation.title(),
            )
        if operation == "heartbeat":
            self.deadlines[callback_id] = self.clock.now + self.heartbeat_timeouts[callback_id]
        else:
            self.outcomes[callback_id] = "CallbackSucceeded" if operation == "success" else "CallbackFailed"
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def send_durable_execution_callback_heartbeat(self, *, CallbackId: str) -> dict[str, Any]:
        return self._send("heartbeat", CallbackId)

    def send_durable_execution_callback_success(self, *, CallbackId: str, Result: bytes) -> dict[str, Any]:
        return self._send("success", CallbackId, json.loads(Result))

    def send_durable_execution_callback_failure(self, *, CallbackId: str, Error: dict[str, Any]) -> dict[str, Any]:
        return self._send("failure", CallbackId, Error)


def _validate(
    monkeypatch: pytest.MonkeyPatch,
    service: _CallbackService,
    actions: list[CallbackAction],
    expected_status: str = "SUCCEEDED",
) -> validation.AsyncValidationResult:
    monkeypatch.setattr(validation, "time", service.clock)
    return validation.PollingValidator(service).validate(
        execution_arn="arn:test:execution",
        expected_events=[],
        expected_result={"ExecutionStatus": expected_status},
        callback_actions=actions,
    )


@pytest.mark.parametrize("history_retry_delay", [10, 25])
def test_heartbeat_followup_precedes_slow_history_poll(
    monkeypatch: pytest.MonkeyPatch, history_retry_delay: float
) -> None:
    clock = _Clock()
    service = _CallbackService(clock, history_retry_delay=history_retry_delay)

    result = _validate(
        monkeypatch,
        service,
        [CallbackAction("*", "heartbeat"), CallbackAction("*", "success", "done", delay_seconds=1)],
    )

    assert result.passed, result.errors
    assert result.final_status == "SUCCEEDED"
    assert result.callbacks_sent == 2
    assert service.attempts == [("heartbeat", "callback", 2, None), ("success", "callback", 3, "done")]
    assert service.history_reads == 2
    assert clock.now == 5 + history_retry_delay
    assert service.outcomes == {"callback": "CallbackSucceeded"}


@pytest.mark.parametrize("operation", ["success", "failure"])
def test_heartbeat_chain_keeps_delays_and_stops_after_terminal_action(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    service = _CallbackService(_Clock(), history_retry_delay=10)
    payload: Any = "done" if operation == "success" else {"ErrorType": "ExternalFailure"}

    result = _validate(
        monkeypatch,
        service,
        [
            CallbackAction("*", "heartbeat"),
            CallbackAction("*", "heartbeat", delay_seconds=1),
            CallbackAction("*", operation, payload, delay_seconds=2),
            CallbackAction("*", "success", "must not send"),
        ],
        "SUCCEEDED" if operation == "success" else "FAILED",
    )

    assert result.passed, result.errors
    assert result.callbacks_sent == 3
    assert service.attempts == [
        ("heartbeat", "callback", 2, None),
        ("heartbeat", "callback", 3, None),
        (operation, "callback", 5, payload),
    ]


def test_followup_uses_the_same_callback_and_leaves_other_actions_available(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _CallbackService(_Clock(), names=("first", "second"))

    result = _validate(
        monkeypatch,
        service,
        [
            CallbackAction("first", "heartbeat"),
            CallbackAction("first", "success", "first result", delay_seconds=1),
            CallbackAction("second", "success", "second result"),
        ],
    )

    assert result.passed, result.errors
    assert result.callbacks_sent == 3
    assert service.attempts == [
        ("heartbeat", "first", 2, None),
        ("success", "second", 2, "second result"),
        ("success", "first", 3, "first result"),
    ]


@pytest.mark.parametrize("second_timeout", [5.0, 10.0])
def test_delayed_followup_does_not_starve_another_known_callback(
    monkeypatch: pytest.MonkeyPatch, second_timeout: float
) -> None:
    service = _CallbackService(_Clock(), heartbeat_timeouts={"first": 10, "second": second_timeout})

    result = _validate(
        monkeypatch,
        service,
        [
            CallbackAction("first", "heartbeat"),
            CallbackAction("first", "success", "first result", delay_seconds=4),
            CallbackAction("second", "success", "second result"),
        ],
    )

    assert result.passed, result.errors
    assert result.final_status == "SUCCEEDED"
    assert result.callbacks_sent == 3
    assert service.attempts == [
        ("heartbeat", "first", 2, None),
        ("success", "second", 2, "second result"),
        ("success", "first", 6, "first result"),
    ]
    assert service.outcomes == {"first": "CallbackSucceeded", "second": "CallbackSucceeded"}
    assert service.history_reads == 2


def test_heartbeat_followup_keeps_first_round_wildcard_action_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _CallbackService(_Clock(), names=("first", "second"))

    result = _validate(
        monkeypatch,
        service,
        [
            CallbackAction("*", "heartbeat"),
            CallbackAction("*", "success", "second result"),
            CallbackAction("*", "success", "first result", delay_seconds=1),
        ],
    )

    assert result.passed, result.errors
    assert result.callbacks_sent == 3
    assert service.attempts == [
        ("heartbeat", "first", 2, None),
        ("success", "second", 2, "second result"),
        ("success", "first", 3, "first result"),
    ]


def test_configured_delay_that_exceeds_heartbeat_timeout_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _CallbackService(_Clock())

    result = _validate(
        monkeypatch,
        service,
        [CallbackAction("*", "heartbeat"), CallbackAction("*", "success", "too late", delay_seconds=11)],
    )

    assert not result.passed
    assert result.final_status == "FAILED"
    assert result.callbacks_sent == 1
    assert any("Callback success failed" in error for error in result.errors)
    assert service.attempts == [("heartbeat", "callback", 2, None), ("success", "callback", 13, "too late")]
    assert service.outcomes == {"callback": "CallbackTimedOut"}


def test_failed_heartbeat_does_not_retry_or_dispatch_followup_before_next_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _CallbackService(_Clock(), reject_heartbeat=True)

    result = _validate(
        monkeypatch,
        service,
        [CallbackAction("*", "heartbeat"), CallbackAction("*", "success", "done", delay_seconds=1)],
    )

    assert not result.passed
    assert result.final_status == "SUCCEEDED"
    assert result.callbacks_sent == 1
    assert any("Callback heartbeat failed" in error for error in result.errors)
    assert service.attempts == [("heartbeat", "callback", 2, None), ("success", "callback", 5, "done")]


def test_terminal_history_does_not_dispatch_callback_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _CallbackService(_Clock())
    service.outcomes["callback"] = "CallbackSucceeded"

    result = _validate(monkeypatch, service, [CallbackAction("*", "heartbeat")])

    assert result.passed, result.errors
    assert result.callbacks_sent == 0
    assert service.attempts == []


def test_heartbeat_without_followup_keeps_existing_unmatched_action_error(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _CallbackService(_Clock())

    result = _validate(monkeypatch, service, [CallbackAction("*", "heartbeat")], "FAILED")

    assert not result.passed
    assert any("No matching CallbackAction" in error for error in result.errors)
    assert result.callbacks_sent == 1
    assert service.attempts == [("heartbeat", "callback", 2, None)]
    assert service.outcomes == {"callback": "CallbackTimedOut"}
