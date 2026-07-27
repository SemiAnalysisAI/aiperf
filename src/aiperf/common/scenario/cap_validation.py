# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.common.scenario.base import ScenarioSpec, ScenarioViolation

if TYPE_CHECKING:
    from aiperf.common.aiperf_logger import AIPerfLogger


def validate_delay_caps(
    user_config: Any,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    logger: AIPerfLogger,
) -> None:
    """Apply scenario locks for per-turn, per-trace, and global idle caps."""
    _validate_inter_turn_cap(user_config, spec, violations, logger)
    _validate_trace_idle_cap(user_config, spec, violations, logger)
    _validate_system_idle_cap(user_config, spec, violations, logger)


def _validate_inter_turn_cap(
    user_config: Any,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    logger: AIPerfLogger,
) -> None:
    if spec.inter_turn_delay_cap_seconds is not None:
        explicit = getattr(
            user_config.loadgen, "_inter_turn_delay_cap_explicitly_set", False
        )
        value = user_config.loadgen.inter_turn_delay_cap_seconds
        if explicit and value != spec.inter_turn_delay_cap_seconds:
            violations.append(
                ScenarioViolation(
                    flag="--inter-turn-delay-cap-seconds",
                    current_value=value,
                    required_value=spec.inter_turn_delay_cap_seconds,
                    message=f"scenario {spec.name!r} locks the cap to {spec.inter_turn_delay_cap_seconds}",
                )
            )
        elif not explicit and value is None:
            user_config.loadgen.inter_turn_delay_cap_seconds = (
                spec.inter_turn_delay_cap_seconds
            )
            logger.info(
                "Scenario %r: auto-set --inter-turn-delay-cap-seconds=%s (was unset).",
                spec.name,
                spec.inter_turn_delay_cap_seconds,
            )

    if (
        spec.forbid_inter_turn_delay_cap
        and user_config.loadgen.inter_turn_delay_cap_seconds is not None
    ):
        violations.append(
            ScenarioViolation(
                flag="--inter-turn-delay-cap-seconds",
                current_value=user_config.loadgen.inter_turn_delay_cap_seconds,
                required_value=None,
                message=f"scenario {spec.name!r} preserves original inter-turn timing",
            )
        )


def _validate_trace_idle_cap(
    user_config: Any,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    logger: AIPerfLogger,
) -> None:
    if spec.trace_idle_gap_cap_seconds is not None:
        explicit = getattr(
            user_config.loadgen, "_trace_idle_gap_cap_explicitly_set", False
        )
        value = user_config.loadgen.trace_idle_gap_cap_seconds
        if explicit and value != spec.trace_idle_gap_cap_seconds:
            violations.append(
                ScenarioViolation(
                    flag="--trace-idle-gap-cap-seconds",
                    current_value=value,
                    required_value=spec.trace_idle_gap_cap_seconds,
                    message=f"scenario {spec.name!r} locks the per-trace idle-gap cap to {spec.trace_idle_gap_cap_seconds}",
                )
            )
        elif not explicit and value is None:
            user_config.loadgen.trace_idle_gap_cap_seconds = (
                spec.trace_idle_gap_cap_seconds
            )
            logger.info(
                "Scenario %r: auto-set --trace-idle-gap-cap-seconds=%s (was unset).",
                spec.name,
                spec.trace_idle_gap_cap_seconds,
            )

    if (
        spec.forbid_trace_idle_gap_cap
        and user_config.loadgen.trace_idle_gap_cap_seconds is not None
    ):
        violations.append(
            ScenarioViolation(
                flag="--trace-idle-gap-cap-seconds",
                current_value=user_config.loadgen.trace_idle_gap_cap_seconds,
                required_value=None,
                message=f"scenario {spec.name!r} preserves original per-trace timing",
            )
        )


def _validate_system_idle_cap(
    user_config: Any,
    spec: ScenarioSpec,
    violations: list[ScenarioViolation],
    logger: AIPerfLogger,
) -> None:
    if spec.system_idle_gap_cap_seconds is None:
        return
    explicit = getattr(
        user_config.loadgen, "_system_idle_gap_cap_explicitly_set", False
    )
    value = user_config.loadgen.system_idle_gap_cap_seconds
    if explicit and value != spec.system_idle_gap_cap_seconds:
        violations.append(
            ScenarioViolation(
                flag="--system-idle-gap-cap-seconds",
                current_value=value,
                required_value=spec.system_idle_gap_cap_seconds,
                message=f"scenario {spec.name!r} locks the global system-idle cap to {spec.system_idle_gap_cap_seconds}",
            )
        )
    elif not explicit and value is None:
        user_config.loadgen.system_idle_gap_cap_seconds = (
            spec.system_idle_gap_cap_seconds
        )
        logger.info(
            "Scenario %r: auto-set --system-idle-gap-cap-seconds=%s (was unset).",
            spec.name,
            spec.system_idle_gap_cap_seconds,
        )
