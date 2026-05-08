# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subprocess entry point for running isolated benchmark iterations.

This module provides the entry point for running a single benchmark in a subprocess.
It's used by MultiRunOrchestrator to execute each run in complete isolation,
allowing the SystemController to call os._exit() without affecting the orchestrator.
"""

import os
import sys
from pathlib import Path

import orjson

from aiperf.common.constants import IS_WINDOWS


def _release_inherited_pipes_on_windows() -> None:
    """Redirect stdin/stdout/stderr to NUL on Windows.

    See bootstrap.py::_redirect_stdio_to_devnull for the full rationale.
    Sweep iterations are spawned via subprocess.run with stdout=sys.stdout
    inherited from the orchestrator master, which on Windows propagates
    pytest's subprocess.PIPE all the way down. The iteration's own grandchild
    workers already redirect via the bootstrap fix, but the iteration process
    itself still holds the inherited pipe handle, so its os._exit() can hang
    or segfault during DLL_PROCESS_DETACH. Releasing the inherited FDs to NUL
    here unblocks shutdown for that intermediate process.
    """
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull_fd, fd)
    os.close(devnull_fd)
    sys.stdin = os.fdopen(0, "r", closefd=False)
    sys.stdout = os.fdopen(1, "w", closefd=False)
    sys.stderr = os.fdopen(2, "w", closefd=False)


def main() -> None:
    """Run a single benchmark from a JSON config file.

    This function is the entry point for subprocess execution. It:
    1. Loads serialized config from a JSON file (path passed as argv[1])
    2. Deserializes into UserConfig and ServiceConfig using Pydantic
    3. Calls _run_single_benchmark() which runs the SystemController
    4. SystemController calls os._exit() at the end, terminating this subprocess

    Usage:
        python -m aiperf.orchestrator.subprocess_runner /path/to/config.json

    Exit codes:
        0: Benchmark completed successfully
        1: Benchmark failed (errors occurred)
        Other: Unexpected error or signal
    """
    if IS_WINDOWS:
        _release_inherited_pipes_on_windows()

    if len(sys.argv) != 2:
        print(
            "Usage: python -m aiperf.orchestrator.subprocess_runner <config.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    config_file = Path(sys.argv[1])

    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    # Import here to avoid loading heavy modules at import time
    # This keeps the subprocess startup fast
    from aiperf.cli_runner import _run_single_benchmark
    from aiperf.common.config import ServiceConfig, UserConfig

    try:
        # Load config from JSON file
        with open(config_file, "rb") as f:
            config_data = orjson.loads(f.read())

        # Deserialize using Pydantic validation
        user_config = UserConfig.model_validate(config_data["user_config"])
        service_config = ServiceConfig.model_validate(config_data["service_config"])

        # Run the benchmark
        # Note: _run_single_benchmark() will call SystemController which calls os._exit()
        # at the end, so this function will never return normally
        _run_single_benchmark(user_config, service_config)

    except KeyError as e:
        print(f"Error: Missing required config key: {e}", file=sys.stderr)
        sys.exit(1)
    except orjson.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to run benchmark: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
