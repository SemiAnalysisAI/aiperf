# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import platform
import signal
from collections.abc import Callable, Coroutine

from aiperf.common.mixins import AIPerfLoggerMixin


class SignalHandlerMixin(AIPerfLoggerMixin):
    """Mixin for services that need to handle system signals."""

    def __init__(self, **kwargs) -> None:
        # Set to store signal handler tasks to prevent them from being garbage collected
        self._signal_tasks = set()
        super().__init__(**kwargs)

    def setup_signal_handlers(self, callback: Callable[[int], Coroutine]) -> None:
        """Set up signal handler for the SIGINT signal to trigger graceful shutdown.

        Args:
            callback: The callback to call when a signal is received
        """
        loop = asyncio.get_running_loop()
        self.debug(f"Setting up SIGINT handler on loop {loop}")

        def signal_handler(sig: int) -> None:
            self.warning(f"Signal {sig} received, initiating graceful shutdown")
            task = asyncio.create_task(callback(sig))
            self._signal_tasks.add(task)
            task.add_done_callback(self._signal_tasks.discard)

        # Windows ProactorEventLoop does not implement add_signal_handler.
        # Fall back to signal.signal(), which Windows supports for SIGINT.
        # The handler is dispatched on the main thread and re-enters the loop
        # via run_coroutine_threadsafe to invoke the async callback.
        if platform.system() == "Windows":
            def windows_signal_handler(sig: int, _frame: object) -> None:
                self.warning(f"Signal {sig} received, initiating graceful shutdown")
                asyncio.run_coroutine_threadsafe(callback(sig), loop)

            signal.signal(signal.SIGINT, windows_signal_handler)
        else:
            loop.add_signal_handler(signal.SIGINT, signal_handler, signal.SIGINT)
        self.debug("SIGINT handler installed successfully")
