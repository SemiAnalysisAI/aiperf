# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tests/harness/utils.py — specifically the platform-branching
shlex behavior for Windows command parsing.

On Windows the harness needs two properties from shlex that the standard
modes don't give together:

  1) preserve backslashes in interpolated paths (C:\\Users\\... must survive
     parsing; POSIX shlex.split would treat `\\` as an escape char and strip
     it — Bug 6).
  2) strip surrounding quotes from quoted values (so f-strings like
     `--sequence-distribution "64|10,32|8:70..."` pass the unquoted value
     to aiperf; non-POSIX shlex would keep the literal `"` in the token,
     and aiperf's config validator then rejects the value — AIP-896).

The harness now uses ``shlex.shlex(posix=True, escape="")`` on Windows to
get POSIX-style quote handling without backslash escaping. On non-Windows
plain POSIX ``shlex.split`` is used.
"""

from __future__ import annotations

from unittest.mock import patch

from tests.harness.utils import AIPerfCLI


class TestParseCommandShlexMode:
    """Verify _parse_command picks POSIX vs non-POSIX shlex mode by platform."""

    def test_unix_uses_posix_mode_normal_path(self) -> None:
        """On non-Windows the parser runs in POSIX mode (default shlex behavior)."""
        with patch("tests.harness.utils.sys.platform", "linux"):
            args = AIPerfCLI._parse_command(
                "aiperf profile --file /tmp/data.jsonl --request-count 5"
            )
        assert args == ["profile", "--file", "/tmp/data.jsonl", "--request-count", "5"]

    def test_windows_preserves_backslashes_in_paths(self) -> None:
        """On Windows backslashes in interpolated paths (C:\\Users\\...) are
        preserved as literal chars rather than treated as escape introducers."""
        cmd = r"aiperf profile --file C:\Users\test\data.jsonl --request-count 5"
        with patch("tests.harness.utils.sys.platform", "win32"):
            args = AIPerfCLI._parse_command(cmd)
        assert args == [
            "profile",
            "--file",
            r"C:\Users\test\data.jsonl",
            "--request-count",
            "5",
        ]

    def test_windows_strips_double_quotes_around_value(self) -> None:
        """On Windows, surrounding double quotes are stripped from quoted
        values. Regression for AIP-896 part 1: with non-POSIX shlex the
        quote chars stayed in the token, and aiperf's config validator
        then rejected the (quoted) value.

        Example from test_seq_dist.py: --sequence-distribution
        "64|10,32|8:70;..." must arrive at aiperf as the unquoted string."""
        cmd = (
            "aiperf profile "
            '--sequence-distribution "64|10,32|8:70;256|40,128|20:20" '
            "--request-count 1"
        )
        with patch("tests.harness.utils.sys.platform", "win32"):
            args = AIPerfCLI._parse_command(cmd)
        assert args == [
            "profile",
            "--sequence-distribution",
            "64|10,32|8:70;256|40,128|20:20",
            "--request-count",
            "1",
        ]

    def test_windows_strips_single_quotes_around_value(self) -> None:
        """Single-quoted values also have their surrounding quotes stripped
        on Windows, matching POSIX shlex.split behavior."""
        cmd = "aiperf profile --extra-inputs 'foo:bar,baz:qux' --request-count 1"
        with patch("tests.harness.utils.sys.platform", "win32"):
            args = AIPerfCLI._parse_command(cmd)
        assert args == [
            "profile",
            "--extra-inputs",
            "foo:bar,baz:qux",
            "--request-count",
            "1",
        ]

    def test_windows_handles_quoted_value_with_backslash_path(self) -> None:
        """Both fixes together: a Windows path AND a quoted value in the same
        command must each parse correctly — backslashes preserved, quotes
        stripped."""
        cmd = (
            r"aiperf profile --file C:\Users\test\data.jsonl "
            '--sequence-distribution "64|10,32|8:70" --request-count 1'
        )
        with patch("tests.harness.utils.sys.platform", "win32"):
            args = AIPerfCLI._parse_command(cmd)
        assert args == [
            "profile",
            "--file",
            r"C:\Users\test\data.jsonl",
            "--sequence-distribution",
            "64|10,32|8:70",
            "--request-count",
            "1",
        ]

    def test_unix_posix_mode_strips_backslashes_from_windows_style_paths(self) -> None:
        """Confirms the bug: with POSIX shlex (the pre-fix Linux/macOS code
        path), backslashes are eaten as escape characters. This is why
        Windows-runtime tests need the platform branch — the fix is not
        just cosmetic."""
        cmd = r"aiperf profile --file C:\Users\test\data.jsonl"
        with patch("tests.harness.utils.sys.platform", "linux"):
            args = AIPerfCLI._parse_command(cmd)
        # POSIX shlex consumed every backslash as an escape introducer
        assert "C:Userstestdata.jsonl" in args
        assert r"C:\Users\test\data.jsonl" not in args

    def test_drops_leading_aiperf_token_on_both_platforms(self) -> None:
        """Sanity: the post-shlex slicing logic (drop leading 'aiperf') runs
        identically regardless of shlex mode."""
        for plat in ("linux", "win32"):
            with patch("tests.harness.utils.sys.platform", plat):
                args = AIPerfCLI._parse_command("aiperf profile --request-count 1")
            assert args[0] == "profile", f"first arg wrong on {plat}: {args}"

    def test_handles_continuation_backslashes_on_both_platforms(self) -> None:
        """Backslash-newline continuations are normalized BEFORE shlex runs
        (cmd.replace("\\\\\\n", " ")), so they work regardless of platform."""
        cmd = "aiperf profile \\\n  --request-count 1"
        for plat in ("linux", "win32"):
            with patch("tests.harness.utils.sys.platform", plat):
                args = AIPerfCLI._parse_command(cmd)
            assert args == ["profile", "--request-count", "1"], (
                f"continuation handling wrong on {plat}: {args}"
            )
