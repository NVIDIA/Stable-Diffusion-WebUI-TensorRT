import sys
from unittest.mock import MagicMock, patch

import install


def test_uninstall_commands_use_sys_executable():
    """pip uninstall invocations must target the current interpreter."""
    with patch("install.launch") as mock_launch:
        mock_launch.is_installed.return_value = True
        mock_launch.run = MagicMock()

        with patch("install.version", return_value="8.9.4.25"):
            install.install()

    uninstall_calls = [
        call_args[0][0]
        for call_args in mock_launch.run.call_args_list
        if len(call_args[0]) > 0
        and isinstance(call_args[0][0], list)
        and len(call_args[0][0]) > 3
        and call_args[0][0][1] == "-m"
        and call_args[0][0][2] == "pip"
        and "uninstall" in call_args[0][0]
    ]

    assert len(uninstall_calls) >= 2, "expected at least two uninstall calls"

    for cmd in uninstall_calls:
        assert cmd[0] == sys.executable, (
            f"uninstall command should use sys.executable ({sys.executable}), "
