import pytest

from fpdiag.cli import parser


@pytest.mark.parametrize("command", ["preflight", "verify", "weight-delta", "diagnose", "intervene", "report"])
def test_commands_are_exposed(command):
    args = parser().parse_args([command, "--config", "x.yaml"])
    assert args.command == command
