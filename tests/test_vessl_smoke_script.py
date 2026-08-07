"""Local static and CLI-help checks for the GPU-only VESSL smoke script."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = PROJECT_ROOT / "scripts" / "vessl_smoke.py"


def test_smoke_script_has_a_safe_help_entry_point_without_requiring_cuda() -> None:
    """CLI help validates import and argument wiring without allocating the production model."""
    result = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--output-dir" in result.stdout
    assert "/output" in result.stdout


def test_smoke_script_contains_one_backward_pass_without_optimizer_updates() -> None:
    """The smoke workload checks gradients but deliberately contains no optimizer or train loop."""
    source = SMOKE_SCRIPT.read_text()

    assert "torch.cuda.is_available()" in source
    assert "GPT2Model(GPT2Config())" in source
    assert "loss.backward()" in source
    assert "torch.optim" not in source
    assert "optimizer.step" not in source
    assert "smoke_report.json" in source
    assert "vessl.log" in source
