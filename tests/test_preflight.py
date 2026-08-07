"""Tests for local-only VESSL run scaffolding and environment preflight."""

import json
from pathlib import Path

import yaml

from gpt2_124m.preflight import (
    GPT2_SMALL_TRAINABLE_PARAMETER_COUNT,
    collect_local_preflight_report,
    main,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VESSL_TEMPLATE_PATHS = (
    PROJECT_ROOT / "configs" / "vessl" / "preflight.template.yaml",
    PROJECT_ROOT / "configs" / "vessl" / "smoke-train.template.yaml",
    PROJECT_ROOT / "configs" / "vessl" / "main-pretrain.template.yaml",
)


def test_local_preflight_runs_without_vessl_and_reports_gpt2_small_count() -> None:
    """The local utility uses only project and PyTorch imports, never the VESSL SDK."""
    report = collect_local_preflight_report()

    assert report.package_import_status == {"gpt2_124m": True, "torch": True}
    assert report.selected_device in {"cpu", "cuda"}
    assert report.gpt2_small_trainable_parameter_count == GPT2_SMALL_TRAINABLE_PARAMETER_COUNT
    assert report.gpt2_small_trainable_parameter_count == 124_439_808


def test_local_preflight_main_prints_serializable_json(capsys: object) -> None:
    """The terminal entry point produces a compact report ready for local inspection."""
    main()

    output = capsys.readouterr().out  # type: ignore[attr-defined]
    parsed = json.loads(output)
    assert parsed["gpt2_small_trainable_parameter_count"] == 124_439_808


def test_vessl_templates_contain_required_run_yaml_sections_and_placeholders() -> None:
    """All future run templates declare their required VESSL YAML sections explicitly."""
    for path in VESSL_TEMPLATE_PATHS:
        template = path.read_text()
        assert path.is_file()
        parsed_template = yaml.safe_load(template)
        assert isinstance(parsed_template, dict)
        for section in ("name:", "resources:", "image:", "import:", "export:", "run:"):
            assert section in template
        for placeholder in (
            "REPLACE_WITH_VESSL_CLUSTER",
            "REPLACE_WITH_GPU_PRESET",
            "REPLACE_WITH_PYTORCH_IMAGE",
            "REPLACE_WITH_VESSL_GIT_CREDENTIAL_NAME",
        ):
            assert placeholder in template


def test_private_repository_templates_use_credential_placeholders_not_embedded_tokens() -> None:
    """Git imports use VESSL credentials by name and never put a secret in the repository URL."""
    prohibited_secret_markers = ("ghp_", "github_pat_", "x-access-token:", "vessl_api_key")
    for path in VESSL_TEMPLATE_PATHS:
        template = path.read_text().lower()
        assert "credential_name: replace_with_vessl_git_credential_name" in template
        assert "https://github.com/uzbekswe/gpt2-124m-from-scratch" in template
        assert all(marker not in template for marker in prohibited_secret_markers)
        assert "organization:" not in template
        assert "project:" not in template
