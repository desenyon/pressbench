"""Public CLI contracts, without credentials or paid requests."""

import json

import pytest
from click.testing import CliRunner

from press.cli import main
from press.config import Settings


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch, tmp_path):
    settings = Settings(
        _env_file=None,
        openai_api_key="",
        anthropic_api_key="",
        google_api_key="",
        together_api_key="",
        output_dir=tmp_path / "out",
    )
    monkeypatch.setattr("press.config.get_settings", lambda: settings)
    monkeypatch.setattr("press.cli.load_dotenv", lambda: None)
    return settings


def test_dry_run_has_no_network_or_output(clean_settings):
    result = CliRunner().invoke(main, ["run", "--model", "gpt-test", "--limit", "2", "--dry-run"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["questions"] == 2
    assert plan["instances_per_model"] == 18
    assert plan["max_requests_before_retries"] == 36
    assert not clean_settings.output_dir.exists()


def test_cli_respects_settings_defaults(clean_settings):
    clean_settings.runs_per_instance = 7
    clean_settings.concurrency = 3
    result = CliRunner().invoke(main, ["run", "--limit", "1", "--dry-run"])
    plan = json.loads(result.output)
    assert plan["runs_per_instance"] == 7
    assert plan["concurrency"] == 3


@pytest.mark.parametrize(
    "args",
    [
        ["--runs", "0"],
        ["--runs", "11"],
        ["--concurrency", "0"],
        ["--temperature", "3"],
        ["--retry-failed"],
        ["--discover", "--dry-run"],
        ["--timeout", "0"],
    ],
)
def test_invalid_options_fail_early(args):
    result = CliRunner().invoke(main, ["run", *args])
    assert result.exit_code != 0


@pytest.mark.parametrize("model", ["gpt-test", "o1", "o3", "o4-mini"])
def test_openai_credentials_use_shared_routing(model):
    result = CliRunner().invoke(main, ["run", "--model", model, "--limit", "1"])
    assert result.exit_code != 0
    assert "Missing OPENAI_API_KEY" in result.output


def test_dataset_validate_is_automation_friendly():
    result = CliRunner().invoke(main, ["dataset", "validate"])
    assert result.exit_code == 0, result.output


def test_invalid_dataset_validation_fails(monkeypatch):
    monkeypatch.setattr("press.dataset.loader.validate_dataset", lambda _: ["broken"])
    result = CliRunner().invoke(main, ["dataset", "validate"])
    assert result.exit_code != 0


def test_full_cli_run_partial_failure_resume_and_reports(monkeypatch, clean_settings):
    from press.dataset.loader import load_dataset
    from press.models.clients import LLMResponse, ModelClient

    answer = next(q.answer for q in load_dataset().questions if q.id == "SCI-001")

    class ScriptedClient(ModelClient):
        fail = True
        calls = []

        async def generate(self, messages, **kwargs):
            self.calls.append(messages)
            if len(messages) == 4 and self.fail:
                raise ConnectionError("simulated outage")
            return LLMResponse(text=answer, model="gpt-test")

    client = ScriptedClient("gpt-test")
    clean_settings.openai_api_key = "fake"
    monkeypatch.setattr("press.evaluation.pipeline.get_client", lambda *args: client)
    args = ["run", "--model", "gpt-test", "--domain", "science", "--limit", "1", "--runs", "1"]
    runner = CliRunner()
    partial = runner.invoke(main, args)
    assert partial.exit_code == 1, partial.output
    assert "partial artifacts saved" in partial.output
    output = clean_settings.output_dir
    assert (output / "press_report.html").exists()
    assert len(list((output / "charts").glob("*.png"))) == 3
    assert len(client.calls) == 6
    client.fail = False
    resumed = runner.invoke(main, [*args, "--resume", "--retry-failed"])
    assert resumed.exit_code == 0, resumed.output
    assert len(client.calls) == 9
    result = json.loads(next(output.glob("*_result.json")).read_text())
    assert result["completed_instances"] == 3
    assert result["failed_instances"] == 0
    assert result["press_score"] == 100
    assert result["run_metadata"]["question_count"] == 1
    assert "data:image/png;base64" in (output / "press_report.html").read_text()
    assert runner.invoke(main, ["report", str(output)]).exit_code == 0
    assert runner.invoke(main, ["leaderboard", str(output)]).exit_code == 0


def test_model_discovery_json_without_keys_is_parseable():
    result = CliRunner().invoke(main, ["models", "list", "--json-out"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {}
    assert "no API key" in result.stderr
