from press.models.data_models import ModelResult
from press.reporting.html_report import generate_html_report


def test_report_embeds_charts_escapes_names_and_uses_metadata(tmp_path):
    charts = tmp_path / "charts"
    charts.mkdir()
    (charts / "press_scores.png").write_bytes(b"fake image")
    result = ModelResult(
        model_id="test",
        model_name="<script>alert(1)</script>",
        total_instances=6,
        completed_instances=4,
        failed_instances=2,
        run_metadata={
            "question_count": 1,
            "settings": {"runs_per_instance": 2, "temperature": 0.5},
        },
    )
    html = generate_html_report([result], tmp_path).read_text()
    assert "data:image/png;base64" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "4/6 completed; 2 failed" in html
    assert "1 questions · 2 repeats · temperature 0.5" in html


def test_legacy_report_does_not_invent_run_configuration(tmp_path):
    result = ModelResult(model_id="legacy", model_name="Legacy")
    html = generate_html_report([result], tmp_path).read_text()
    assert "Legacy: run configuration unknown" in html
