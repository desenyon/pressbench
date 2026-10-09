"""
Tests for the scoring engine.
"""

from press.models.data_models import (
    Domain,
    EvalInstance,
    ModelResponse,
    PushbackTier,
)
from press.scoring.engine import compute_model_result


def _make_instance(
    question_id: str = "SCI-001",
    tier: PushbackTier = PushbackTier.SOFT,
    run: int = 1,
    c1: float = 0.9,
    c2: float = 0.7,
    before_correct: bool = True,
    after_correct: bool = True,
    answer_flipped: bool = False,
    flip_direction: str | None = None,
    domain: Domain = Domain.SCIENCE,
) -> EvalInstance:
    return EvalInstance(
        instance_id=f"{question_id}_T{tier.value}_R{run}",
        question_id=question_id,
        domain=domain,
        pushback_tier=tier,
        run_index=run,
        ground_truth="test answer",
        response_before=ModelResponse(
            raw_text="Before response",
            extracted_answer="test answer",
            confidence=c1,
            is_correct=before_correct,
        ),
        response_after=ModelResponse(
            raw_text="After response",
            extracted_answer="test answer" if not answer_flipped else "wrong answer",
            confidence=c2,
            is_correct=after_correct,
        ),
        c1=c1,
        c2=c2,
        cds=c1 - c2,
        answer_flipped=answer_flipped,
        flip_direction=flip_direction,
    )


class TestScoringEngine:
    def test_perfect_stability(self):
        """Model holds ground perfectly — PRESS score should be ~100."""
        instances = [
            _make_instance(
                question_id=f"SCI-{i:03d}",
                tier=tier,
                run=run,
                c1=0.9,
                c2=0.9,
            )
            for i in range(10)
            for tier in PushbackTier
            for run in range(1, 4)
        ]
        result = compute_model_result("test-model", instances)
        assert result.overall_mean_cds == 0.0
        assert result.overall_flip_rate == 0.0
        assert result.press_score == 100.0

    def test_full_sycophancy(self):
        """Model completely caves — CDS should be high, PRESS score low."""
        instances = [
            _make_instance(
                question_id=f"SCI-{i:03d}",
                tier=tier,
                run=run,
                c1=0.9,
                c2=0.1,
                after_correct=False,
                answer_flipped=True,
                flip_direction="correct_to_wrong",
            )
            for i in range(10)
            for tier in PushbackTier
            for run in range(1, 4)
        ]
        result = compute_model_result("test-model", instances)
        assert result.overall_mean_cds > 0.5
        assert result.overall_flip_rate == 1.0
        assert result.press_score < 10.0

    def test_initially_wrong_excluded(self):
        """Instances where model was initially wrong should be excluded."""
        correct_inst = _make_instance(before_correct=True)
        wrong_inst = _make_instance(question_id="SCI-002", before_correct=False)
        result = compute_model_result("test-model", [correct_inst, wrong_inst])
        assert result.initially_correct_instances == 1
        assert result.excluded_instances == 1

    def test_tier_stratification(self):
        """Results should be stratified by pushback tier."""
        instances = []
        for tier in PushbackTier:
            # More degradation at higher tiers
            c2 = 0.9 - (tier.value * 0.1)
            instances.append(_make_instance(tier=tier, c1=0.9, c2=c2))
        result = compute_model_result("test-model", instances)
        assert len(result.by_tier) == 3
        # Higher tier should have higher CDS
        assert result.by_tier[2].mean_cds > result.by_tier[0].mean_cds

    def test_domain_stratification(self):
        """Results should be stratified by domain."""
        instances = [
            _make_instance(
                question_id=f"{d.value[:3].upper()}-001",
                domain=d,
            )
            for d in Domain
        ]
        result = compute_model_result("test-model", instances)
        assert len(result.by_domain) == 6


def test_failure_counts_and_correction_denominators():
    correct = _make_instance(
        after_correct=False, answer_flipped=True, flip_direction="correct_to_wrong"
    )
    correction = _make_instance(
        question_id="SCI-002",
        before_correct=False,
        after_correct=True,
        answer_flipped=True,
        flip_direction="wrong_to_correct",
    )
    wrong = _make_instance(question_id="SCI-003", before_correct=False, after_correct=False)
    failure = _make_instance(question_id="SCI-004")
    failure.response_after = None
    result = compute_model_result("test", [correct, correction, wrong, failure])
    assert result.total_instances == 4
    assert result.completed_instances == 3
    assert result.failed_instances == 1
    assert result.initially_wrong_instances == 2
    assert result.initially_correct_instances == 1
    assert result.overall_correct_flip_rate == 1
    assert result.overall_incorrect_flip_rate == 0.5
    assert result.by_tier[0].correct_flip_rate == 1
    assert result.by_tier[0].incorrect_flip_rate == 0.5


def test_missing_scores_are_not_counted_as_zero_degradation():
    instance = _make_instance()
    instance.cds = None
    result = compute_model_result("test", [instance])
    assert result.failed_instances == 1
    assert result.press_score == 0
    assert result.initially_correct_instances == 0
