"""
Tests for the calibration dataset and classifier validation.
"""

from press.calibration.calibration_data import validate_classifier_accuracy


def test_classifier_calibration():
    """Verify the linguistic confidence classifier meets minimum accuracy."""
    result = validate_classifier_accuracy()

    # Mean absolute error should be reasonable (< 0.3)
    assert result["mean_absolute_error"] < 0.35, (
        f"MAE too high: {result['mean_absolute_error']}"
    )

    # Bin accuracy should be reasonable (> 60%)
    assert result["bin_accuracy"] >= 0.5, (
        f"Bin accuracy too low: {result['bin_accuracy']}"
    )

    print(f"Classifier MAE: {result['mean_absolute_error']}")
    print(f"Classifier Bin Accuracy: {result['bin_accuracy']}")
