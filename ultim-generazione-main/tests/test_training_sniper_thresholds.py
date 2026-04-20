from config.settings import (
    ORARI_MIGLIORI_UTC,
    ORARI_VIETATI_UTC,
    SNIPER_FUSION_THRESHOLD,
    SNIPER_NON_OPTIMAL_HOUR_PENALTY,
    THRESHOLD_BASE,
    TRAINING_FUSION_THRESHOLD,
    TRAINING_MIN_FUSION_SCORE,
)


def test_sniper_fusion_threshold_value():
    assert SNIPER_FUSION_THRESHOLD == 0.28


def test_sniper_non_optimal_hour_penalty_is_disabled():
    assert SNIPER_NON_OPTIMAL_HOUR_PENALTY == 0.0


def test_base_threshold_value():
    assert THRESHOLD_BASE == 0.28


def test_time_filters_allow_all_hours():
    assert ORARI_VIETATI_UTC == []
    assert ORARI_MIGLIORI_UTC == list(range(0, 24))


def test_training_threshold_values():
    assert TRAINING_FUSION_THRESHOLD == 0.28
    assert TRAINING_MIN_FUSION_SCORE == 0.20
