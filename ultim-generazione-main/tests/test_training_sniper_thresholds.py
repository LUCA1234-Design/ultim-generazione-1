from config.settings import SNIPER_FUSION_THRESHOLD, SNIPER_NON_OPTIMAL_HOUR_PENALTY


def test_sniper_fusion_threshold_value():
    assert SNIPER_FUSION_THRESHOLD == 0.28


def test_sniper_non_optimal_hour_penalty_is_disabled():
    assert SNIPER_NON_OPTIMAL_HOUR_PENALTY == 0.0
