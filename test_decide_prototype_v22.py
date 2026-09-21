from decide_prototype_v22 import build_decision_from_payloads


def valid_payloads():
    mechanism = (
        {
            "long_run_ready": True,
            "mean_training_batch_fraction": 1.0,
            "mechanism_gates": {"positive": True, "background": True},
        }
    )
    paired = (
        {
            "attribution_valid": True,
            "paired_optimization_ready": True,
            "optimization_gates": {"classification": True, "regression": True},
        }
    )
    offline = (
        {
            "screening_pass": True,
            "configured_feature_space_comparison": {
                "valid": True,
                "raw": {
                    "balanced_accuracy": 0.70,
                    "foreground_recall": 0.72,
                    "hard_background_recall": 0.68,
                },
                "projected": {
                    "balanced_accuracy": 0.73,
                    "foreground_recall": 0.74,
                    "hard_background_recall": 0.70,
                },
                "projected_minus_raw": {"balanced_accuracy": 0.03},
            },
        }
    )
    return mechanism, paired, offline


def test_rapid_pass_only_authorizes_faithful_validation():
    mechanism, paired, offline = valid_payloads()
    decision = build_decision_from_payloads(
        mechanism, paired, offline, mode="rapid"
    )

    assert decision["all_gates_pass"]
    assert decision["next_action"] == "run_faithful_paired_validation"
    assert not decision["ready_for_long_run"]


def test_faithful_pass_authorizes_long_run():
    mechanism, paired, offline = valid_payloads()
    decision = build_decision_from_payloads(
        mechanism, paired, offline, mode="faithful"
    )

    assert decision["all_gates_pass"]
    assert decision["ready_for_long_run"]
    assert decision["next_action"] == "run_long_training"


def test_projected_space_must_not_hide_foreground_failure():
    mechanism, paired, offline = valid_payloads()
    offline["configured_feature_space_comparison"]["projected"][
        "foreground_recall"
    ] = 0.55
    decision = build_decision_from_payloads(
        mechanism, paired, offline, mode="faithful"
    )

    assert not decision["all_gates_pass"]
    assert "projected_foreground_recall_at_least_0p60" in decision["failed_gates"]
    assert decision["next_action"] == "stop_and_fix_failed_mechanism"


def test_faithful_mode_rejects_truncated_epoch_updates():
    mechanism, paired, offline = valid_payloads()
    mechanism["mean_training_batch_fraction"] = 0.05
    decision = build_decision_from_payloads(
        mechanism, paired, offline, mode="faithful"
    )

    assert not decision["all_gates_pass"]
    assert "faithful_updates_use_full_epoch" in decision["failed_gates"]


def main():
    test_rapid_pass_only_authorizes_faithful_validation()
    test_faithful_pass_authorizes_long_run()
    test_projected_space_must_not_hide_foreground_failure()
    test_faithful_mode_rejects_truncated_epoch_updates()
    print("Prototype-v2.2 decision tests passed")


if __name__ == "__main__":
    main()
