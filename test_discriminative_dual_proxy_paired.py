from decide_discriminative_dual_proxy_paired import evaluate_paired


def passing_payload():
    return {
        "control": {"f1": 0.6100, "fp_background_far": 170000},
        "prototype": {"f1": 0.6110, "fp_background_far": 169000},
        "source_control": {"f1": 0.7560},
        "source_prototype": {"f1": 0.7530},
        "mechanism": {
            "foreground_similarity_gap": 0.07,
            "background_similarity_gap": 0.06,
            "foreground_effective_prototypes": 3.4,
            "background_effective_prototypes": 3.3,
            "foreground_pairwise_similarity_max": 0.84,
            "background_pairwise_similarity_max": 0.85,
            "foreground_assignment_share_min": 0.08,
            "background_assignment_share_min": 0.07,
            "gradient_all_cosine": 0.01,
        },
    }


def test_passing_paired_result_advances_only_to_confirmation():
    decision = evaluate_paired(passing_payload())
    assert decision["gate_pass"]
    assert decision["next_action"] == "consider_fifty_epoch_confirmation"


def test_each_detector_or_geometry_regression_blocks_advancement():
    mutations = (
        ("prototype", "f1", 0.6099),
        ("prototype", "fp_background_far", 170001),
        ("source_prototype", "f1", 0.7509),
        ("mechanism", "foreground_similarity_gap", 0.049),
        ("mechanism", "background_similarity_gap", 0.049),
        ("mechanism", "gradient_all_cosine", None),
    )
    for section, key, value in mutations:
        payload = passing_payload()
        payload[section][key] = value
        decision = evaluate_paired(payload)
        assert not decision["gate_pass"], (section, key)


def main():
    test_passing_paired_result_advances_only_to_confirmation()
    test_each_detector_or_geometry_regression_blocks_advancement()
    print("Discriminative dual-proxy paired-decision tests passed")


if __name__ == "__main__":
    main()
