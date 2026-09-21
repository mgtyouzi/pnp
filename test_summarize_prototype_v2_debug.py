from summarize_prototype_v2_debug import audit


def test_audit_accepts_single_initialization_and_complete_sampling():
    rows = [
        {
            "epoch": 20,
            "step_mean": {
                "prototype_positive": 4.0,
                "prototype_rejected_positive": 1.0,
                "prototype_hard_background": 16.0,
                "prototype_random_background": 16.0,
                "prototype_loss_raw": 0.0,
                "train_batch_fraction": 1.0,
            },
            "prototype_state": {
                "prototype_ready": 1,
                "prototype_initialized_now": 1,
                "prototype_periodic_refresh_enabled": 1,
                "prototype_refresh_interval_steps": 100,
                "prototype_epoch_refresh_events": 10,
                "prototype_epoch_refresh_post_positive_correct_rate": 0.9,
                "prototype_epoch_refresh_post_background_correct_rate": 0.7,
                "prototype_epoch_refresh_projector_drift_min": 0.99,
                "prototype_epoch_refresh_center_age_max": 100,
                "prototype_epoch_refresh_reprojected_rate": 1.0,
                "foreground_center_drift_cosine": 1.0,
                "background_center_drift_cosine": 1.0,
                "foreground_background_similarity_mean": 0.2,
                "foreground_background_similarity_max": 0.4,
            },
        },
        *[
        {
            "epoch": epoch,
            "step_mean": {
                "prototype_positive": 4.0,
                "prototype_rejected_positive": 1.0,
                "prototype_hard_background": 16.0,
                "prototype_random_background": 16.0,
                "prototype_loss_raw": 0.5,
                "gradient_all_proto_cls_ratio": 0.08,
                "gradient_all_cosine": 0.1,
                "prototype_positive_margin": 1.0,
                "prototype_background_margin": -1.0,
                "prototype_positive_correct_rate": 0.9,
                "prototype_background_correct_rate": 0.9,
                "prototype_hard_background_correct_rate": 0.8,
                "prototype_random_background_correct_rate": 1.0,
                "prototype_hard_background_bg_center_0_count": 8.0,
                "prototype_random_background_bg_center_0_count": 8.0,
                "train_batch_fraction": 1.0,
            },
            "prototype_state": {
                "prototype_ready": 1,
                "prototype_initialized_now": 0,
                "prototype_periodic_refresh_enabled": 1,
                "prototype_refresh_interval_steps": 100,
                "prototype_epoch_refresh_events": 10,
                "prototype_epoch_refresh_post_positive_correct_rate": 0.9,
                "prototype_epoch_refresh_post_background_correct_rate": 0.7,
                "prototype_epoch_refresh_projector_drift_min": 0.99,
                "prototype_epoch_refresh_center_age_max": 100,
                "prototype_epoch_refresh_reprojected_rate": 1.0,
                "foreground_center_drift_cosine": 0.99,
                "background_center_drift_cosine": 0.99,
                "foreground_background_similarity_mean": 0.2,
                "foreground_background_similarity_max": 0.4,
                "fresh_foreground_background_similarity_mean": 0.2,
                "fresh_foreground_background_similarity_max": 0.4,
                "foreground_fresh_old_alignment_cosine": 0.98,
                "background_fresh_old_alignment_cosine": 0.98,
                "foreground_updated_fresh_cosine": 0.99,
                "background_updated_fresh_cosine": 0.99,
                "projector_probe_drift_cosine": 0.99,
                "foreground_assignment_counts": [25, 25, 25, 25],
                "background_assignment_counts": [12, 13, 12, 13],
                "foreground_effective_prototypes": 4.0,
                "background_effective_prototypes": 4.0,
            },
        }
        for epoch in (21, 22, 23)
        ],
    ]

    failures, warnings, summary = audit(rows)

    assert failures == []
    assert warnings == []
    assert summary["status"] == "PASS"
    assert summary["mean_training_batch_fraction"] == 1.0
    assert summary["mechanism_gates"]["periodic_refresh_events_present"]
    assert summary["mechanism_gates"]["refresh_background_correct_at_least_0p50"]
    assert summary["mechanism_gates"]["refresh_uses_current_projector"]
    assert summary["final_background_center_source_composition"] == [
        {
            "center": 0,
            "hard_count_per_step": 8.0,
            "random_count_per_step": 8.0,
            "hard_fraction": 0.5,
        }
    ]


def test_audit_rejects_repeated_kmeans_and_allows_hard_only_background():
    rows = [
        {
            "epoch": 20,
            "step_mean": {
                "prototype_positive": 4.0,
                "prototype_hard_background": 16.0,
                "prototype_random_background": 0.0,
                "gradient_all_proto_cls_ratio": 0.1,
            },
            "prototype_state": {
                "prototype_ready": 1,
                "prototype_initialized_now": 1,
            },
        },
        {
            "epoch": 21,
            "step_mean": {
                "prototype_positive": 4.0,
                "prototype_hard_background": 16.0,
                "prototype_random_background": 0.0,
                "gradient_all_proto_cls_ratio": 0.1,
            },
            "prototype_state": {
                "prototype_ready": 1,
                "prototype_initialized_now": 1,
            },
        },
    ]

    failures, warnings, summary = audit(rows)

    assert any("exactly one KMeans" in message for message in failures)
    assert not any("random-background" in message for message in failures)
    assert any("hard background only" in message for message in warnings)
    assert summary["status"] == "FAIL"


def test_audit_warns_for_overlap_occupancy_weak_margin_and_weak_transfer():
    rows = [
        {
            "epoch": 20,
            "step_mean": {
                "prototype_positive": 10.0,
                "prototype_hard_background": 16.0,
                "prototype_random_background": 16.0,
            },
            "prototype_state": {
                "prototype_ready": 1,
                "prototype_initialized_now": 1,
                "foreground_background_similarity_mean": 0.6,
                "foreground_background_similarity_max": 0.99,
            },
        },
        *[
            {
                "epoch": epoch,
                "step_mean": {
                    "prototype_positive": 10.0,
                    "prototype_hard_background": 16.0,
                    "prototype_random_background": 16.0,
                    "prototype_loss_raw": 0.6,
                    "prototype_positive_margin": 0.2,
                    "prototype_background_margin": -0.2,
                    "gradient_all_proto_cls_ratio": 0.01,
                    "gradient_all_cosine": 0.0,
                },
                "prototype_state": {
                    "prototype_ready": 1,
                    "prototype_initialized_now": 0,
                    "foreground_center_drift_cosine": 0.99,
                    "background_center_drift_cosine": 0.99,
                    "foreground_background_similarity_mean": 0.6,
                    "foreground_background_similarity_max": 0.99,
                    "foreground_assignment_counts": [65, 20, 10, 5],
                    "background_assignment_counts": [52, 12, 10, 8, 7, 5, 5, 1],
                },
            }
            for epoch in (21, 22, 23)
        ],
    ]

    failures, warnings, summary = audit(rows)

    assert failures == []
    assert summary["status"] == "WARN"
    assert any("nearly overlaps" in message for message in warnings)
    assert any("occupancy is concentrated" in message for message in warnings)
    assert any("margin is weak" in message for message in warnings)
    assert any("gradient reaching shared" in message for message in warnings)


def test_audit_labels_truncated_epochs_as_rapid_only():
    rows = [
        {
            "epoch": 0,
            "step_mean": {
                "prototype_positive": 1.0,
                "prototype_hard_background": 1.0,
                "prototype_random_background": 1.0,
                "train_batch_fraction": 0.05,
            },
            "prototype_state": {
                "prototype_ready": 1,
                "prototype_initialized_now": 1,
            },
        }
    ]

    _, warnings, summary = audit(rows)

    assert summary["mean_training_batch_fraction"] == 0.05
    assert any("rapid mechanism gate" in message for message in warnings)


def test_audit_rejects_unavailable_shared_gradient_diagnostics():
    rows = [
        {
            "epoch": 0,
            "step_mean": {
                "prototype_positive": 10.0,
                "prototype_hard_background": 16.0,
                "prototype_random_background": 0.0,
                "gradient_all_proto_cls_ratio": 0.0,
                "gradient_all_cosine": 0.0,
                "gradient_cls_features_available": 0.0,
            },
            "prototype_state": {
                "prototype_ready": 1,
                "prototype_initialized_now": 1,
            },
        }
    ]

    failures, _, summary = audit(rows)

    assert any("shared-feature gradient diagnostics are unavailable" in item for item in failures)
    assert not summary["mechanism_gates"]["shared_gradient_diagnostics_available"]


def main():
    test_audit_accepts_single_initialization_and_complete_sampling()
    test_audit_rejects_repeated_kmeans_and_allows_hard_only_background()
    test_audit_warns_for_overlap_occupancy_weak_margin_and_weak_transfer()
    test_audit_labels_truncated_epochs_as_rapid_only()
    test_audit_rejects_unavailable_shared_gradient_diagnostics()
    print("Prototype-v2 debug summary tests passed")


if __name__ == "__main__":
    main()
