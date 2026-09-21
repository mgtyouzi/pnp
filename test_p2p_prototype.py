import torch

from models.p2p_prototype import (
    PROTOTYPE_IMPLEMENTATION_VERSION,
    CandidatePrototypeBank,
)


def test_expected_prototype_implementation_version():
    assert (
        PROTOTYPE_IMPLEMENTATION_VERSION
        == "prototype_v2_4_window_refresh_20260816"
    )


def test_candidate_masks_ignore_near_gt_and_select_hard_far_background():
    torch.manual_seed(0)
    module = CandidatePrototypeBank(
        feat_dim=4,
        embedding_dim=2,
        num_fg_prototypes=2,
        num_bg_prototypes=2,
        positive_radius=15.0,
        initial_positive_radius=10.0,
        background_radius=30.0,
        max_positive_per_image=8,
        max_hard_background_per_image=1,
        max_random_background_per_image=1,
    )
    points = torch.tensor([[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [40.0, 0.0], [50.0, 0.0]]])
    logits = torch.tensor([[[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [5.0, 0.0], [1.0, 0.0]]])
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0]])],
        "gt_labels": [torch.tensor([0])],
        "gt_nums": [1],
    }
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    selection = module.select_candidates(points, logits, targets, indices)

    assert selection.positive_mask.tolist() == [[True, False, False, False, False]]
    assert selection.hard_background_mask.tolist() == [[False, False, False, True, False]]
    assert selection.random_background_mask.tolist() == [[False, False, False, False, True]]
    assert selection.background_mask.tolist() == [[False, False, False, True, True]]
    assert selection.ignored_near_mask.tolist() == [[False, True, True, False, False]]


def test_far_hungarian_match_is_rejected_and_never_relabelled_as_background():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        positive_radius=15.0,
        initial_positive_radius=10.0,
        background_radius=30.0,
        max_hard_background_per_image=1,
        max_random_background_per_image=0,
    )
    points = torch.tensor([[[40.0, 0.0], [50.0, 0.0]]])
    logits = torch.tensor([[[5.0, 0.0], [4.0, 0.0]]])
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0]])],
        "gt_labels": [torch.tensor([0])],
        "gt_nums": [1],
    }
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    selection = module.select_candidates(points, logits, targets, indices)

    assert selection.matched_mask.tolist() == [[True, False]]
    assert selection.positive_mask.tolist() == [[False, False]]
    assert selection.rejected_positive_mask.tolist() == [[True, False]]
    assert selection.background_mask.tolist() == [[False, True]]
    assert torch.allclose(selection.matched_distance[0, 0], torch.tensor(40.0))


def test_background_selection_uses_anchor_and_regressed_point_geometry():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        positive_radius=15.0,
        background_radius=30.0,
        max_hard_background_per_image=1,
        max_random_background_per_image=0,
    )
    predicted = torch.tensor([[[50.0, 0.0], [60.0, 0.0]]])
    anchors = torch.tensor([[[10.0, 0.0], [60.0, 0.0]]])
    logits = torch.tensor([[[5.0, 0.0], [1.0, 0.0]]])
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0]])],
        "gt_labels": [torch.tensor([0])],
        "gt_nums": [1],
    }
    indices = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]

    selection = module.select_candidates(
        predicted, logits, targets, indices, anchor_points=anchors
    )

    assert selection.ignored_near_mask.tolist() == [[True, False]]
    assert selection.background_mask.tolist() == [[False, True]]


def test_multi_prototype_logits_use_the_nearest_center():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=2,
        num_bg_prototypes=2,
        temperature=0.1,
    )
    module.fg_prototypes.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    module.bg_prototypes.copy_(torch.tensor([[-1.0, 0.0], [0.0, -1.0]]))
    module.prototype_ready.fill_(True)

    embeddings = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])
    logits, distances = module.prototype_logits_from_embeddings(embeddings)

    assert logits[0, 0, 0] > logits[0, 0, 1]
    assert logits[0, 1, 1] > logits[0, 1, 0]
    assert torch.allclose(distances[0, 0], torch.tensor([0.0, 2.0]), atol=1e-6)


def test_fusion_is_disabled_until_prototypes_are_ready():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        fusion_alpha=0.1,
    )
    raw_logits = torch.zeros(1, 1, 2)
    proto_logits = torch.tensor([[[2.0, -2.0]]])

    assert torch.equal(module.fuse_logits(raw_logits, proto_logits), raw_logits)

    module.prototype_ready.fill_(True)
    fused = module.fuse_logits(raw_logits, proto_logits)
    assert torch.allclose(fused[0, 0], torch.tensor([0.1, -0.1]), atol=1e-6)


def test_epoch_finalize_builds_normalized_multi_prototypes_and_queues():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=2,
        num_bg_prototypes=2,
        foreground_queue_size=8,
        background_queue_size=8,
        kmeans_iterations=5,
    )
    module.begin_epoch()
    module.cache_embeddings(
        torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
        torch.tensor([[-1.0, 0.0], [-0.9, -0.1], [0.0, -1.0], [-0.1, -0.9]]),
    )

    diagnostics = module.finalize_epoch()

    assert module.prototype_ready.item()
    assert module.fg_queue_count.item() == 4
    assert module.bg_queue_count.item() == 4
    assert torch.allclose(module.fg_prototypes.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(module.bg_prototypes.norm(dim=1), torch.ones(2), atol=1e-5)
    assert diagnostics["foreground_gathered"] == 4
    assert diagnostics["background_gathered"] == 4


def test_epoch_finalize_keeps_fifo_history_and_ema_updates_existing_center():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        foreground_queue_size=8,
        background_queue_size=8,
        prototype_momentum=0.5,
    )
    module.begin_epoch()
    module.cache_embeddings(torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]]))
    module.finalize_epoch()

    module.begin_epoch()
    module.cache_embeddings(torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, -1.0]]))
    module.finalize_epoch()

    expected = torch.tensor([2.0 ** -0.5, 2.0 ** -0.5])
    assert module.fg_queue_count.item() == 2
    assert module.bg_queue_count.item() == 2
    assert torch.allclose(module.fg_queue[:2], torch.eye(2), atol=1e-6)
    assert torch.allclose(module.bg_queue[:2], -torch.eye(2), atol=1e-6)
    assert torch.allclose(module.fg_prototypes[0], expected, atol=1e-6)
    assert torch.allclose(module.bg_prototypes[0], -expected, atol=1e-6)


def test_projector_scales_only_the_gradient_returning_to_p2p_features():
    torch.manual_seed(7)
    full = CandidatePrototypeBank(
        feat_dim=4,
        embedding_dim=3,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        input_gradient_scale=1.0,
    )
    scaled = CandidatePrototypeBank(
        feat_dim=4,
        embedding_dim=3,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        input_gradient_scale=0.25,
    )
    scaled.load_state_dict(full.state_dict())
    full.train()
    scaled.train()
    x_full = torch.tensor([[[0.2, -0.3, 0.5, 0.7]]], requires_grad=True)
    x_scaled = x_full.detach().clone().requires_grad_(True)
    weights = torch.tensor([[[0.4, -0.2, 0.7]]])

    (full.project(x_full) * weights).sum().backward()
    (scaled.project(x_scaled) * weights).sum().backward()

    assert torch.allclose(x_scaled.grad, 0.25 * x_full.grad, atol=1e-6, rtol=1e-5)
    for full_param, scaled_param in zip(full.projector.parameters(), scaled.projector.parameters()):
        assert torch.allclose(scaled_param.grad, full_param.grad, atol=1e-6, rtol=1e-5)


def test_positive_queue_sampling_keeps_hard_and_easy_matches():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        max_positive_per_image=3,
    )
    scores = torch.tensor([0.05, 0.15, 0.50, 0.85, 0.95])
    indices = torch.arange(5)

    sampled = module.sample_positive_indices(indices, scores)

    assert sampled.tolist() == [0, 2, 4]


def test_random_background_sampling_does_not_advance_global_rng():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        max_hard_background_per_image=0,
        max_random_background_per_image=2,
        sampling_seed=17,
    )
    points = torch.tensor([[[40.0, 0.0], [50.0, 0.0], [60.0, 0.0]]])
    logits = torch.zeros(1, 3, 2)
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0]])],
        "gt_labels": [torch.tensor([0])],
        "gt_nums": [1],
    }
    indices = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]

    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    first = module.select_candidates(points, logits, targets, indices)
    after = torch.random.get_rng_state().clone()

    assert torch.equal(before, after)
    assert first.random_background_mask.sum().item() == 2
    assert module.sampling_step.item() == 1


def test_random_background_can_be_disabled_for_a_hard_only_bank():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=1,
        num_bg_prototypes=2,
        background_radius=30.0,
        max_hard_background_per_image=2,
        max_random_background_per_image=0,
    )
    points = torch.tensor([[[40.0, 0.0], [50.0, 0.0], [60.0, 0.0]]])
    logits = torch.tensor([[[5.0, 0.0], [4.0, 0.0], [1.0, 0.0]]])
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0]])],
        "gt_labels": [torch.tensor([0])],
        "gt_nums": [1],
    }
    indices = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]

    selection = module.select_candidates(points, logits, targets, indices)

    assert selection.hard_background_mask.sum().item() == 2
    assert selection.random_background_mask.sum().item() == 0
    assert torch.equal(selection.background_mask, selection.hard_background_mask)


def test_step_diagnostics_separate_candidate_sources_and_center_assignments():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=2,
        num_bg_prototypes=2,
        max_hard_background_per_image=1,
        max_random_background_per_image=1,
    )
    module.fg_prototypes.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    module.bg_prototypes.copy_(torch.tensor([[-1.0, 0.0], [0.0, -1.0]]))
    module.prototype_ready.fill_(True)
    embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]])
    points = torch.tensor([[[0.0, 0.0], [20.0, 0.0], [40.0, 0.0], [50.0, 0.0]]])
    logits = torch.tensor([[[4.0, 0.0], [2.0, 0.0], [3.0, 0.0], [1.0, 0.0]]])
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0]])],
        "gt_labels": [torch.tensor([0])],
        "gt_nums": [1],
    }
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    _, diagnostics = module.compute_loss_and_cache(
        embeddings, points, logits, targets, indices
    )

    for key in (
        "prototype_positive_margin_p10",
        "prototype_positive_correct_rate",
        "prototype_hard_background_margin_mean",
        "prototype_random_background_margin_mean",
        "raw_score_rejected_positive_count",
        "prototype_positive_fg_center_0_count",
        "prototype_background_bg_center_0_count",
        "prototype_hard_background_bg_center_0_count",
        "prototype_random_background_bg_center_0_count",
    ):
        assert key in diagnostics, (
            f"missing diagnostic key: {key}; available keys: "
            f"{sorted(diagnostics)}"
        )
    assert diagnostics["prototype_positive_correct_rate"] == 1.0
    assert diagnostics["prototype_background_correct_rate"] == 1.0


def test_source_weighted_loss_emphasizes_hard_background():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        temperature=0.2,
        background_radius=30.0,
        max_hard_background_per_image=1,
        max_random_background_per_image=1,
        loss_mode="source_weighted",
        positive_term_weight=1.0,
        hard_background_term_weight=2.0,
        random_background_term_weight=0.25,
    )
    module.fg_prototypes.copy_(torch.tensor([[1.0, 0.0]]))
    module.bg_prototypes.copy_(torch.tensor([[-1.0, 0.0]]))
    module.prototype_ready.fill_(True)
    embeddings = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]])
    points = torch.tensor([[[0.0, 0.0], [40.0, 0.0], [50.0, 0.0]]])
    raw_logits = torch.tensor([[[2.0, 0.0], [5.0, 0.0], [1.0, 0.0]]])
    targets = {
        "gt_points": [torch.tensor([[0.0, 0.0]])],
        "gt_labels": [torch.tensor([0])],
        "gt_nums": [1],
    }
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    loss, diagnostics = module.compute_loss_and_cache(
        embeddings, points, raw_logits, targets, indices
    )
    prototype_logits, _ = module.prototype_logits_from_embeddings(embeddings)
    positive = torch.nn.functional.cross_entropy(
        prototype_logits[:, 0], torch.zeros(1, dtype=torch.long)
    )
    hard = torch.nn.functional.cross_entropy(
        prototype_logits[:, 1], torch.ones(1, dtype=torch.long)
    )
    random = torch.nn.functional.cross_entropy(
        prototype_logits[:, 2], torch.ones(1, dtype=torch.long)
    )
    expected = (positive + 2.0 * hard + 0.25 * random) / 3.25

    assert torch.allclose(loss, expected, atol=1e-6)
    assert diagnostics["prototype_loss_mode_source_weighted"] == 1.0
    assert diagnostics["prototype_loss_hard_background"] > diagnostics[
        "prototype_loss_random_background"
    ]


def test_kmeans_ema_refreshes_every_center_from_current_epoch_modes():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=2,
        num_bg_prototypes=2,
        foreground_queue_size=16,
        background_queue_size=16,
        prototype_momentum=0.0,
        update_mode="kmeans_ema",
        minimum_assignment_share=0.1,
        kmeans_iterations=10,
    )
    module.fg_prototypes.copy_(torch.tensor([[1.0, 0.0], [0.99, 0.01]]))
    module.bg_prototypes.copy_(torch.tensor([[-1.0, 0.0], [-0.99, -0.01]]))
    module.prototype_ready.fill_(True)
    module.begin_epoch()
    module.cache_embeddings(
        torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]),
        torch.tensor([[-1.0, 0.0], [-0.99, -0.01], [0.0, -1.0], [-0.01, -0.99]]),
    )

    diagnostics = module.finalize_epoch()
    fg_similarity = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).matmul(
        module.fg_prototypes.t()
    )
    bg_similarity = torch.tensor([[-1.0, 0.0], [0.0, -1.0]]).matmul(
        module.bg_prototypes.t()
    )

    assert torch.all(fg_similarity.max(dim=1).values > 0.99)
    assert torch.all(bg_similarity.max(dim=1).values > 0.99)
    assert diagnostics["prototype_update_mode_kmeans_ema"] == 1
    assert sorted(diagnostics["foreground_assignment_counts"]) == [2, 2]
    assert sorted(diagnostics["background_assignment_counts"]) == [2, 2]
    for key in (
        "foreground_fresh_old_alignment_cosine",
        "background_fresh_old_alignment_cosine",
        "foreground_updated_fresh_cosine",
        "background_updated_fresh_cosine",
        "fresh_foreground_background_similarity_mean",
        "fresh_foreground_background_similarity_max",
    ):
        assert key in diagnostics
        assert torch.isfinite(torch.tensor(diagnostics[key]))


def test_projector_probe_reports_embedding_space_drift():
    torch.manual_seed(11)
    module = CandidatePrototypeBank(
        feat_dim=4,
        embedding_dim=3,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        foreground_queue_size=4,
        background_queue_size=4,
    )
    module.begin_epoch()
    module.cache_embeddings(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[-1.0, 0.0, 0.0]]))
    first = module.finalize_epoch()

    with torch.no_grad():
        module.projector[-1].weight.mul_(-1.0)
        module.projector[-1].bias.mul_(-1.0)
    module.begin_epoch()
    module.cache_embeddings(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[-1.0, 0.0, 0.0]]))
    second = module.finalize_epoch()

    assert first["projector_probe_initialized_now"] == 1
    assert second["projector_probe_initialized_now"] == 0
    assert second["projector_probe_drift_cosine"] < 0.0


def test_epoch_diagnostics_identify_exact_cross_class_collision():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=2,
        num_bg_prototypes=2,
        dead_prototype_patience=3,
    )
    module.fg_prototypes.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    module.bg_prototypes.copy_(torch.tensor([[0.99, 0.01], [0.0, -1.0]]))
    module.prototype_ready.fill_(True)
    module.begin_epoch()
    module.cache_embeddings(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, -1.0]]),
    )

    diagnostics = module.finalize_epoch()

    assert len(diagnostics["foreground_background_similarity_matrix"]) == 2
    assert len(diagnostics["foreground_background_similarity_matrix"][0]) == 2
    assert diagnostics["worst_cross_foreground_index"] == 0
    assert diagnostics["worst_cross_background_index"] == 0
    assert diagnostics["foreground_assignment_counts"] == [1, 1]
    assert diagnostics["background_assignment_counts"] == [1, 1]
    assert diagnostics["foreground_effective_prototypes"] == 2.0
    assert diagnostics["background_effective_prototypes"] == 2.0


def test_periodic_refresh_initializes_only_after_complete_window():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        foreground_queue_size=8,
        background_queue_size=8,
        prototype_momentum=0.0,
        update_mode="kmeans_ema",
        refresh_interval_steps=2,
    )
    module.begin_epoch()
    module.cache_embeddings(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]])
    )
    first = module.maybe_refresh_prototypes()

    assert first["prototype_refresh_event"] == 0
    assert module.prototype_ready.item() == 0
    assert module.refresh_window_steps.item() == 1
    assert module.center_age_steps.item() == 1

    module.cache_embeddings(
        torch.tensor([[0.9, 0.1]]), torch.tensor([[-0.9, -0.1]])
    )
    second = module.maybe_refresh_prototypes()

    assert second["prototype_refresh_event"] == 1
    assert second["prototype_initialized_now"] == 1
    assert second["prototype_refresh_post_positive_correct_rate"] == 1.0
    assert second["prototype_refresh_post_background_correct_rate"] == 1.0
    assert module.prototype_ready.item() == 1
    assert module.prototype_refresh_count.item() == 1
    assert module.refresh_window_steps.item() == 0
    assert module.center_age_steps.item() == 0


def test_periodic_refresh_uses_only_latest_window_for_center_update():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        foreground_queue_size=8,
        background_queue_size=8,
        prototype_momentum=0.0,
        update_mode="kmeans_ema",
        refresh_interval_steps=1,
    )
    module.begin_epoch()
    module.cache_embeddings(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]])
    )
    module.maybe_refresh_prototypes()
    module.cache_embeddings(
        torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, -1.0]])
    )
    second = module.maybe_refresh_prototypes()

    assert second["prototype_refresh_event"] == 1
    assert torch.allclose(module.fg_prototypes[0], torch.tensor([0.0, 1.0]))
    assert torch.allclose(module.bg_prototypes[0], torch.tensor([0.0, -1.0]))
    assert module.fg_queue_count.item() == 2
    assert module.bg_queue_count.item() == 2


def test_periodic_refresh_reprojects_raw_support_with_current_projector():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        foreground_queue_size=8,
        background_queue_size=8,
        prototype_momentum=0.0,
        update_mode="kmeans_ema",
        refresh_interval_steps=1,
    )
    projector = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        projector.weight.copy_(torch.eye(2))
    module.projector = projector
    module.begin_epoch()
    module.cache_source_features(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]])
    )
    with torch.no_grad():
        projector.weight.copy_(torch.tensor([[0.0, -1.0], [1.0, 0.0]]))

    diagnostics = module.maybe_refresh_prototypes()

    assert diagnostics["prototype_refresh_reprojected_current_projector"] == 1
    assert torch.allclose(module.fg_prototypes[0], torch.tensor([0.0, 1.0]))
    assert torch.allclose(module.bg_prototypes[0], torch.tensor([0.0, -1.0]))


def test_periodic_finalize_does_not_mix_full_epoch_embeddings_back_in():
    module = CandidatePrototypeBank(
        feat_dim=2,
        embedding_dim=2,
        num_fg_prototypes=1,
        num_bg_prototypes=1,
        foreground_queue_size=8,
        background_queue_size=8,
        prototype_momentum=0.0,
        update_mode="kmeans_ema",
        refresh_interval_steps=1,
    )
    module.begin_epoch()
    module.cache_embeddings(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]])
    )
    module.maybe_refresh_prototypes()
    module.cache_embeddings(
        torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, -1.0]])
    )
    module.maybe_refresh_prototypes()

    before_fg = module.fg_prototypes.clone()
    before_bg = module.bg_prototypes.clone()
    diagnostics = module.finalize_epoch()

    assert diagnostics["prototype_periodic_refresh_enabled"] == 1
    assert diagnostics["prototype_epoch_end_refresh_event"] == 0
    assert diagnostics["prototype_epoch_refresh_events"] == 2
    assert diagnostics["prototype_epoch_refresh_center_age_max"] == 1.0
    assert torch.equal(module.fg_prototypes, before_fg)
    assert torch.equal(module.bg_prototypes, before_bg)


def main():
    test_expected_prototype_implementation_version()
    test_candidate_masks_ignore_near_gt_and_select_hard_far_background()
    test_far_hungarian_match_is_rejected_and_never_relabelled_as_background()
    test_background_selection_uses_anchor_and_regressed_point_geometry()
    test_multi_prototype_logits_use_the_nearest_center()
    test_fusion_is_disabled_until_prototypes_are_ready()
    test_epoch_finalize_builds_normalized_multi_prototypes_and_queues()
    test_epoch_finalize_keeps_fifo_history_and_ema_updates_existing_center()
    test_projector_scales_only_the_gradient_returning_to_p2p_features()
    test_positive_queue_sampling_keeps_hard_and_easy_matches()
    test_random_background_sampling_does_not_advance_global_rng()
    test_random_background_can_be_disabled_for_a_hard_only_bank()
    test_step_diagnostics_separate_candidate_sources_and_center_assignments()
    test_source_weighted_loss_emphasizes_hard_background()
    test_kmeans_ema_refreshes_every_center_from_current_epoch_modes()
    test_projector_probe_reports_embedding_space_drift()
    test_epoch_diagnostics_identify_exact_cross_class_collision()
    test_periodic_refresh_initializes_only_after_complete_window()
    test_periodic_refresh_uses_only_latest_window_for_center_update()
    test_periodic_refresh_reprojects_raw_support_with_current_projector()
    test_periodic_finalize_does_not_mix_full_epoch_embeddings_back_in()
    print("P2P prototype tests passed")


if __name__ == "__main__":
    main()
