import ast
from pathlib import Path


def _source(path: str) -> str:
    value = Path(path)
    assert value.is_file(), path
    return value.read_text(encoding="utf-8")


def test_archive_uses_exact_final_p2p_candidate_protocol():
    source = _source("extract_candidate_conditioned_proto_archive.py")
    assert "label_candidate_diagnostics" in source
    assert "evaluate_image" in source
    assert "_validate_exact_counts" in source
    assert '"border_filter": True' in source
    assert '"raw_cell_only_for_fp": True' in source
    assert '"reg_features"' in source
    assert '"cls_attn"' in source
    assert '"reg_attn"' in source
    assert '"reg_offset"' in source
    assert (
        "reg_features, cls_features, reg_attn, cls_attn = "
        "model.extract_features"
    ) in source


def test_loso_trains_only_tp_and_background_far():
    source = _source("audit_candidate_conditioned_proto_loso.py")
    assert "score_matched_group_sample" in source
    assert "build_loso_image_split" in source
    assert '"training_categories": ["tp", "background_far_fp"]' in source
    assert '"near_miss_fp"' in source
    assert "candidate_conditioned_gate" in source


def test_failed_gate_cannot_export_a_bank():
    source = _source("audit_candidate_conditioned_proto_loso.py")
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    conditional = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Subscript)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "gate"
    )
    lines = source.splitlines()

    def node_text(nodes):
        if not nodes:
            return ""
        start = min(node.lineno for node in nodes) - 1
        end = max(node.end_lineno for node in nodes)
        return "\n".join(lines[start:end])

    body_text = node_text(conditional.body)
    else_text = node_text(conditional.orelse)
    assert "torch.save" in body_text
    assert "torch.save" not in else_text


def test_bank_metadata_rejects_old_three_class_contract():
    source = _source("models/candidate_conditioned_prototype.py")
    assert "candidate_conditioned_metric_proto_v2_20260829" in source
    assert 'metadata.get("training_categories")' in source
    assert '["tp", "background_far_fp"]' in source
    assert "len(prototype_counts) != 2" in source


def test_gate_runner_rejects_stale_resume_artifacts_and_streams_progress():
    source = _source("run_candidate_conditioned_proto_gate_gpu3.sh")
    assert "candidate_conditioned_proto_archive_v2_20260829" in source
    assert "checkpoint_sha256" in source
    assert "archive identity mismatch" in source
    assert 'rm -f "${output}/candidate_conditioned_proto_bank.pth"' in source
    assert source.count("| tee ") >= 4
    assert "source_train_loso_cls_only" in source
    assert "source_train_loso_context" in source


def test_loso_and_fixed_scoring_use_the_bank_feature_mode():
    audit = _source("audit_candidate_conditioned_proto_loso.py")
    scorer = _source("score_candidate_conditioned_proto_bank.py")
    assert 'choices=("cls_only", "cls_reg_context")' in audit
    assert "build_candidate_feature_matrix" in audit
    assert 'metadata.get("feature_mode")' in scorer
    assert "build_candidate_feature_matrix" in scorer


def main():
    test_archive_uses_exact_final_p2p_candidate_protocol()
    test_loso_trains_only_tp_and_background_far()
    test_failed_gate_cannot_export_a_bank()
    test_bank_metadata_rejects_old_three_class_contract()
    test_gate_runner_rejects_stale_resume_artifacts_and_streams_progress()
    test_loso_and_fixed_scoring_use_the_bank_feature_mode()
    print("Candidate-conditioned prototype structure tests passed")


if __name__ == "__main__":
    main()
