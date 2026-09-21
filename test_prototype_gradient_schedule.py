from prototype_gradient_schedule import select_gradient_probe


def test_gradient_probe_rotates_one_tensor_at_a_time():
    probes = [
        ("cls_features", object()),
        ("fpn_0", object()),
        ("fpn_1", object()),
    ]

    selected_names = []
    for event_index in range(6):
        selected = select_gradient_probe(probes, event_index)
        assert len(selected) == 1
        selected_names.append(selected[0][0])

    assert selected_names == [
        "cls_features",
        "fpn_0",
        "fpn_1",
        "cls_features",
        "fpn_0",
        "fpn_1",
    ]


def test_gradient_probe_handles_empty_input():
    assert select_gradient_probe([], 3) == []


if __name__ == "__main__":
    test_gradient_probe_rotates_one_tensor_at_a_time()
    test_gradient_probe_handles_empty_input()
    print("Prototype gradient schedule tests passed")
