from diagnostic_aggregation import accumulate_finite_diagnostics


def test_non_finite_values_do_not_poison_epoch_means():
    sums = {}
    counts = {}
    accumulate_finite_diagnostics(
        sums,
        counts,
        {"selected_probability": float("nan"), "selected_count": 0.0},
    )
    accumulate_finite_diagnostics(
        sums,
        counts,
        {"selected_probability": 0.31, "selected_count": 4.0},
    )
    assert sums["selected_probability"] == 0.31
    assert counts["selected_probability"] == 1
    assert sums["selected_count"] == 4.0
    assert counts["selected_count"] == 2


def main():
    test_non_finite_values_do_not_poison_epoch_means()
    print("Finite diagnostic aggregation tests passed")


if __name__ == "__main__":
    main()
