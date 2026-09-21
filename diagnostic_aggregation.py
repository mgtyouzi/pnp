import math


def accumulate_finite_diagnostics(sums, counts, diagnostics):
    """Accumulate scalar diagnostics without allowing empty-batch NaNs to spread."""
    for key, value in diagnostics.items():
        if not isinstance(value, (int, float)):
            continue
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            continue
        sums[key] = sums.get(key, 0.0) + numeric_value
        counts[key] = counts.get(key, 0) + 1
