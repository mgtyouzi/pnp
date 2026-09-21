# Prototype-v2.2 decision

- Mode: `rapid`
- All gates pass: `False`
- Ready for long run: `False`
- Next action: `stop_and_fix_failed_mechanism`

| gate | pass |
|---|---:|
| prototype_mechanism_ready | False |
| paired_attribution_valid | True |
| paired_optimization_ready | False |
| configured_raw_projected_comparison_valid | True |
| projected_balanced_accuracy_at_least_0p65 | True |
| projected_foreground_recall_at_least_0p60 | False |
| projected_hard_background_recall_at_least_0p60 | True |
| projected_balanced_accuracy_not_worse_than_raw | True |
| faithful_updates_use_full_epoch | True |
