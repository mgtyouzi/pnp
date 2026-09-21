# Prototype-v2 paired mechanism check

- Base initialization exact: `True`
- Init checkpoint exact: `True`
- Paired epochs present: `True`
- Dataset clean: `True`
- Runtime data errors absent: `True`
- Data sanitization exact across runs: `True`
- First-batch inputs exact: `True`
- Initial raw P2P forward exact: `True`
- Initial matcher exact: `True`
- Sample order exact: `True`
- RNG start exact: `True`
- RNG end exact: `True`
- Attribution valid: `True`

| epoch | same samples | same RNG end | cls delta | reg delta | grad delta | control/proto clip |
|---:|---|---|---:|---:|---:|---|
| 0 | True | True | -0.001145 | 0.021079 | -0.13335977047681813 | 1.0/1.0 |
| 1 | True | True | 0.001126 | -0.087812 | -0.17037258207798 | 1.0/1.0 |
| 2 | True | True | 0.000830 | 0.309086 | -0.7159276142716409 | 1.0/1.0 |
| 3 | True | True | 0.025636 | -0.104415 | -0.42578561455011377 | 1.0/1.0 |
| 4 | True | True | -0.022963 | 0.258277 | -0.012419082224369049 | 1.0/1.0 |
| 5 | True | True | 0.061655 | 0.479677 | -0.2142510566115381 | 1.0/1.0 |
| 6 | True | True | -0.017786 | -0.071242 | -0.14904040664434426 | 1.0/1.0 |
| 7 | True | True | -0.015074 | 0.475813 | 0.314867443740368 | 1.0/1.0 |
