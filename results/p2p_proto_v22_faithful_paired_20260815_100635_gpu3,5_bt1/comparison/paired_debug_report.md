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

- Mean classification relative delta: `-5.8411039277635405e-05` (limit `0.0`)
- Mean regression relative delta: `0.003540546268799268` (limit `0.005`)
- Paired optimization ready: `True`

| epoch | same samples | same RNG end | cls delta | reg delta | grad delta | control/proto clip |
|---:|---|---|---:|---:|---:|---|
| 0 | True | True | -0.009878 | 0.365201 | -0.1191695164396529 | 1.0/1.0 |
| 1 | True | True | -0.019913 | 0.962218 | -0.09748734821301852 | 1.0/1.0 |
| 2 | True | True | 0.013146 | 0.142572 | -0.03423289956121778 | 1.0/1.0 |
| 3 | True | True | 0.010727 | -0.187766 | 0.021802792280007477 | 1.0/1.0 |
