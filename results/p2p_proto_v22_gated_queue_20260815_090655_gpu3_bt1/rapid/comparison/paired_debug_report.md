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

- Mean classification relative delta: `0.0034384444439921552` (limit `0.0`)
- Mean regression relative delta: `0.0027859101149440638` (limit `0.02`)
- Paired optimization ready: `False`

| epoch | same samples | same RNG end | cls delta | reg delta | grad delta | control/proto clip |
|---:|---|---|---:|---:|---:|---|
| 0 | True | True | -0.000549 | -0.002307 | -0.10678522408008573 | 1.0/1.0 |
| 1 | True | True | 0.002165 | 0.050193 | -0.21998553305864332 | 1.0/1.0 |
| 2 | True | True | 0.017602 | 0.087354 | 0.36545503765344645 | 1.0/1.0 |
| 3 | True | True | 0.013659 | -0.449299 | -0.055408156812191134 | 1.0/1.0 |
| 4 | True | True | 0.004903 | -0.065752 | -0.267289489209652 | 1.0/1.0 |
| 5 | True | True | 0.026926 | -0.298166 | -0.4062657746672633 | 1.0/1.0 |
| 6 | True | True | -0.000107 | 0.536668 | 0.04291816949844374 | 1.0/1.0 |
| 7 | True | True | 0.010869 | 0.397046 | 0.6476287880539895 | 1.0/1.0 |
