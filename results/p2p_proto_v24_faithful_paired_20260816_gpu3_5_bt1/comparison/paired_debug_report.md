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

- Mean classification relative delta: `0.000498667590126929` (limit `0.0`)
- Mean regression relative delta: `0.0007364490528033804` (limit `0.005`)
- Paired optimization ready: `False`

| epoch | same samples | same RNG end | cls delta | reg delta | grad delta | control/proto clip |
|---:|---|---|---:|---:|---:|---|
| 0 | True | True | 0.042981 | -0.820291 | 0.09739257567484638 | 1.0/1.0 |
| 1 | True | True | 0.036552 | -0.271742 | 0.0035968104725836447 | 1.0/1.0 |
| 2 | True | True | -0.080757 | 1.057571 | 0.20687184374082568 | 1.0/1.0 |
| 3 | True | True | 0.055123 | 0.315485 | 0.0075791261426148004 | 1.0/1.0 |
