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
| 0 | True | True | -0.000862 | 0.016157 | -0.18322914838790894 | 1.0/1.0 |
| 1 | True | True | 0.010976 | 0.046545 | 0.23885908037424097 | 1.0/1.0 |
| 2 | True | True | 0.016681 | 0.139031 | 0.10490151703357697 | 1.0/1.0 |
| 3 | True | True | 0.007665 | -0.114573 | 0.08894878894090663 | 1.0/1.0 |
| 4 | True | True | -0.067629 | -0.153770 | -0.6611261552572252 | 1.0/1.0 |
| 5 | True | True | -0.017514 | 0.354082 | 0.017293736338615195 | 1.0/1.0 |
| 6 | True | True | -0.051911 | 0.722223 | -0.3557866734266282 | 1.0/1.0 |
| 7 | True | True | -0.006736 | 0.139955 | 0.25614642292261114 | 1.0/1.0 |
