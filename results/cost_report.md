# Cost report (PART 4 — real cost measurement)

Structured (neuron-level) pruning on the two hidden layers of the 2-128-128-3 MLP, then `prune/compress.py:compress_model` into a genuinely smaller dense model. FLOP and parameter counts are exact; the speedup is a microbenchmark (`>1x` is the durable claim, the exact multiple is machine/BLAS-dependent).

Environment: Windows-11-10.0.26200-SP0 | Python 3.14.0 | NumPy 2.4.2

| target sparsity | achieved | active params | total params | dense FLOPs | sparse FLOPs | dense NumPy (s) | compressed (s) | speedup |
|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.4887 | 8963 | 17283 | 2195648 | 1130688 | 0.0696 | 0.0169 | 4.12x |
| 0.75 | 0.7331 | 4803 | 17283 | 2195648 | 598208 | 0.0555 | 0.0104 | 5.33x |
| 0.90 | 0.8782 | 2333 | 17283 | 2195648 | 282048 | 0.0502 | 0.0085 | 5.89x |

Reproduce: `python -m evaluation.run_cost_report`. Note that `x @ (W*mask)` is a full dense matmul, so the FLOP/param savings are realized only by the compressed model, not by the masked one (see `evaluation/cost.py:HONEST_COST_NOTE` and DESIGN.md §3).
