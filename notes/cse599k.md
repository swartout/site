---
title: CSE 599K
---

# CSE 599K: Systems for ML

## Lecture 1, Sept 25

*Missed lecture, transformers overview.*

## Lecture 2, Sept 30

- Consider the difference between prefill and autoregressive phases
- Consider FLOPs / mem bandwidth for finding memory vs compute bottlenecks
- For a linear layer (GEMM, general matmul) between $X \times W$ where $X \sim (N, K)$, $W \sim (K, M)$:
    - Compute: $2 \cdot N \cdot K \cdot M$
    - Memory: $2 \cdot N \cdot K + 2 \cdot K \cdot M + 2 \cdot N \cdot M$
- Attention is memory bound, wheras linear layers are compute bound
    - Due to attention having lower "arithmetic density"
- Because decoding is only multiplying by a linear layer (GEMV), it is also memory-bound
- Execution time heuristic: $S \cdot  \frac{2 \cdot n_{\text{params}}}{\text{FLOPS}} + G \cdot \frac{2 \cdot n_{\text{params}}}{\text{MBW}}$
    - $S$: prefill length
    - $G$: generated length
- Useful SLOs:
    - TTFT: time to first token (prefill)
    - TPOT: time per output token (autoregressive)
