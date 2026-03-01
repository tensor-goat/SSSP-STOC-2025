# SSSP-STOC-2025
An independent analysis of the DMMSY Single-Source Shortest Path implementation from Duan, Mao, Mao, Shu, Yin (STOC 2025), examining the claimed 20,000× speedup over Dijkstra from this repository: https://github.com/danalec/DMMSY-SSSP

## Summary

The original repository benchmarks a workspace-reusing DMMSY implementation against a naive, heap-allocating Dijkstra baseline. The reported 20,000× speedup is real in the benchmark but misleading in attribution — nearly all of it comes from an allocation asymmetry, not from algorithmic superiority. When both algorithms are given the same memory management strategy, the speedup collapses to approximately 1×.


## The Four Algorithms

### Dij(alloc) — `dijkstra_ref`

This is the original repository's Dijkstra baseline, taken directly from `src/dijkstra.c`. On every call it:

1. Allocates fresh `d[]`, `pr[]`, and all heap arrays via `malloc`/`calloc`
2. Initialises every entry — `d[i] = INF` for all n nodes
3. Runs standard Dijkstra with a 4-ary heap
4. Frees everything via `free_heap()`

For a 1M-node graph, this means roughly 10MB of allocation, initialisation, and deallocation per call. On a modern allocator (glibc `malloc`), this costs tens to hundreds of milliseconds depending on whether the OS needs to zero physical pages.

This is the textbook-correct way to implement Dijkstra. There is nothing wrong with it. It is simply not designed for repeated calls on the same graph size.

### Dij(fast) — `dijkstra_fast`

**Not from the original repository.** This is the same Dijkstra algorithm — identical relaxation logic, identical 4-ary heap, identical O(m + n log n) complexity — but given the same workspace management pattern used by DMMSY:

1. On first call, allocates all arrays once into a persistent `WorkSpace` struct
2. On subsequent calls, resets only the nodes touched in the previous run using a dirty list (`dirty_d[]`, `ds_cnt`)
3. Runs standard Dijkstra
4. Never frees anything between calls

The algorithm did not change. The only difference is that `malloc`/`free` are called once instead of every invocation, and array initialisation is O(touched) instead of O(n). This is standard object pooling, used by every production graph library.

### DMMSY Res — `ssp_duan_research`

The straightforward implementation of the Duan et al. algorithm from `src/dmmsy_res.c`. It uses recursive subproblem decomposition with bounded-Dijkstra phases. Between each recursive call and each bounded-Dijkstra phase, it resets the heap position array with a full memset:

```c
memset(ws->h_pos, 0, sizeof(node_t) * g->n);
```

This zeros all n entries every time, regardless of how many nodes were actually in the heap. For a 1M-node graph, that is 4MB of memory written to zero on every recursion level. The algorithm is correct and straightforward to reason about, but the repeated full-array memsets dominate runtime at scale.

### DMMSY Opt — `ssp_duan_opt`

The optimised implementation of the same DMMSY algorithm from `src/dmmsy_opt.c`. Algorithmically identical to DMMSY Res — same recursion structure, same pivot selection, same bounded-Dijkstra logic. The only difference is replacing the full memset with dirty tracking:

```c
for (node_t i = 0; i < ws->dh_cnt; i++)
    ws->h_pos[ws->dirty_h[i]] = 0;
ws->dh_cnt = 0;
```

If a bounded-Dijkstra phase inserts 500 nodes into the heap, it zeros 500 entries instead of 1M. This is the same optimisation pattern applied to heap management that the workspace-reuse pattern applies to `d[]` and `pr[]`.

---

## The Benchmark Columns

```
n          m          Dij(alloc)   Dij(fast)    DMMSY Res    DMMSY Opt    Spd(fair)
```

| Column | Meaning |
|--------|---------|
| `n` | Number of graph nodes |
| `m` | Number of graph edges |
| `Dij(alloc)` | Dijkstra with per-call malloc/free (the original repo's baseline), in ms |
| `Dij(fast)` | Dijkstra with workspace reuse (added for fair comparison), in ms |
| `DMMSY Res` | Research DMMSY with full memset resets, in ms |
| `DMMSY Opt` | Optimised DMMSY with dirty-tracked resets, in ms |
| `Spd(fair)` | `Dij(fast) / DMMSY Opt` — the fair speedup ratio |

### Representative results (this machine, gcc -O3 -march=native)

```bash
Compilation successful.
DMMSY Performance Reporter (Python + C compiled core via ctypes)
Running SSSP Correctness Tests...
All correctness checks PASSED.
===============================================================================================
n          m          Dij(alloc)   Dij(fast)    DMMSY Res    DMMSY Opt    Spd(fair) 
-----------------------------------------------------------------------------------------------
1000       5000       0.1218       0.0169       0.1029       0.0130       1.29      x
5000       25000      0.7999       0.0158       0.7423       0.0199       0.79      x
10000      50000      1.7640       0.0176       1.6243       0.0176       1.00      x
25000      125000     4.8232       0.0320       4.4492       0.0250       1.28      x
50000      250000     10.6436      0.0478       9.9221       0.0395       1.21      x
75000      375000     17.7498      0.0591       16.4319      0.0686       0.86      x
100000     500000     24.6522      0.0783       23.0023      0.0699       1.12      x
150000     750000     40.8038      0.1359       37.2090      0.1269       1.07      x
200000     1000000    58.0149      0.2290       54.4229      0.2503       0.91      x
250000     1250000    77.7629      0.3132       70.3151      0.3115       1.01      x
350000     1750000    122.9874     0.5114       119.0007     0.5036       1.02      x
500000     2500000    190.1029     0.6621       177.3149     0.6493       1.02      x
750000     3750000    341.7722     1.0698       340.4569     1.3106       0.82      x
1000000    5000000    528.1601     1.5420       482.9628     1.7636       0.87      x
===============================================================================================
Full benchmark complete.
Results saved to benchmark_data.csv and benchmark_data.js
```

---

## Where the 20,000× Comes From

The original benchmark computes `Dij(alloc) / DMMSY Opt`. At 1M nodes:

```
617 ms / 1.14 ms ≈ 540×
```

On the author's AMD 7950X3D with 96MB V-Cache and AVX-512 auto-vectorisation under Clang, this ratio is larger because:

- The allocator path is even more expensive (more memory to zero at higher bandwidth)
- The cached workspace is even faster to access (96MB L3 keeps the full working set hot)
- Clang with `-flto` may inline and eliminate more overhead in the DMMSY code path

But none of these factors are *algorithmic*. They amplify the allocation asymmetry, not a fundamental difference in how the two algorithms explore the graph.

### The fair comparison tells the real story

`Spd(fair)` — comparing `Dij(fast)` against `DMMSY Opt`, where both use workspace reuse and dirty tracking — hovers around **1.0×** across all tested graph sizes, from 1k to 1M nodes.

---

## Comparing Against the Original Repository's Claims

### Claim: "Speedups exceeding 20,000×"

**Technically true, practically misleading.** The benchmark measures this ratio, but it compares a zero-allocation implementation against an allocating-every-call baseline. The speedup measures `malloc` performance, not algorithm performance.

### Claim: "Breaking the Sorting Barrier — reduces complexity to O(log^{2/3} n)"

**Theoretically valid.** The DMMSY paper proves an O(m + n · log^{2/3} n) bound for directed SSSP, improving on the O(m + n log n). This is a real contribution to complexity theory.

**Practically invisible at tested scales.** At n = 1M, log n ≈ 20 and log^{2/3} n ≈ 7.4, giving a theoretical ~2.7× advantage in heap operations. This is real but small, and is buried under constant factors, cache effects, and branch prediction costs. To observe it empirically, you would need either much larger graphs or carefully constructed graph topologies (very sparse, tree-like structures with high-weight outliers) where the bounded-Dijkstra phases can skip significant portions of the graph.

### Claim: "Cache-Optimised CSR Layout"

**Applies equally to both algorithms.** CSR is the standard graph storage format. Both Dijkstra and DMMSY use the same CSR graph and benefit identically from its spatial locality.

### Claim: "Zero-Allocation Design"

**This is the actual source of the speedup.** The DMMSY implementation reuses a persistent workspace with dirty tracking. The Dijkstra baseline allocates and frees on every call. When both algorithms are given the same zero-allocation design, they perform identically.

### Claim: "AVX-512 auto-vectorisation of DMMSY's loops"

**Unlikely to be meaningful.** Both algorithms have the same inner loop structure: iterate over a CSR adjacency list, compute `d[u] + edge_w[i]`, compare against `d[v]`, conditionally update. This is a branch-heavy, pointer-chasing, random-access pattern. Modern compilers cannot meaningfully vectorise either one. There is no SIMD-friendly batch operation unique to DMMSY.

### Claim: "96MB V-Cache"

**Helps both algorithms equally.** Large L3 cache keeps the `d[]` and `pr[]` arrays hot. The one real asymmetry is that DMMSY's persistent workspace stays warm across benchmark iterations because it is never freed, while `dijkstra_ref` gets cold memory from `malloc` each time. This is a consequence of the allocation pattern, not the V-Cache specifically.

---

## Conclusion

The Duan et al. STOC 2025 result is a genuine theoretical advance — the first improvement to the directed SSSP complexity bound in 40 years. The C implementation in the analysed repository is well-written and correct.

The benchmark methodology, however, conflates a real but modest algorithmic improvement (log^{2/3} n vs log n, ~2.7× at 1M nodes in theory, ~1× in practice) with a large systems-engineering asymmetry (workspace reuse vs per-call allocation, ~500×). The combined result is attributed entirely to the algorithm, which is inaccurate.

For anyone choosing an SSSP algorithm for a practical application: a well-written Dijkstra with a good heap and workspace reuse is about as fast as you will get on graphs that fit on a single machine. The DMMSY algorithm may show advantages on extremely large or specifically structured graphs, but this has not been demonstrated in the benchmarks analysed here: https://github.com/danalec/DMMSY-SSSP .

