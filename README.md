# CUDA Vector Triad — SIMD vs. SIMT Comparative Report

**Course:** CEPARCO · **Section:** S01 · **Project:** Deep Dive: CUDA Concept and Programming

**Group members:** Saoud Abdulrahman; Justin Lance Lim; Kent Xavier Lopez; Francis Medin Nicolas.

## 1. Project specification and implementation

For single-precision (`float`) vectors of length `n`, the computation is `A[i] = B[i] + C[i] * D[i]`. The five implementations are (1) C/C++ CPU baseline, (2) CUDA Unified Memory without prefetch/advice, (3) CUDA with prefetching, (4) CUDA with prefetching plus CPU page touching, and (5) CUDA with prefetching, page touching, and memory advice.

Every CUDA version uses the **grid-stride loop**:

```cpp
__global__ void vectorTriadCUDA(int n, float *A, float *B, float *C, float *D) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        A[i] = B[i] + C[i] * D[i];
    }
}
```

The input initialization is unchanged from our SIMD project: `B[i] = sin(i * 0.001)`, `C[i] = cos(i * 0.002)`, and `D[i] = sin(i * 0.0005) + 1.5`. All arrays use float storage. Our CUDA implementation uses `cudaMallocManaged()` and `cudaDeviceSynchronize()`. Prefetching and memory advice are included because the assignment asks for them. Compatibility wrappers adjust their function arguments for newer CUDA toolkits without changing the core kernel.

## 2. Environment and measurement procedure

| Item | Details |
|---|---|
| CUDA execution | Google Colab, NVIDIA GPU — **[insert actual GPU model from `!nvidia-smi`]** |
| Colab CPU | **[insert CPU model from `!lscpu`]** |
| CUDA toolkit | **[insert actual `!nvcc --version` result]** |
| Earlier SIMD | Windows, Visual Studio Debug and Release; CPU/flags **[insert verified details]** |
| Input sizes | `2^20`, `2^24`, `2^26` elements (and smaller boundary tests) |
| GPU threads per block | 256, 512, 1024 |
| Timing repetitions | 30 measured runs after a separately reported first launch |
| Timing method | CPU host steady clock; CUDA events for GPU kernel-only time |
| Correctness | CPU reference, relative-scaled tolerance `1e-5`; 16 guard floats after A |

Allocation, initialization, migration/prefetch preparation, checking, and (for page variants) CPU page touching are excluded from the main average kernel timings. All reported speedups are therefore **CPU loop time / warmed GPU kernel-only time**, not end-to-end speedups. The historical Windows data use an additional, different machine.

## 3. Measured Colab results

### 3.1 Main comparison — 256 threads per block

| Version | `2^20` avg (ms) | `2^24` avg (ms) | `2^26` avg (ms) |
|---|---:|---:|---:|
| C/C++ (Colab) | 2.0994 | 31.1446 | 84.0676 |
| CUDA Unified | 0.0695 | 1.0186 | 4.1323 |
| CUDA Prefetch | 0.0687 | 1.0338 | 4.1163 |
| CUDA Prefetch + Pages | 0.0691 | 1.0252 | 4.1188 |
| CUDA Prefetch + Pages + Advice | 0.0692 | 1.0254 | 4.1191 |

### 3.2 Speedup relative to the Colab C baseline (same session)

**Speedup = average CPU time / average GPU kernel time.**

| Version | `2^20` | `2^24` | `2^26` |
|---|---:|---:|---:|
| CUDA Unified | 30.22× | 30.57× | 20.34× |
| CUDA Prefetch | 30.57× | 30.13× | 20.42× |
| CUDA Prefetch + Pages | 30.39× | 30.38× | 20.41× |
| CUDA Prefetch + Pages + Advice | 30.33× | 30.37× | 20.41× |

At `2^26`, the Colab CPU averaged **84.0676 ms** while Unified averaged **4.1323 ms**, and Prefetch averaged **4.1163 ms** (at 256 threads/block). The CUDA variants deliver roughly **20× lower steady-state kernel time** in this run, but this excludes overhead associated with moving and preparing the arrays. The CPU averages varied significantly between earlier Colab sessions, so this is an observed result for this specific run rather than a guaranteed ratio.

### 3.3 Effect of thread-block size at `2^26`

| CUDA variant | 256 (ms) | 512 (ms) | 1024 (ms) | Best measured |
|---|---:|---:|---:|---|
| CUDA Unified | 4.1323 | 4.1015 | 4.0843 | 1024 |
| CUDA Prefetch | 4.1163 | 4.0775 | 4.2155 | 512 |
| CUDA Prefetch + Pages | 4.1188 | 4.0724 | 4.2297 | 512 |
| CUDA Prefetch + Pages + Advice | 4.1191 | 4.0777 | 4.3801 | 512 |

**Block recommendation:** 512 threads per block is a practical general choice in the `2^26` results: it is fastest for Prefetch, Pages, and Advice in this run, though Unified is slightly fastest at 1024. The 1024-thread configuration showed greater variation in some versions; one setting should not be called universally optimal without repeating full sessions.

### 3.4 Memory preparation at `2^26`, 256 threads/block

| Mode | Reported setup (ms) | Warmed kernel avg (ms) | First launch (ms) |
|---|---:|---:|---:|
| CUDA Unified | 0.0001 | 4.1323 | 194.9371 |
| CUDA Prefetch | 69.8939 | 4.1163 | 4.2397 |
| CUDA Prefetch + Pages | 91.4547 | 4.1188 | 4.2554 |
| CUDA Prefetch + Pages + Advice | 161.5593 | 4.1191 | 4.2844 |

Prefetching reduces the cost of some first GPU launches but takes preparation time itself. The Unified version has almost no *explicit* setup cost, yet its first launch is about 195 ms here, consistent with on-demand migration and startup effects. The page and advice variants add setup cost. Because CPU page touching, host initialization, and other work are outside the timed setup region, **these figures cannot establish end-to-end performance**. In a one-shot workload, more memory preparation is not necessarily worthwhile; repeated GPU processing can make upfront work easier to amortize.

## 4. Historical SIMD results (Windows Visual Studio Release)

| Input length | C (ms) | x86-64 (ms) | SIMD XMM (ms) | SIMD YMM (ms) |
|---|---:|---:|---:|---:|
| 2^20 | 0.2288 | 0.4212 | 0.2374 | 0.1930 |
| 2^26 | 32.2542 | 35.4736 | 30.7271 | 33.9873 |
| 2^29 | 245.5390 | 264.0704 | 242.5343 | 241.2624 |

The SIMD figures come from the earlier group README and ran on Windows, unlike the GPU figures collected on Colab. They show that explicit SIMD does not always improve on an optimized C compiler: XMM wins at `2^26`, while YMM is slightly faster at `2^29` (by about 0.5%). At `2^20`, scalar x86-64 assembly took 0.4212 ms against Release C at 0.2288 ms, meaning the assembly was approximately 84% longer, not “46% longer.” A compiler-vectorization explanation is plausible, but it has not been verified with compiler diagnostics or disassembly.

## 5. Nsight Systems (`nsys`) and Nsight Compute (`ncu`)

Both tools successfully produced reports for all four CUDA variants in the Colab session, using `2^20` elements and 256 threads/block. The profiler console output is preserved in [`benchmark_and_profiler_console_log.txt`](benchmark_and_profiler_console_log.txt). 

| Mode | `nsys` kernels | Median (µs) | Mean incl. first (µs) |
|---|---:|---:|---:|
| CUDA Unified | 31 | 65.438 | 191.100 |
| CUDA Prefetch | 31 | 66.047 | 65.956 |
| CUDA Prefetch + Pages | 31 | 67.551 | 67.520 |
| CUDA Prefetch + Pages + Advice | 31 | 65.310 | 65.158 |

The Unified summary mean is elevated by the first, migration-heavy kernel launch. The median is less sensitive to this outlier. Each `ncu` run successfully profiled one launch using multiple measurement passes. **The very high `ncu`-instrumented first-launch timings are not normal benchmark timings** and are excluded from speedup calculations. The available console log verifies profiler success but does not itself provide detailed memory-throughput counters; therefore we do not claim profiling proved that the computation is memory-bound.

**Real profiler screenshot slots (still required):**
- `nsys` unified: **[insert actual screenshot of `nsys stats` or Nsight Systems timeline]**
- `ncu` unified: **[insert actual screenshot of metric report / profiler output]**
- `nsys` prefetch: **[insert actual screenshot of `nsys stats` or Nsight Systems timeline]**
- `ncu` prefetch: **[insert actual screenshot of metric report / profiler output]**
- `nsys` pages: **[insert actual screenshot of `nsys stats` or Nsight Systems timeline]**
- `ncu` pages: **[insert actual screenshot of metric report / profiler output]**
- `nsys` advise: **[insert actual screenshot of `nsys stats` or Nsight Systems timeline]**
- `ncu` advise: **[insert actual screenshot of metric report / profiler output]**

## 6. SIMD versus SIMT

In the x86 SIMD implementations, a single CPU instruction processes multiple floating-point values at once: XMM registers can hold four 32-bit floats and YMM registers eight. The program must explicitly handle vector lanes and remaining elements. In CUDA SIMT, the programmer writes one element operation for a thread and launches many threads; each thread computes its start index and advances by the grid stride. The GPU schedules threads in groups called warps, ordinarily 32 threads, and distributes many warps over streaming multiprocessors.

SIMD advantages for this use case include low overhead for arrays already in CPU memory, good performance without device transfers, and relatively simple integration into a CPU application. Limitations include fewer simultaneous lanes and explicit remainder handling. CUDA advantages include high parallel throughput for large independent vectors and a scalable grid-stride loop. Its drawbacks include launch overhead, memory migration/transfer costs, available-device-memory limits, and variable performance in shared Colab environments. With one add and one multiply per element and multiple vector reads/writes, a bandwidth limit is plausible at large sizes, but we do not make a definitive memory-bound classification without bandwidth-counter evidence.

## 7. Problems encountered and solutions

1. **CUDA API version mismatch.** Older examples passed a numeric device ID to memory prefetch/advice calls, whereas the newer Colab CUDA toolkit expected a `cudaMemLocation` structure. Small version-specific wrapper functions fixed compilation while leaving the lecture-style grid-stride kernel unchanged.
2. **Boundary handling.** Grid-stride loop condition `i < n` and 16 guard floats after A were tested using sizes 1, 3, 1003 and 1025. All 20 mode/size test combinations passed in the logged run.
3. **First-launch overhead.** Unified Memory first access often includes substantial migration/startup cost. We measured the first launch separately and averaged the next 30 executions. This helps compare steady-state computation but is not end-to-end timing.
4. **Profiler overhead.** Nsight Compute instruments and can replay launches. We kept profiling results separate from normal performance averages.
5. **Memory management interpretation.** We documented page touching as an experimental assumption; the professor’s intended meaning should be verified, and the measured `setup_ms` does not cover its page-touch loop.
6. **Memory constraints.** The `2^26` experiment succeeded with all correctness checks passing. `2^28` and `2^30` were not reported as executed, and must not be claimed as tested.

## 8. Screenshots of actual notebook outputs

**CPU baseline**

![CPU baseline](images/cpu_output.png)

**CUDA Unified Memory**

![CUDA Unified Memory](images/unified_output.png)

**CUDA Prefetch**

![CUDA Prefetch](images/prefetch_output.png)

**CUDA Prefetch + Pages**

![CUDA Prefetch + Pages](images/pages_output.png)

**CUDA Prefetch + Pages + Advice**

![CUDA Prefetch + Pages + Advice](images/advice_output.png)

**Main comparative output, first page**

![Main comparative output, first page](images/comparison_1.png)

**Main comparative output, second page**

![Main comparative output, second page](images/comparison_2.png)

**Memory check before large benchmark**

![Memory check before large benchmark](images/available_memory.png)

**Expanded benchmark summary including 2^26**

![Expanded benchmark summary including 2^26](images/large_vector_results.png)

**Boundary evidence:** the text log includes all 20 PASS results. Add screenshots of the Section 9 cell showing 1003 and 1025 for SIMD XMM/YMM where relevant, and CUDA variants. **[Actual complete boundary screenshot still to add.]**

## 9. Run / reproduce

**Google Colab link:** **[INSERT shareable Colab notebook URL and verify viewer access]**

## 10. AI usage declaration

We used ChatGPT to help draft and simplify the CUDA implementations, resolve CUDA API-version errors, organize Google Colab experiments, and prepare the benchmark report. Our group executed the programs, gathered timing and profiler output, checked correctness, and reviewed the final submitted work.

