#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>

// Constants used in all five versions.
#define RUNS 30
#define GUARD 16
#define GUARD_VALUE -12345.0f

// Colab may have CUDA 13, while the slides show older CUDA signatures.
// These two functions keep the lecture's behavior in either CUDA 12 or 13.
cudaError_t prefetch(float *array, size_t bytes, int device) {
#if CUDART_VERSION >= 13000
    cudaMemLocation location = {};
    location.type = cudaMemLocationTypeDevice;
    location.id = device;
    return cudaMemPrefetchAsync(array, bytes, location, 0, 0);
#else
    return cudaMemPrefetchAsync(array, bytes, device, 0);
#endif
}

cudaError_t memoryAdvice(float *array, size_t bytes, cudaMemoryAdvise advice, int device) {
#if CUDART_VERSION >= 13000
    cudaMemLocation location = {};
    location.type = cudaMemLocationTypeDevice;
    location.id = device;
    return cudaMemAdvise(array, bytes, advice, location);
#else
    return cudaMemAdvise(array, bytes, advice, device);
#endif
}

// This is the same grid-stride loop taught in ELec 2.
__global__ void vectorTriadCUDA(int n, float *A, float *B, float *C, float *D) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) {
        A[i] = B[i] + C[i] * D[i];
    }
}

// C baseline: same mathematical operation, executed by the CPU.
// noinline makes CPU timing less susceptible to removal/inlining by the compiler.
__attribute__((noinline))
void vectorTriadCPU(int n, float *A, float *B, float *C, float *D) {
    for (int i = 0; i < n; i++) A[i] = B[i] + C[i] * D[i];
}

// Print errors rather than continuing with invalid GPU data.
void check(cudaError_t error, const char *step) {
    if (error != cudaSuccess) {
        printf("CUDA error during %s: %s\n", step, cudaGetErrorString(error));
        exit(1);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: ./triad_lecture MODE N BLOCK_SIZE\n");
        return 1;
    }
    char *mode = argv[1];
    int n = atoi(argv[2]);
    int blockSize = atoi(argv[3]);
    if (n <= 0 || blockSize <= 0 || blockSize > 1024) return 1;
    int cpu = strcmp(mode, "cpu") == 0;
    int unified = strcmp(mode, "unified") == 0;
    int withPrefetch = strcmp(mode, "prefetch") == 0;
    int withPages = strcmp(mode, "pages") == 0;
    int withAdvice = strcmp(mode, "advise") == 0;
    if (!(cpu || unified || withPrefetch || withPages || withAdvice)) return 1;

    size_t bytes = (size_t)n * sizeof(float);
    size_t outputBytes = (size_t)(n + GUARD) * sizeof(float);
    float *A = NULL, *B = NULL, *C = NULL, *D = NULL;
    if (cpu) {
        A = (float*)malloc(outputBytes);
        B = (float*)malloc(bytes);
        C = (float*)malloc(bytes);
        D = (float*)malloc(bytes);
        if (!A || !B || !C || !D) { printf("CPU allocation failed\n"); return 1; }
    } else {
        check(cudaMallocManaged(&A, outputBytes), "allocate A");
        check(cudaMallocManaged(&B, bytes), "allocate B");
        check(cudaMallocManaged(&C, bytes), "allocate C");
        check(cudaMallocManaged(&D, bytes), "allocate D");
    }

    // Preserve initialization from the group's SIMD project.
    for (int i = 0; i < n; i++) {
        B[i] = (float)sin(i * 0.001);
        C[i] = (float)cos(i * 0.002);
        D[i] = (float)(sin(i * 0.0005) + 1.5);
    }
    for (int i = n; i < n + GUARD; i++) A[i] = GUARD_VALUE;

    // Page-creation experiment: touch one float per 4096-byte output page on CPU.
    // This is an experimental interpretation, not a separate CUDA allocation API.
    if (withPages || withAdvice) {
        for (int i = 0; i < n; i += 1024) A[i] = 0.0f;
        A[n - 1] = 0.0f;
    }

    int device = 0;
    double setupMs = 0;
    if (!cpu) check(cudaGetDevice(&device), "get GPU");
    auto setupStart = std::chrono::steady_clock::now();
    if (withAdvice) {
        check(memoryAdvice(B, bytes, cudaMemAdviseSetReadMostly, device), "advise B");
        check(memoryAdvice(C, bytes, cudaMemAdviseSetReadMostly, device), "advise C");
        check(memoryAdvice(D, bytes, cudaMemAdviseSetReadMostly, device), "advise D");
        check(memoryAdvice(A, outputBytes, cudaMemAdviseSetPreferredLocation, device), "advise A");
    }
    if (withPrefetch || withPages || withAdvice) {
        check(prefetch(B, bytes, device), "prefetch B");
        check(prefetch(C, bytes, device), "prefetch C");
        check(prefetch(D, bytes, device), "prefetch D");
        check(prefetch(A, outputBytes, device), "prefetch A");
        check(cudaDeviceSynchronize(), "finish prefetch");
    }
    auto setupEnd = std::chrono::steady_clock::now();
    setupMs = std::chrono::duration<double, std::milli>(setupEnd - setupStart).count();

    int numBlocks = (n + blockSize - 1) / blockSize;
    if (numBlocks > 65535) numBlocks = 65535; // grid-stride loop handles the rest.
    double total = 0, minimum = 1e100, maximum = 0, first = 0;

    if (cpu) {
        for (int run = 0; run <= RUNS; run++) {
            auto start = std::chrono::steady_clock::now();
            vectorTriadCPU(n, A, B, C, D);
            auto stop = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(stop - start).count();
            if (run == 0) first = ms; // warmup, not included in average
            else {
                total += ms;
                if (ms < minimum) minimum = ms;
                if (ms > maximum) maximum = ms;
            }
        }
    } else {
        cudaEvent_t start, stop;
        check(cudaEventCreate(&start), "create start event");
        check(cudaEventCreate(&stop), "create stop event");
        for (int run = 0; run <= RUNS; run++) {
            check(cudaEventRecord(start), "start timer");
            vectorTriadCUDA<<<numBlocks, blockSize>>>(n, A, B, C, D);
            check(cudaGetLastError(), "launch kernel");
            check(cudaEventRecord(stop), "stop timer");
            check(cudaEventSynchronize(stop), "finish kernel");
            float ms = 0;
            check(cudaEventElapsedTime(&ms, start, stop), "read time");
            if (run == 0) first = ms;
            else {
                total += ms;
                if (ms < minimum) minimum = ms;
                if (ms > maximum) maximum = ms;
            }
        }
        check(cudaEventDestroy(start), "free event");
        check(cudaEventDestroy(stop), "free event");
    }

    // Check results against the CPU mathematical expression using a tolerance.
    int errors = 0;
    double maxDifference = 0;
    for (int i = 0; i < n; i++) {
        float expected = B[i] + C[i] * D[i];
        double difference = fabs((double)A[i] - expected);
        double tolerance = 1e-5 * fmax(1.0, fabs((double)expected));
        if (!(difference <= tolerance)) errors++;
        if (difference > maxDifference) maxDifference = difference;
    }
    int boundaryOK = 1;
    for (int i = n; i < n + GUARD; i++) {
        if (A[i] != GUARD_VALUE) boundaryOK = 0;
    }
    printf("MODE=%s N=%d BLOCK=%d RUNS=%d\n", mode, n, blockSize, RUNS);
    printf("First: %.6f ms | Average: %.6f ms | Min: %.6f ms | Max: %.6f ms\n", first, total/RUNS, minimum, maximum);
    printf("Memory setup: %.6f ms (not included in kernel time)\n", setupMs);
    printf("Correctness: %s | Errors: %d | Maximum difference: %.9g\n", errors == 0 ? "PASS" : "FAIL", errors, maxDifference);
    printf("Boundary check: %s\n", boundaryOK ? "PASS" : "FAIL");
    int shown = n < 5 ? n : 5;
    printf("First %d A: ", shown);
    for (int i = 0; i < shown; i++) printf("%.5f ", A[i]);
    printf("\nLast %d A: ", shown);
    for (int i = n - shown; i < n; i++) printf("%.5f ", A[i]);
    printf("\n");
    printf("RESULT,%s,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%d,%.9g,%s\n", mode, n, blockSize, first, total/RUNS, minimum, maximum, setupMs, errors, maxDifference, boundaryOK ? "PASS" : "FAIL");
    if (cpu) { free(A); free(B); free(C); free(D); }
    else {
        check(cudaFree(A), "free A"); check(cudaFree(B), "free B");
        check(cudaFree(C), "free C"); check(cudaFree(D), "free D");
    }
    return (errors == 0 && boundaryOK) ? 0 : 1;
}
