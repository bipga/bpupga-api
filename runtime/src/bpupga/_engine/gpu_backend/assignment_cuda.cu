// Exact double-precision rectangular shortest augmenting path assignment.
// All search, dual updates, tie decisions and augmentations execute on CUDA.
// One persistent block avoids host round trips and inter-block global barriers.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstdint>
#include <limits>

using Index = int64_t;
constexpr int THREADS = 256;

__device__ __forceinline__ bool better(double candidate, Index position,
                                      bool is_free, double best,
                                      Index best_position, bool best_free) {
    if (candidate < best) return true;
    if (candidate != best) return false;
    if (is_free != best_free) return is_free;
    // SciPy JV scans remaining[] in order, replacing an equal minimizer iff
    // the newly visited column is free: last free, otherwise first matched.
    return is_free ? position > best_position : position < best_position;
}

__global__ void assignment_kernel(
    const double* cost, Index n, Index real_m, Index m,
    bool private_dummies, double dummy_cost, double* u, double* v,
    double* distance, Index* row_match, Index* col_match,
    Index* predecessor, Index* remaining, Index* seen_row,
    Index* seen_col, int* status) {
    __shared__ double best_cost[THREADS];
    __shared__ Index best_position[THREADS];
    __shared__ int best_free[THREADS];
    __shared__ Index active_row, remaining_count, sink;
    __shared__ double minimum;
    __shared__ int done, failed;
    const int tid = threadIdx.x;
    if (tid == 0) {
        *status = 0;
        failed = 0;
    }
    __syncthreads();
    for (Index k = tid; k < n * real_m; k += THREADS) {
        const double value = cost[k];
        if (isnan(value) || (isinf(value) && value < 0.0))
            atomicExch(status, 1);
    }
    for (Index i = tid; i < n; i += THREADS) {
        row_match[i] = -1;
        u[i] = 0.0;
    }
    for (Index j = tid; j < m; j += THREADS) {
        col_match[j] = -1;
        v[j] = 0.0;
    }
    __syncthreads();
    if (*status != 0) return;

    for (Index current_row = 0; current_row < n; ++current_row) {
        for (Index i = tid; i < n; i += THREADS) seen_row[i] = 0;
        for (Index j = tid; j < m; j += THREADS) {
            distance[j] = CUDART_INF;
            predecessor[j] = -1;
            seen_col[j] = 0;
            remaining[j] = m - j - 1;
        }
        if (tid == 0) {
            active_row = current_row;
            remaining_count = m;
            minimum = 0.0;
            done = 0;
        }
        __syncthreads();
        while (!done && !failed) {
            if (tid == 0) seen_row[active_row] = 1;
            __syncthreads();
            double local_cost = CUDART_INF;
            Index local_position = INT64_MAX;
            bool local_free = false;
            for (Index position = tid; position < remaining_count;
                 position += THREADS) {
                const Index column = remaining[position];
                const double edge = column < real_m
                    ? cost[active_row * real_m + column]
                    : (column - real_m == active_row ? dummy_cost : CUDART_INF);
                const double reduced = minimum + edge
                                             - u[active_row] - v[column];
                if (reduced < distance[column]) {
                    distance[column] = reduced;
                    predecessor[column] = active_row;
                }
                const double value = distance[column];
                const bool is_free = col_match[column] < 0;
                if (better(value, position, is_free,
                           local_cost, local_position, local_free)) {
                    local_cost = value;
                    local_position = position;
                    local_free = is_free;
                }
            }
            best_cost[tid] = local_cost;
            best_position[tid] = local_position;
            best_free[tid] = local_free;
            __syncthreads();
            for (int stride = THREADS / 2; stride > 0; stride /= 2) {
                if (tid < stride &&
                    better(best_cost[tid + stride], best_position[tid + stride],
                           best_free[tid + stride], best_cost[tid],
                           best_position[tid], best_free[tid])) {
                    best_cost[tid] = best_cost[tid + stride];
                    best_position[tid] = best_position[tid + stride];
                    best_free[tid] = best_free[tid + stride];
                }
                __syncthreads();
            }
            if (tid == 0) {
                minimum = best_cost[0];
                if (!isfinite(minimum)) {
                    failed = 1;
                    *status = 2;
                } else {
                    const Index position = best_position[0];
                    const Index column = remaining[position];
                    seen_col[column] = 1;
                    --remaining_count;
                    remaining[position] = remaining[remaining_count];
                    const Index next_row = col_match[column];
                    if (next_row < 0) {
                        sink = column;
                        done = 1;
                    } else {
                        active_row = next_row;
                    }
                }
            }
            __syncthreads();
        }
        if (failed) return;
        for (Index i = tid; i < n; i += THREADS) {
            if (seen_row[i]) {
                if (i == current_row) u[i] += minimum;
                else u[i] += minimum - distance[row_match[i]];
            }
        }
        for (Index j = tid; j < m; j += THREADS)
            if (seen_col[j]) v[j] -= minimum - distance[j];
        __syncthreads();
        if (tid == 0) {
            while (true) {
                const Index previous_row = predecessor[sink];
                const Index previous_column = row_match[previous_row];
                row_match[previous_row] = sink;
                col_match[sink] = previous_row;
                if (previous_row == current_row) break;
                sink = previous_column;
            }
        }
        __syncthreads();
    }
}

// Ordered greedy has sequential dependence by definition. The ordering is
// sorted on CUDA before launch; this device loop never exports a score or
// makes a host decision. Active masks and the count are also CUDA-resident.
__global__ void ordered_greedy_kernel(
    const Index* order, Index order_size, Index n, Index m,
    const bool* row_active, const bool* col_active, Index budget,
    unsigned char* row_used, unsigned char* col_used,
    Index* rows, Index* cols, Index* count) {
    for (Index i = threadIdx.x; i < n; i += THREADS) row_used[i] = 0;
    for (Index j = threadIdx.x; j < m; j += THREADS) col_used[j] = 0;
    for (Index k = threadIdx.x; k < budget; k += THREADS) rows[k] = cols[k] = -1;
    __syncthreads();
    if (threadIdx.x == 0) {
        Index accepted = 0;
        for (Index k = 0; k < order_size && accepted < budget; ++k) {
            const Index flat = order[k];
            if (flat < 0 || flat >= n * m) continue;
            const Index i = flat / m;
            const Index j = flat % m;
            if (!row_used[i] && !col_used[j] &&
                (!row_active || row_active[i]) &&
                (!col_active || col_active[j])) {
                row_used[i] = col_used[j] = 1;
                rows[accepted] = i;
                cols[accepted] = j;
                ++accepted;
            }
        }
        *count = accepted;
    }
}

extern "C" int bpupga_assignment(
    const double* cost, Index n, Index m, double* u, double* v,
    double* distance, Index* row_match, Index* col_match,
    Index* predecessor, Index* remaining, Index* seen_row,
    Index* seen_col, int* status, void* stream) {
    assignment_kernel<<<1, THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        cost, n, m, m, false, 0.0, u, v, distance, row_match, col_match, predecessor,
        remaining, seen_row, seen_col, status);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int bpupga_assignment_private(
    const double* cost, Index n, Index real_m, double dummy_cost,
    double* u, double* v, double* distance, Index* row_match, Index* col_match,
    Index* predecessor, Index* remaining, Index* seen_row,
    Index* seen_col, int* status, void* stream) {
    assignment_kernel<<<1, THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        cost, n, real_m, real_m + n, true, dummy_cost, u, v, distance,
        row_match, col_match, predecessor, remaining, seen_row, seen_col, status);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int bpupga_ordered_greedy(
    const Index* order, Index order_size, Index n, Index m,
    const bool* row_active, const bool* col_active, Index budget,
    unsigned char* row_used, unsigned char* col_used,
    Index* rows, Index* cols, Index* count, void* stream) {
    ordered_greedy_kernel<<<1, THREADS, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        order, order_size, n, m, row_active, col_active, budget,
        row_used, col_used, rows, cols, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" const char* bpupga_cuda_error(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}
