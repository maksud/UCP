#include <cstdio>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#define MAX_BLOCK_SZ 1024
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

typedef double SumDataType;
// typedef float SumDataType;

#define NORMAL_WORKLOAD 1

#define CUDA_CHECK(call)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
    do                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
    {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  \
        cudaError_t e = call;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          \
        if (e != cudaSuccess)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          \
        {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              \
            printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
            exit(EXIT_FAILURE);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        \
        }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              \
    } while (0)

struct BoundaryMessage
{
    int id;
    SumDataType from;
    SumDataType to;
    SumDataType capacity;
};

__global__ void gpu_sum_scan_blelloch(SumDataType *const d_out, const SumDataType *const d_in, SumDataType *const d_block_sums, const size_t numElems)
{
    extern __shared__ SumDataType s_out[];

    // Zero out shared memory
    // Especially important when padding shmem for
    //  non-power of 2 sized input
    s_out[threadIdx.x] = 0;
    s_out[threadIdx.x + blockDim.x] = 0;

    __syncthreads();

    unsigned long cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (cpy_idx < numElems)
    {
        s_out[threadIdx.x] = d_in[cpy_idx];
        if (cpy_idx + blockDim.x < numElems)
            s_out[threadIdx.x + blockDim.x] = d_in[cpy_idx + blockDim.x];
    }

    __syncthreads();

    // Reduce/Upsweep step

    // 2^11 = 2048, the max amount of data a block can blelloch scan
    unsigned long max_steps = 11;

    unsigned long r_idx = 0;
    unsigned long l_idx = 0;
    SumDataType sum = 0; // global sum can be passed to host if needed
    unsigned long t_active = 0;
    for (long s = 0; s < max_steps; ++s)
    {
        t_active = 0;

        // calculate necessary indexes
        // right index must be (t+1) * 2^(s+1)) - 1
        r_idx = ((threadIdx.x + 1) * (1 << (s + 1))) - 1;
        if (r_idx >= 0 && r_idx < 2048)
            t_active = 1;

        if (t_active)
        {
            // left index must be r_idx - 2^s
            l_idx = r_idx - (1 << s);

            // do the actual add operation
            sum = s_out[l_idx] + s_out[r_idx];
        }
        __syncthreads();

        if (t_active)
            s_out[r_idx] = sum;
        __syncthreads();
    }

    // Copy last element (total sum of block) to block sums array
    // Then, reset last element to operation's identity (sum, 0)
    if (threadIdx.x == 0)
    {
        d_block_sums[blockIdx.x] = s_out[r_idx];
        s_out[r_idx] = 0;
    }

    __syncthreads();

    // Downsweep step

    for (long s = max_steps - 1; s >= 0; --s)
    {
        // calculate necessary indexes
        // right index must be (t+1) * 2^(s+1)) - 1
        r_idx = ((threadIdx.x + 1) * (1 << (s + 1))) - 1;
        if (r_idx >= 0 && r_idx < 2048)
        {
            t_active = 1;
        }

        SumDataType r_cpy = 0;
        SumDataType lr_sum = 0;
        if (t_active)
        {
            // left index must be r_idx - 2^s
            l_idx = r_idx - (1 << s);

            // do the downsweep operation
            r_cpy = s_out[r_idx];
            lr_sum = s_out[l_idx] + s_out[r_idx];
        }
        __syncthreads();

        if (t_active)
        {
            s_out[l_idx] = r_cpy;
            s_out[r_idx] = lr_sum;
        }
        __syncthreads();
    }

    // Copy the results to global memory
    if (cpy_idx < numElems)
    {
        d_out[cpy_idx] = s_out[threadIdx.x];
        if (cpy_idx + blockDim.x < numElems)
            d_out[cpy_idx + blockDim.x] = s_out[threadIdx.x + blockDim.x];
    }
}

__global__ void gpu_add_block_sums(SumDataType *const d_out, const SumDataType *const d_in, SumDataType *const d_block_sums, const size_t numElems)
{
    // unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
    SumDataType d_block_sum_val = d_block_sums[blockIdx.x];

    // Simple implementation's performance is not significantly (if at all)
    //  better than previous verbose implementation
    unsigned long cpy_idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    if (cpy_idx < numElems)
    {
        d_out[cpy_idx] = d_in[cpy_idx] + d_block_sum_val;
        if (cpy_idx + blockDim.x < numElems)
            d_out[cpy_idx + blockDim.x] = d_in[cpy_idx + blockDim.x] + d_block_sum_val;
    }
}

// Modified version of Mark Harris' implementation of the Blelloch scan
//  according to https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf
__global__ void gpu_prescan(SumDataType *const d_out, const SumDataType *const d_in, SumDataType *const d_block_sums, const unsigned long len, const unsigned long shmem_sz, const unsigned long max_elems_per_block)
{
    // Allocated on invocation
    extern __shared__ SumDataType s_out[];

    long thid = threadIdx.x;
    long ai = thid;
    long bi = thid + blockDim.x;

    // Zero out the shared memory
    // Helpful especially when input size is not power of two
    s_out[thid] = 0;
    s_out[thid + blockDim.x] = 0;
    // If CONFLICT_FREE_OFFSET is used, shared memory
    //  must be a few more than 2 * blockDim.x
    if (thid + max_elems_per_block < shmem_sz)
        s_out[thid + max_elems_per_block] = 0;

    __syncthreads();

    // Copy d_in to shared memory
    // Note that d_in's elements are scattered into shared memory
    //  in light of avoiding bank conflicts
    unsigned long cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
    if (cpy_idx < len)
    {
        s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
        if (cpy_idx + blockDim.x < len)
            s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
    }

    // For both upsweep and downsweep:
    // Sequential indices with conflict free padding
    //  Amount of padding = target index / num banks
    //  This "shifts" the target indices by one every multiple
    //   of the num banks
    // offset controls the stride and starting index of
    //  target elems at every iteration
    // d just controls which threads are active
    // Sweeps are pivoted on the last element of shared memory

    // Upsweep/Reduce step
    long offset = 1;
    for (long d = max_elems_per_block >> 1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)
        {
            long ai = offset * ((thid << 1) + 1) - 1;
            long bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_out[bi] += s_out[ai];
        }
        offset <<= 1;
    }

    // Save the total sum on the global block sums array
    // Then clear the last element on the shared memory
    if (thid == 0)
    {
        d_block_sums[blockIdx.x] = s_out[max_elems_per_block - 1 + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
        s_out[max_elems_per_block - 1 + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
    }

    // Downsweep step
    for (long d = 1; d < max_elems_per_block; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();

        if (thid < d)
        {
            long ai = offset * ((thid << 1) + 1) - 1;
            long bi = offset * ((thid << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            SumDataType temp = s_out[ai];
            s_out[ai] = s_out[bi];
            s_out[bi] += temp;
        }
    }
    __syncthreads();

    // Copy contents of shared memory to global memory
    if (cpy_idx < len)
    {
        d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
        if (cpy_idx + blockDim.x < len)
            d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
    }
}

void sum_scan_blelloch(SumDataType *d_out, SumDataType *d_in, const size_t numElems)
{
    // Zero out d_out
    CUDA_CHECK(cudaMemset(d_out, 0, numElems * sizeof(SumDataType)));

    // Set up number of threads and blocks

    unsigned long block_sz = MAX_BLOCK_SZ / 2;
    unsigned long max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

    // If input size is not power of two, the remainder will still need a whole block
    // Thus, number of blocks must be the ceiling of input size / max elems that a block can handle
    // unsigned int grid_sz = (unsigned int) std::ceil((double) numElems / (double) max_elems_per_block);
    // UPDATE: Instead of using ceiling and risking miscalculation due to precision, just automatically
    //  add 1 to the grid size when the input size cannot be divided cleanly by the block's capacity
    unsigned long grid_sz = numElems / max_elems_per_block;
    // Take advantage of the fact that integer division drops the decimals
    if (numElems % max_elems_per_block != 0)
        grid_sz += 1;

    // Conflict free padding requires that shared memory be more than 2 * block_sz
    unsigned long shmem_sz = max_elems_per_block + ((max_elems_per_block - 1) >> LOG_NUM_BANKS);

    // Allocate memory for array of total sums produced by each block
    // Array length must be the same as number of blocks
    SumDataType *d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, sizeof(SumDataType) * grid_sz));
    CUDA_CHECK(cudaMemset(d_block_sums, 0, sizeof(SumDataType) * grid_sz));

    // Sum scan data allocated to each block
    gpu_prescan<<<grid_sz, block_sz, sizeof(SumDataType) * shmem_sz>>>(d_out, d_in, d_block_sums, numElems, shmem_sz, max_elems_per_block);

    // Sum scan total sums produced by each block
    // Use basic implementation if number of total sums is <= 2 * block_sz
    //  (This requires only one block to do the scan)
    if (grid_sz <= max_elems_per_block)
    {
        SumDataType *d_dummy_blocks_sums;
        CUDA_CHECK(cudaMalloc(&d_dummy_blocks_sums, sizeof(SumDataType)));
        CUDA_CHECK(cudaMemset(d_dummy_blocks_sums, 0, sizeof(SumDataType)));
        gpu_prescan<<<1, block_sz, sizeof(SumDataType) * shmem_sz>>>(d_block_sums, d_block_sums, d_dummy_blocks_sums, grid_sz, shmem_sz, max_elems_per_block);
        CUDA_CHECK(cudaFree(d_dummy_blocks_sums));
    }
    // Else, recurse on this same function as you'll need the full-blown scan
    //  for the block sums
    else
    {
        SumDataType *d_in_block_sums;
        CUDA_CHECK(cudaMalloc(&d_in_block_sums, sizeof(SumDataType) * grid_sz));
        CUDA_CHECK(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(SumDataType) * grid_sz, cudaMemcpyDeviceToDevice));
        sum_scan_blelloch(d_block_sums, d_in_block_sums, grid_sz);
        CUDA_CHECK(cudaFree(d_in_block_sums));
    }

    // Add each block's total sum to its scan output
    // in order to get the final, global scanned array
    gpu_add_block_sums<<<grid_sz, block_sz>>>(d_out, d_out, d_block_sums, numElems);

    CUDA_CHECK(cudaFree(d_block_sums));
}

__global__ void calcCost(SumDataType *d_in, int N)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int BLOCK_GRID = gridDim.x * blockDim.x;
    for (int m = idx; m < N; m += BLOCK_GRID)
    {
        // d_in[m] = d_in[m] * (d_in[m] - 1);
        d_in[m] = NORMAL_WORKLOAD;
    }
}

__global__ void checkUCP(SumDataType *d_out, SumDataType Capacity, int N, int P, BoundaryMessage *startBoundary, BoundaryMessage *endBoundary)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int BLOCK_GRID = gridDim.x * blockDim.x;

    for (int m = idx + 1; m < N; m += BLOCK_GRID)
    {
        int pMm1 = (d_out[m - 1] - 0.00001) / Capacity;
        int pM = (d_out[m] - 0.00001) / Capacity;

        if (pMm1 != pM)
        {
            if (pMm1 < P - 1)
            {

                endBoundary[pMm1].id = m;
                endBoundary[pMm1].from = 0.0;
                endBoundary[pMm1].to = (pMm1 + 1) * Capacity - d_out[m - 1];
                endBoundary[pMm1].capacity = d_out[m] - d_out[m - 1];
            }

            for (int k = pMm1 + 1; k < pM; k++)
            {
                endBoundary[pMm1].id = m;
                endBoundary[pMm1].from = k * Capacity - d_out[m - 1];
                endBoundary[pMm1].to = (k + 1) * Capacity - d_out[m - 1];
                endBoundary[pMm1].capacity = d_out[m] - d_out[m - 1];
            }

            if (pM < P)
            {
                startBoundary[pM].id = m;
                startBoundary[pM].from = pM * Capacity - d_out[m - 1];
                startBoundary[pM].to = d_out[m] - d_out[m - 1];
                startBoundary[pM].capacity = d_out[m] - d_out[m - 1];
            }
        }
    }
}

int main(int argc, char *argv[])
{
    // Default values
    int N = 10000000;
    int P = 8;

    if (argc > 1)
    {
        N = std::atoi(argv[1]); // Convert first argument
    }

    if (argc > 2)
    {
        P = std::atoi(argv[2]); // Convert second argument
    }

    printf("N = %d, P = %d\n", N, P);

    // 1. Create events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    SumDataType *num = new SumDataType[N + 1];
    SumDataType *res = new SumDataType[N + 1];

    for (int i = 0; i < N; i++)
    {
        num[i] = 100;
        res[i] = 0;
    }
    num[N] = 0;
    res[N] = 0;

    SumDataType *d_input = nullptr;
    SumDataType *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, (N + 1) * sizeof(SumDataType)));
    CUDA_CHECK(cudaMalloc(&d_output, (N + 1) * sizeof(SumDataType)));

    CUDA_CHECK(cudaMemcpy(d_input, num, (N + 1) * sizeof(SumDataType), cudaMemcpyHostToDevice));

    // 2. Record the start event
    cudaEventRecord(start);

    calcCost<<<5120, 512>>>(d_input, N);

    // Device pointers
    // SumDataType *d_input = nullptr;
    // SumDataType *d_output = nullptr;

    // cudaMalloc(&d_input, num_items * sizeof(SumDataType));
    // cudaMalloc(&d_output, num_items * sizeof(SumDataType));

#if 0
    // Determine temp storage size for CUB inclusive scan
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, d_input, d_output, N + 1);
    cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, d_input, d_output, N + 1);
    printf("Temp storage size: %zu bytes\n", temp_storage_bytes);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, N + 1);
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, N + 1);
#else
    sum_scan_blelloch(d_output, d_input, (N + 1));
#endif

    // cudaDeviceSynchronize();
    // 4. Record the stop event
    cudaEventRecord(stop);
    // 5. Wait for the event to complete
    cudaEventSynchronize(stop);

    // 6. Calculate elapsed time
    float milliseconds1 = 0;
    cudaEventElapsedTime(&milliseconds1, start, stop);
    printf("Kernel 1 execution time: %.3f ms\n", milliseconds1);

    CUDA_CHECK(cudaMemcpy(res, d_output, (N + 1) * sizeof(SumDataType), cudaMemcpyDeviceToHost));

    // for(int i=N-20; i<=N; i++){
    //     printf("%d=%f\n", i, res[i]);
    // }

    SumDataType Capacity = res[N] / P;

    BoundaryMessage *boundaries = new BoundaryMessage[2 * P];
    for (int i = 0; i < 2 * P; i++)
    {
        boundaries[i].id = -1;
        boundaries[i].from = boundaries[i].to = boundaries[i].capacity = -2;
    }

    BoundaryMessage *d_boundaries;
    CUDA_CHECK(cudaMalloc(&d_boundaries, (2 * P) * sizeof(BoundaryMessage)));
    CUDA_CHECK(cudaMemcpy(d_boundaries, boundaries, (2 * P) * sizeof(BoundaryMessage), cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    checkUCP<<<512, 512>>>(d_output, Capacity, N + 1, P, d_boundaries, d_boundaries + P);
    cudaDeviceSynchronize();
    // 4. Record the stop event
    cudaEventRecord(stop);
    // 5. Wait for the event to complete
    cudaEventSynchronize(stop);

    // 6. Calculate elapsed time
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start, stop);
    printf("Kernel 2 execution time: %.3f ms\n", milliseconds2);

    printf("Total execution time: %.5f ms\n", milliseconds1 + milliseconds2);

    CUDA_CHECK(cudaMemcpy(boundaries, d_boundaries, (2 * P) * sizeof(BoundaryMessage), cudaMemcpyDeviceToHost));

    BoundaryMessage *startBoundary = boundaries;
    BoundaryMessage *endBoundary = boundaries + P;

    startBoundary[0].id = 0;
    startBoundary[0].from = 0;
    startBoundary[0].to = res[1] - res[0];
    startBoundary[0].capacity = res[1] - res[0];

    endBoundary[P - 1].id = N;
    endBoundary[P - 1].from = 0;
    endBoundary[P - 1].to = res[N] - res[N - 1];
    endBoundary[P - 1].capacity = res[N] - res[N - 1];

    if (startBoundary[0].id == endBoundary[0].id)
        startBoundary[0].id = -1;

    for (int rank = 0; rank < P; rank++)
    {

        if (rank >= 0)
        {
            int x = startBoundary[rank].id;

            if (x >= 0 && x < N)
            {

                SumDataType eStart = startBoundary[rank].from;
                SumDataType eEnd = startBoundary[rank].to;

                // totalWorkLoad += eEnd - eStart;

                std::cout << "#1. Rank " << rank << " Task " << x << " SubTask: eStart " << eStart << " eEnd " << eEnd << std::endl;
            }
        }

        // Part 2:
        if (startBoundary[rank].id >= 0)
        {
            int ll = startBoundary[rank].id + 1;
            int hl = endBoundary[rank].id - 1;

            for (int i = ll; i <= hl; i++)
            {
                // int c = NORMAL_WORKLOAD;
                // if (i == idx_imbalanced)
                //     c = IMBALANCED_WORKLOAD;
                // totalWorkLoad += c;
            }

            std::cout << "#2. Rank " << rank << " Task " << ll << " to " << hl << std::endl;
        }

        // Part 3:
        // Process End Portions
        if (rank <= P - 1)
        {
            int x = endBoundary[rank].id;

            if (x >= 0 && x < N)
            {

                SumDataType eStart = endBoundary[rank].from;
                SumDataType eEnd = endBoundary[rank].to;

                // totalWorkLoad += eEnd - eStart;

                std::cout << "#3. Rank " << rank << " Task " << x << " SubTask: eStart " << eStart << " eEnd " << eEnd << std::endl;
            }
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
