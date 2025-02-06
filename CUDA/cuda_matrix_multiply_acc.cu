/* CUDA 实现矩阵的乘法，如何做加速？

编译指令：
nvcc cuda_matrix_multiply_acc.cu -o cuda_matrix_multiply_acc -arch=sm_86
*/
#include <random>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#define M 3
#define K 4
#define N 2

#define fixed_seed 100

#define TILE 16
#define checkCudaErrors(val)                                                                             \
    {                                                                                                    \
        cudaError_t err = (val);                                                                         \
        if (err != cudaSuccess)                                                                          \
        {                                                                                                \
            fprintf(stderr, "[CUDA ERROR]: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(err);                                                                                   \
        }                                                                                                \
    }

// 单线程处理C矩阵（沿着M和N两个维度）中元素个数： eg.  4×4.
#define TM 4
#define TN 4

/// @brief 生成矩阵A： m * k
float **genRanArrA()
{
    // float arr[M][K];  在栈上分配内存可能会溢出
    float **arr = new float *[M];
    for (int i = 0; i < M; ++i)
    {
        arr[i] = new float[K];
    }

    // 每次执行会生成不同的随机种子
    // std::random_device rd;
    // const std::uint32_tfixed_seed = rd(); // 这里用 rd() 生成的种子

    std::mt19937 gen(fixed_seed); // Mersenne Twister 随机数引擎
    std::uniform_real_distribution<> dis(0.0F, 1.0F);

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            arr[i][j] = dis(gen);
        }
    }

    return arr;
}

/// @brief 生成矩阵B： k * n
float **genRanArrB()
{
    float **arr = new float *[K];
    for (int i = 0; i < K; ++i)
    {
        arr[i] = new float[N];
    }

    std::mt19937 gen(fixed_seed);
    std::uniform_real_distribution<> dis(0.0F, 1.0F);

    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            arr[i][j] = dis(gen);
        }
    }

    return arr;
}

/// @brief 打印矩阵辅助函数
void printArr(float **arr, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << arr[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

/// @brief 删除动态分配的数组辅助函数
void delArr(float **arr, int rows)
{
    for (int i = 0; i < rows; ++i)
    {
        delete[] arr[i];
    }
    delete[] arr;
}

/// @brief 方法1：使用CPU计算
float **matMulCPU(float **A, float **B)
{
    float **C = new float *[M];
    for (int m = 0; m < M; ++m)
    {
        C[m] = new float[N];
    }

    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0F;
            for (int k = 0; k < K; ++k)
            {
                sum += A[m][k] * B[k][n];
            }
            C[m][n] = sum;
        }
    }
    return C;
}

/// @brief CPU矩阵拷贝的GPU上
float *matH2D(float **h_mat, int rows, int cols)
{
    float *d_mat;
    float *h_mat_flat = (float *)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            h_mat_flat[i * cols + j] = h_mat[i][j];
        }
    }

    checkCudaErrors(cudaMalloc((void **)&d_mat, rows * cols * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_mat, h_mat_flat, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

    free(h_mat_flat);
    return d_mat;
}

/// @brief CUDA Kernel：打印矩阵核函数
__global__ void printMatInKernel(float *d_mat, int rows, int cols)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < rows * cols)
    {
        printf("[CUDA Kernel Print Infos] Element at index %d: %f\n", index, d_mat[index]);
    }
}

/// @brief CUDA Kernel：打印矩阵辅助函数
void printMatLaunch(float *d_mat, int rows, int cols)
{
    dim3 dimBlocks(TILE);
    dim3 dimGrids(((rows * cols) + dimBlocks.x - 1) / dimBlocks.x);
    printMatInKernel<<<dimGrids, dimBlocks>>>(d_mat, rows, cols);
}

/// @brief 方法2： 使用GPU加速基础版 - 一个线程只负责C矩阵中的一个元素
__global__ void matMulGPUKernel(float *A, float *B, float *C)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y; // 行
    int n = blockIdx.x * blockDim.x + threadIdx.x; // 列

    if (m < M && n < N)
    {
        float value = 0.0F;
        for (int k = 0; k < K; ++k)
        {
            value += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = value;
    }
}

float *matMulGPULaunch(float *A, float *B, cudaEvent_t &start, cudaEvent_t &stop, float &time_cost)
{
    float *C;
    cudaError_t err;

    checkCudaErrors(cudaMalloc((void **)&C, M * N * sizeof(float)));
    dim3 dimBlocks(TILE, TILE);
    dim3 dimGrids((N + dimBlocks.x - 1) / dimBlocks.x, (M + dimBlocks.y - 1) / dimBlocks.y);

    cudaEventRecord(start);
    matMulGPUKernel<<<dimGrids, dimBlocks>>>(A, B, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_cost, start, stop);
    std::cout << "[Time Cost] : "
              << "使用GPU加速基础版 Kernel Time = " << time_cost << "[ms]." << std::endl;

    // Check for kernel launch errors
    err = cudaGetLastError(); // Check for errors that occurred during kernel launch
    if (err != cudaSuccess)
    {
        std::cerr << "[Kernel launch failed]: " << cudaGetErrorString(err) << std::endl;
        cudaFree(C);
        return nullptr;
    }

    // Synchronize to catch kernel execution errors
    err = cudaDeviceSynchronize(); // Ensures all previous CUDA calls have completed
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA synchronization failed]: " << cudaGetErrorString(err) << std::endl;
        cudaFree(C);
        return nullptr;
    }

    return C;
};

/// @brief 方法3： 使用GPU加速基础版2 - 一个线程负责C矩阵中的多个元素(TM*TN)
__global__ void matMulGPU2Kernel(float *A, float *B, float *C)
{
    int m = TM * blockIdx.y * blockDim.y + threadIdx.y; // 行
    int n = TN * blockIdx.x * blockDim.x + threadIdx.x; // 列
    float tmp[TM][TN] = {0.0F};

    for (int i = 0; i < TM; ++i)
    {
        for (int j = 0; j < TN; ++j)
        {
            if (m + i < M && n + j < N)
            {
                for (int k = 0; k < K; ++k)
                {
                    tmp[i][j] += A[(m + i) * K + k] * B[k * N + n + j];
                }
            }
        }
    }

    for (int i = 0; i < TM; ++i)
    {
        for (int j = 0; j < TN; ++j)
        {
            if (m + i < M && n + j < N)
            {
                C[(m + i) * N + n + j] = tmp[i][j];
            }
        }
    }
}

float *matMulGPU2Launch(float *A, float *B, cudaEvent_t &start, cudaEvent_t &stop, float &time_cost)
{
    float *C;
    cudaError_t err;

    checkCudaErrors(cudaMalloc((void **)&C, M * N * sizeof(float)));
    dim3 dimBlocks(TILE, TILE);
    dim3 dimGrids((N + dimBlocks.x - 1) / dimBlocks.x, (M + dimBlocks.y - 1) / dimBlocks.y);

    cudaEventRecord(start);
    matMulGPU2Kernel<<<dimGrids, dimBlocks>>>(A, B, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_cost, start, stop);
    std::cout << "[Time Cost] : "
              << "使用GPU加速基础版2 Kernel Time = " << time_cost << "[ms]." << std::endl;

    // Check for kernel launch errors
    err = cudaGetLastError(); // Check for errors that occurred during kernel launch
    if (err != cudaSuccess)
    {
        std::cerr << "[Kernel launch failed]: " << cudaGetErrorString(err) << std::endl;
        cudaFree(C);
        return nullptr;
    }

    // Synchronize to catch kernel execution errors
    err = cudaDeviceSynchronize(); // Ensures all previous CUDA calls have completed
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA synchronization failed]: " << cudaGetErrorString(err) << std::endl;
        cudaFree(C);
        return nullptr;
    }

    return C;
};

/// @brief 方法4：使用GPU加速升级版 - shared memory
__global__ void matMulGPUSMKernel(float *A, float *B, float *C)
{
    // 计算当前线程负责的结果矩阵 C 的行和列
    int m = blockIdx.y * TILE + threadIdx.y; // 行
    int n = blockIdx.x * TILE + threadIdx.x; // 列

    __shared__ float shA[TILE][TILE];
    __shared__ float shB[TILE][TILE];

    float sum = 0.0F;
    // 遍历所有需要的块
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
    {
        // 将矩阵 A 和 B 的块加载到共享内存中
        if (m < M && t * TILE + threadIdx.x < K)
            shA[threadIdx.y][threadIdx.x] = A[m * K + t * TILE + threadIdx.x];
        else
            shA[threadIdx.y][threadIdx.x] = 0;
        if (t * TILE + threadIdx.y < K && n < N)
            shB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + n];
        else
            shB[threadIdx.y][threadIdx.x] = 0;

        // 同步线程，确保共享内存中的数据已经加载完毕
        __syncthreads();

        // 计算线程在当前分块的乘加和
        for (int k = 0; k < TILE; ++k)
        {
            sum += shA[threadIdx.y][k] * shB[k][threadIdx.x];
        }

        // 同步线程，确保所有线程都完成了计算
        __syncthreads();
    }

    // 将结果写入全局内存
    if (m < M && n < N)
    {
        C[m * N + n] = sum;
    }
}

float *matMulGPUSMLaunch(float *A, float *B, cudaEvent_t &start, cudaEvent_t &stop, float &time_cost)
{
    cudaError_t err;

    float *C;
    checkCudaErrors(cudaMalloc((void **)&C, M * N * sizeof(float)));
    dim3 dimBlocks(TILE, TILE);
    dim3 dimGrids((N + dimBlocks.x - 1) / dimBlocks.x, (M + dimBlocks.y - 1) / dimBlocks.y);

    cudaEventRecord(start);
    matMulGPUSMKernel<<<dimGrids, dimBlocks>>>(A, B, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_cost, start, stop);
    std::cout << "[Time Cost] : "
              << "使用GPU加速 - Shared Memory Kernel Time = " << time_cost << "[ms]." << std::endl;
    cudaDeviceSynchronize();

    // Check for kernel launch errors
    err = cudaGetLastError(); // Check for errors that occurred during kernel launch
    if (err != cudaSuccess)
    {
        std::cerr << "[Kernel launch failed]: " << cudaGetErrorString(err) << std::endl;
        cudaFree(C);
        return nullptr;
    }

    // Synchronize to catch kernel execution errors
    err = cudaDeviceSynchronize(); // Ensures all previous CUDA calls have completed
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA synchronization failed]: " << cudaGetErrorString(err) << std::endl;
        cudaFree(C);
        return nullptr;
    }

    return C;
};

/// @brief 方法5：使用GPU加速升级版 - shared memory
__global__ void matMulGPUSM2Kernel(float *A, float *B, float *C)
{
    // 计算当前线程负责的矩阵 C 的行和列 TM*TN
    int m = TM * (blockIdx.y * TILE + threadIdx.y); // 行
    int n = TN * (blockIdx.x * TILE + threadIdx.x); // 列

    __shared__ float shA[TILE][TILE];
    __shared__ float shB[TILE][TILE];

    float tmp[TILE][TILE] = {0.0F};
    float sum = 0.0F;
    // 遍历所有需要的块
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
    {
        // 将矩阵 A 和 B 的块加载到共享内存中
        if (m < M && t * TILE + threadIdx.x < K)
            shA[threadIdx.y][threadIdx.x] = A[m * K + t * TILE + threadIdx.x];
        else
            shA[threadIdx.y][threadIdx.x] = 0;
        if (t * TILE + threadIdx.y < K && n < N)
            shB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + n];
        else
            shB[threadIdx.y][threadIdx.x] = 0;

        // 同步线程，确保共享内存中的数据已经加载完毕
        __syncthreads();

        // 计算线程在当前分块的乘加和
        for (int i = 0; i < TM; i++)
        {
            for (int j = 0; j < TN; j++)
            {
                int reg_c_m = threadIdx.y * TM + i;
                int reg_c_n = threadIdx.x * TN + j;
                for (int k = 0; k < TILE; k++)
                {
                    // tmp[i * TN + j] += shA[reg_c_m * TILE + k] * shB[k * BN + reg_c_n];
                    tmp[i]
                }
            }
        }

        // 同步线程，确保所有线程都完成了计算
        __syncthreads();
    }

    // 将结果写入全局内存
    if (m < M && n < N)
    {
        C[m * N + n] = sum;
    }
}

float *matMulGPUSM2Launch(float *A, float *B, cudaEvent_t &start, cudaEvent_t &stop, float &time_cost)
{
    cudaError_t err;

    float *C;
    checkCudaErrors(cudaMalloc((void **)&C, M * N * sizeof(float)));
    dim3 dimBlocks(TILE, TILE);
    dim3 dimGrids((N + dimBlocks.x - 1) / dimBlocks.x, (M + dimBlocks.y - 1) / dimBlocks.y);

    cudaEventRecord(start);
    matMulGPUSM2Kernel<<<dimGrids, dimBlocks>>>(A, B, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_cost, start, stop);
    std::cout << "[Time Cost] : "
              << "使用GPU加速 - Shared Memory Kernel Time = " << time_cost << "[ms]." << std::endl;
    cudaDeviceSynchronize();

    // Check for kernel launch errors
    err = cudaGetLastError(); // Check for errors that occurred during kernel launch
    if (err != cudaSuccess)
    {
        std::cerr << "[Kernel launch failed]: " << cudaGetErrorString(err) << std::endl;
        cudaFree(C);
        return nullptr;
    }

    // Synchronize to catch kernel execution errors
    err = cudaDeviceSynchronize(); // Ensures all previous CUDA calls have completed
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA synchronization failed]: " << cudaGetErrorString(err) << std::endl;
        cudaFree(C);
        return nullptr;
    }

    return C;
};

int main()
{
    float **A = genRanArrA();
    float **B = genRanArrB();
    // printArr(A, M, K);
    // printArr(B, K, N);

    /// 方法1： 使用CPU计算
    float **C = matMulCPU(A, B);
    printArr(C, M, N);

    /// 计时
    cudaEvent_t start, stop;
    // cudaStream_t stream = nullptr;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    // checkCudaErrors(cudaStreamCreate(&stream));

    /// 方法2： 使用GPU加速基础版 - 一个线程只负责C矩阵中的一个元素
    float *d_A = matH2D(A, M, K);
    float *d_B = matH2D(B, K, N);
    float time_cost = 0.0F;
    float *d_C = matMulGPULaunch(d_A, d_B, start, stop, time_cost);
    cudaDeviceSynchronize();
    // printMatLaunch(d_A, M, K);
    // printMatLaunch(d_B, K, N);
    // printMatLaunch(d_C, M, N);
    // cudaDeviceSynchronize();

    /// 方法3： 使用GPU加速基础版2 - 一个线程负责C矩阵中的多个元素(TM*TN)
    float *d_C2 = matMulGPU2Launch(d_A, d_B, start, stop, time_cost);
    cudaDeviceSynchronize();
    // printMatLaunch(d_C2, M, N);
    // cudaDeviceSynchronize();

    /// 方法4： 使用GPU加速升级版 - shared memory
    float *d_C3 = matMulGPUSMLaunch(d_A, d_B, start, stop, time_cost);
    cudaDeviceSynchronize();
    // printMatLaunch(d_A, M, K);
    // printMatLaunch(d_B, K, N);
    // printMatLaunch(d_C3, M, N);
    // cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_C2));
    checkCudaErrors(cudaFree(d_C3));

    // 销毁 CUDA 事件和流
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    // checkCudaErrors(cudaStreamDestroy(stream));

    delArr(A, M);
    delArr(B, K);
    delArr(C, M);

    return 0;
}

// nvcc  matrix.cu -o matrix.o -arch=sm_86
// Ref : https://www.bilibili.com/video/BV1bH4y1w7mm?spm_id_from=333.788.player.switch&vd_source=115911bd71b74bfcc0cad43e576887e4
