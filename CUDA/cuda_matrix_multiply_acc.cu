/* CUDA 实现矩阵的乘法，如何做加速？

编译指令：
nvcc cuda_matrix_multiply_acc.cu -o cuda_matrix_multiply_acc -arch=sm_86
*/
#include <random>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#define M 300
#define K 400
#define N 200

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

#define WARMUP 10

/// @brief 生成矩阵A： m * k
double **genRanArrA()
{
    // double arr[M][K];  在栈上分配内存可能会溢出
    double **arr = new double *[M];
    for (int i = 0; i < M; ++i)
    {
        arr[i] = new double[K];
    }

    // 每次执行会生成不同的随机种子
    // std::random_device rd;
    // const std::uint32_tfixed_seed = rd(); // 这里用 rd() 生成的种子

    std::mt19937 gen(fixed_seed); // Mersenne Twister 随机数引擎
    std::uniform_real_distribution<> dis(0.0, 1.0);

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
double **genRanArrB()
{
    double **arr = new double *[K];
    for (int i = 0; i < K; ++i)
    {
        arr[i] = new double[N];
    }

    std::mt19937 gen(fixed_seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

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
void printArr(double **arr, int rows, int cols)
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
void delArr(double **arr, int rows)
{
    for (int i = 0; i < rows; ++i)
    {
        delete[] arr[i];
    }
    delete[] arr;
}

/// @brief 方法1：使用CPU计算
double **matMulCPU(double **A, double **B)
{
    double **C = new double *[M];
    for (size_t m = 0; m < M; ++m)
    {
        C[m] = new double[N];
    }

    for (size_t m = 0; m < M; ++m)
    {
        for (size_t n = 0; n < N; ++n)
        {
            float sum = 0.0F;
            for (size_t k = 0; k < K; ++k)
            {
                sum += A[m][k] * B[k][n];
            }
            C[m][n] = sum;
        }
    }
    return C;
}

/// @brief CPU矩阵拷贝的GPU上
double *matH2D(double **h_mat, int rows, int cols)
{
    double *d_mat;
    double *h_mat_flat = (double *)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            h_mat_flat[i * cols + j] = h_mat[i][j];
        }
    }

    checkCudaErrors(cudaMalloc((void **)&d_mat, rows * cols * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_mat, h_mat_flat, rows * cols * sizeof(double), cudaMemcpyHostToDevice));

    free(h_mat_flat);
    return d_mat;
}

/// @brief CUDA Kernel：打印矩阵核函数
__global__ void printMatInKernel(double *d_mat, int rows, int cols)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < rows * cols)
    {
        printf("[CUDA Kernel Print Infos] Element at index %d: %f\n", index, d_mat[index]);
    }
}

/// @brief CUDA Kernel：打印矩阵辅助函数
void printMatLaunch(double *d_mat, int rows, int cols)
{
    dim3 dimBlocks(TILE);
    dim3 dimGrids(((rows * cols) + dimBlocks.x - 1) / dimBlocks.x);
    printMatInKernel<<<dimGrids, dimBlocks>>>(d_mat, rows, cols);
}

/// @brief 方法2： 使用GPU加速基础版本
__global__ void matMulGPUKernel(double *A, double *B, double *C)
{
    size_t m = blockIdx.y * blockDim.y + threadIdx.y; // 行
    size_t n = blockIdx.x * blockDim.x + threadIdx.x; // 列

    if (m < M && n < N)
    {
        float value = 0.0F;
        for (size_t k = 0; k < K; ++k)
        {
            value += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = value;
    }
}

double *matMulGPULaunch(double *A, double *B, cudaEvent_t &start, cudaEvent_t &stop, float &time_cost)
{
    double *C;
    checkCudaErrors(cudaMalloc((void **)&C, M * N * sizeof(double)));
    dim3 dimBlocks(TILE, TILE);
    dim3 dimGrids((N + dimBlocks.x - 1) / dimBlocks.x, (M + dimBlocks.y - 1) / dimBlocks.y);

    cudaEventRecord(start);
    for (int i = 0; i < WARMUP; i++)
    {
        matMulGPUKernel<<<dimGrids, dimBlocks>>>(A, B, C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_cost, start, stop);
    std::cout << "[Time Cost] : "
              << "使用GPU加速基础版 time = " << time_cost / WARMUP << "[ms]." << std::endl;

    cudaError_t err = cudaMalloc((void **)&C, M * N * sizeof(double));
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    return C;
};

/// @brief 方法3：使用GPU加速升级版 - shared memory
__global__ void matMulGPUSMKernel(double *A, double *B, double *C)
{
    // 计算当前线程负责的结果矩阵 C 的行和列
    int m = blockIdx.y * TILE + threadIdx.y; // 行
    int n = blockIdx.x * TILE + threadIdx.x; // 列

    __shared__ double shA[TILE][TILE];
    __shared__ double shB[TILE][TILE];

    double sum = 0.0;
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

double *matMulGPUSMLaunch(double *A, double *B, cudaEvent_t &start, cudaEvent_t &stop, float &time_cost)
{
    cudaError_t err;

    double *C;
    checkCudaErrors(cudaMalloc((void **)&C, M * N * sizeof(double)));
    dim3 dimBlocks(TILE, TILE);
    dim3 dimGrids((N + dimBlocks.x - 1) / dimBlocks.x, (M + dimBlocks.y - 1) / dimBlocks.y);

    cudaEventRecord(start);
    matMulGPUSMKernel<<<dimGrids, dimBlocks>>>(A, B, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_cost, start, stop);
    std::cout << "[Time Cost] : "
              << "使用GPU加速 - shared memory time = " << time_cost << "[ms]." << std::endl;
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
    double **A = genRanArrA();
    double **B = genRanArrB();
    // printArr(A, M, K);
    // printArr(B, K, N);

    /// 方法1： 使用CPU计算
    double **C = matMulCPU(A, B);
    // printArr(C, M, N);

    /// 计时
    cudaEvent_t start, stop;
    // cudaStream_t stream = nullptr;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    // checkCudaErrors(cudaStreamCreate(&stream));

    /// 方法2： 使用GPU加速基础版
    double *d_A = matH2D(A, M, K);
    double *d_B = matH2D(B, K, N);
    float time_cost = 0.0F;

    double *d_C = matMulGPULaunch(d_A, d_B, start, stop, time_cost);
    cudaDeviceSynchronize();
    // printMatLaunch(d_A, M, K);
    // printMatLaunch(d_B, K, N);
    // printMatLaunch(d_C, M, N);
    // cudaDeviceSynchronize();

    /// 方法3： 使用GPU加速升级版 - shared memory
    double *d_C3 = matMulGPUSMLaunch(d_A, d_B, start, stop, time_cost);
    cudaDeviceSynchronize();
    // printMatLaunch(d_A, M, K);
    // printMatLaunch(d_B, K, N);
    // printMatLaunch(d_C3, M, N);
    // cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
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