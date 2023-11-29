/*
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>

#define CUDA_SAFECALL(call)                                                 \
    {                                                                       \
        call;                                                               \
        cudaError err = cudaGetLastError();                                 \
        if (cudaSuccess != err) {                                           \
            fprintf(                                                        \
                stderr,                                                     \
                "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            fflush(stderr);                                                 \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }


// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd_1(double *a, double *b, double *c, int n) {
    uint32_t start = clock();
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we do not go out of bounds
    if (id < n && a[id] > b[id]) c[id] = a[id] + b[id];
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd_2(double *a, double *b, double *c, int n) {
    uint32_t start = clock();
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n && a[id] <= b[id]) c[id] = a[id] + b[id];
}

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd_3(double *a, double *b, double *c, int n) {
    uint32_t start = clock();
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n && a[id] <= b[id]) c[id] = a[id] + b[id];
}



int main(int argc, char *argv[]) {
    // Size of vectors
    int n = 40960;
    if (argc > 1) n = atoi(argv[1]);

    // Host input vectors
    double *h_a_1;
    double *h_b_1;
    double *h_a_2;
    double *h_b_2;
    double *h_a_3;
    double *h_b_3;
    // Host output vector
    double *h_c_1;
    double *h_c_2;
    double *h_c_3;

    // Device input vectors
    double *d_a_1;
    double *d_b_1;
    double *d_a_2;
    double *d_b_2;
    double *d_a_3;
    double *d_b_3;
    // Device output vector
    double *d_c_1;
    double *d_c_2;
    double *d_c_3;

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);

    // Allocate memory for each vector on host
    h_a_1 = (double *)malloc(bytes);
    h_a_2 = (double *)malloc(bytes);
    h_a_3 = (double *)malloc(bytes);
    h_b_1 = (double *)malloc(bytes);
    h_b_2 = (double *)malloc(bytes);
    h_b_3 = (double *)malloc(bytes);
    h_c_1 = (double *)malloc(bytes);
    h_c_2 = (double *)malloc(bytes);
    h_c_3 = (double *)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a_1, bytes);
    cudaMalloc(&d_b_1, bytes);
    cudaMalloc(&d_c_1, bytes);
    cudaMalloc(&d_a_2, bytes);
    cudaMalloc(&d_b_2, bytes);
    cudaMalloc(&d_c_2, bytes);
    cudaMalloc(&d_a_3, bytes);
    cudaMalloc(&d_b_3, bytes);
    cudaMalloc(&d_c_3, bytes);

    int i;
    // Initialize vectors on host
    // for (i = 0; i < n; i++) {
    //     h_a_1[i] = sin(i) * sin(i);
    //     h_b_1[i] = cos(i) * cos(i);
    //     h_c_1[i] = 0;
    // }
    for (i = 0; i < n; i++) {
        h_a_1[i] = 0.1*i;
        h_b_1[i] = 0.1*i;
        h_c_1[i] = 0;
        h_a_2[i] = 0.1*i;
        h_b_2[i] = 0.1*i;
        h_c_2[i] = 0;
        h_a_3[i] = 0.1*i;
        h_b_3[i] = 0.1*i;
        h_c_3[i] = 0;
    }

    // Copy host vectors to device
    cudaMemcpy(d_a_1, h_a_1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_1, h_b_1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_1, h_c_1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_2, h_a_2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_2, h_b_2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_2, h_c_2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_3, h_a_3, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_3, h_b_3, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c_3, h_c_3, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 256;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n / blockSize);
    
    // Execute the kernel
    CUDA_SAFECALL((vecAdd_1<<<gridSize, blockSize>>>(d_a_1, d_b_1, d_c_1, n)));
    CUDA_SAFECALL((vecAdd_2<<<gridSize, blockSize>>>(d_a_2, d_b_2, d_c_2, n)));
    CUDA_SAFECALL((vecAdd_3<<<gridSize, blockSize>>>(d_a_3, d_b_3, d_c_3, n)));
    
    // Copy array back to host
    cudaMemcpy(h_c_1, d_c_1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_2, d_c_2, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_3, d_c_3, bytes, cudaMemcpyDeviceToHost);

    // Sum up vector c and print result divided by n, this should equal 1 within
    // error
    double sum;
    sum = 0;
    for (i = 0; i < n; i++) sum += h_c_1[i];
    printf("Final sum_1 = %f; sum_1/n = %f (should be ~1)\n", sum, sum / n);
    sum = 0;
    for (i = 0; i < n; i++) sum += h_c_2[i];
    printf("Final sum_2 = %f; sum_2/n = %f (should be ~1)\n", sum, sum / n);
    sum = 0;
    for (i = 0; i < n; i++) sum += h_c_3[i];
    printf("Final sum_3 = %f; sum_3/n = %f (should be ~1)\n", sum, sum / n);

    // Release device memory
    cudaFree(d_a_1);
    cudaFree(d_b_1);
    cudaFree(d_c_1);
    cudaFree(d_a_2);
    cudaFree(d_b_2);
    cudaFree(d_c_2);
    cudaFree(d_a_3);
    cudaFree(d_b_3);
    cudaFree(d_c_3);

    // Release host memory
    free(h_a_1);
    free(h_b_1);
    free(h_c_1);
    free(h_a_2);
    free(h_b_2);
    free(h_c_2);
    free(h_a_3);
    free(h_b_3);
    free(h_c_3);

    return 0;
}
*/

/**/
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include <cuComplex.h>
#include <cublas_api.h>
#include <cuda_runtime_api.h>
#include <library_types.h>

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// Returns cudaDataType value as defined in library_types.h for the string
// containing type name
cudaDataType get_cuda_library_type(std::string type_string) {
    if (type_string.compare("CUDA_R_16F") == 0)
        return CUDA_R_16F;
    else if (type_string.compare("CUDA_C_16F") == 0)
        return CUDA_C_16F;
    else if (type_string.compare("CUDA_R_32F") == 0)
        return CUDA_R_32F;
    else if (type_string.compare("CUDA_C_32F") == 0)
        return CUDA_C_32F;
    else if (type_string.compare("CUDA_R_64F") == 0)
        return CUDA_R_64F;
    else if (type_string.compare("CUDA_C_64F") == 0)
        return CUDA_C_64F;
    else if (type_string.compare("CUDA_R_8I") == 0)
        return CUDA_R_8I;
    else if (type_string.compare("CUDA_C_8I") == 0)
        return CUDA_C_8I;
    else if (type_string.compare("CUDA_R_8U") == 0)
        return CUDA_R_8U;
    else if (type_string.compare("CUDA_C_8U") == 0)
        return CUDA_C_8U;
    else if (type_string.compare("CUDA_R_32I") == 0)
        return CUDA_R_32I;
    else if (type_string.compare("CUDA_C_32I") == 0)
        return CUDA_C_32I;
    else if (type_string.compare("CUDA_R_32U") == 0)
        return CUDA_R_32U;
    else if (type_string.compare("CUDA_C_32U") == 0)
        return CUDA_C_32U;
    else
        throw std::runtime_error("Unknown CUDA datatype");
}

__global__ void cublasHgemm111(                       cublasHandle_t handle, 
                                                      cublasOperation_t transa,
                                                      cublasOperation_t transb, 
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const __half *alpha, /* host or device pointer */  
                                                      const __half *A, 
                                                      int lda,
                                                      const __half *B,
                                                      int ldb, 
                                                      const __half *beta, /* host or device pointer */  
                                                      __half *C,
                                                      int ldc) {
    uint32_t start = clock();
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    C[id] = A[id];
}

using data_type = half;

int main(int argc, char *argv[]) {
    
    int m = 257;
    int k = 1;
    int n = 513; 
    // int m = 513;
    // int k = 513;
    // int n = 513; 
    
    const int lda = k;
    const int ldb = n;
    const int ldc = n;

    cublasHandle_t cublasH1,cublasH2,cublasH3,cublasH4,cublasH5,cublasH6,cublasH7,cublasH8,cublasH9,cublasH10,cublasH11,cublasH12,cublasH13,cublasH14,cublasH15,cublasH16,cublasH17,cublasH18,cublasH19,cublasH20,cublasH21,cublasH22,cublasH23,cublasH24,cublasH25,cublasH26,cublasH27,cublasH28,cublasH29,cublasH30,cublasH31,cublasH32,cublasH33,cublasH34,cublasH35,cublasH36,cublasH37,cublasH38,cublasH39,cublasH40,cublasH41,cublasH42,cublasH43,cublasH44,cublasH45,cublasH46,cublasH47,cublasH48,cublasH49,cublasH50,cublasH51,cublasH52,cublasH53,cublasH54,cublasH55,cublasH56,cublasH57,cublasH58,cublasH59,cublasH60,cublasH61,cublasH62,cublasH63,cublasH64,cublasH65,cublasH66,cublasH67,cublasH68,cublasH69,cublasH70,cublasH71,cublasH72,cublasH73,cublasH74,cublasH75,cublasH76,cublasH77,cublasH78,cublasH79,cublasH80,cublasH81 = NULL;


    cudaStream_t stream1,stream2,stream3,stream4,stream5,stream6,stream7,stream8,stream9,stream10,stream11,stream12,stream13,stream14,stream15,stream16,stream17,stream18,stream19,stream20,stream21,stream22,stream23,stream24,stream25,stream26,stream27,stream28,stream29,stream30,stream31,stream32,stream33,stream34,stream35,stream36,stream37,stream38,stream39,stream40,stream41,stream42,stream43,stream44,stream45,stream46,stream47,stream48,stream49,stream50,stream51,stream52,stream53,stream54,stream55,stream56,stream57,stream58,stream59,stream60,stream61,stream62,stream63,stream64,stream65,stream66,stream67,stream68,stream69,stream70,stream71,stream72,stream73,stream74,stream75,stream76,stream77,stream78,stream79,stream80,stream81 = NULL;

    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<> dis(0.2, 1.0);

    //
    //   A = | 1.0 | 2.0 |
    //       | 3.0 | 4.0 |
    //
    //   B = | 5.0 | 6.0 |
    //       | 7.0 | 8.0 |
    //

    // const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    // const std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};
    std::vector<data_type> A(m * k, 1./13.);
    std::vector<data_type> B(k * n, 1./13.);

    // for(std::vector<data_type>::iterator it = A.begin(); it != A.end(); ++it)
    //     *it = dis(gen);
    // for(std::vector<data_type>::iterator it = B.begin(); it != B.end(); ++it)
    //     *it = dis(gen);

    std::vector<data_type> C(m * n);
    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    // define 71 d_a_x device data
    data_type *d_a_1 = nullptr;
    data_type *d_b_1 = nullptr;
    data_type *d_c_1 = nullptr;
    data_type *d_a_2 = nullptr;
    data_type *d_b_2 = nullptr;
    data_type *d_c_2 = nullptr;
    data_type *d_a_3 = nullptr;
    data_type *d_b_3 = nullptr;
    data_type *d_c_3 = nullptr;
    data_type *d_a_4 = nullptr;
    data_type *d_b_4 = nullptr;
    data_type *d_c_4 = nullptr;
    data_type *d_a_5 = nullptr;
    data_type *d_b_5 = nullptr;
    data_type *d_c_5 = nullptr;
    data_type *d_a_6 = nullptr;
    data_type *d_b_6 = nullptr;
    data_type *d_c_6 = nullptr;
    data_type *d_a_7 = nullptr;
    data_type *d_b_7 = nullptr;
    data_type *d_c_7 = nullptr;
    data_type *d_a_8 = nullptr;
    data_type *d_b_8 = nullptr;
    data_type *d_c_8 = nullptr;
    data_type *d_a_9 = nullptr;
    data_type *d_b_9 = nullptr;
    data_type *d_c_9 = nullptr;
    data_type *d_a_10 = nullptr;
    data_type *d_b_10 = nullptr;
    data_type *d_c_10 = nullptr;
    data_type *d_a_11 = nullptr;
    data_type *d_b_11 = nullptr;
    data_type *d_c_11 = nullptr;
    data_type *d_a_12 = nullptr;
    data_type *d_b_12 = nullptr;
    data_type *d_c_12 = nullptr;
    data_type *d_a_13 = nullptr;
    data_type *d_b_13 = nullptr;
    data_type *d_c_13 = nullptr;
    data_type *d_a_14 = nullptr;
    data_type *d_b_14 = nullptr;
    data_type *d_c_14 = nullptr;
    data_type *d_a_15 = nullptr;
    data_type *d_b_15 = nullptr;
    data_type *d_c_15 = nullptr;
    data_type *d_a_16 = nullptr;
    data_type *d_b_16 = nullptr;
    data_type *d_c_16 = nullptr;
    data_type *d_a_17 = nullptr;
    data_type *d_b_17 = nullptr;
    data_type *d_c_17 = nullptr;
    data_type *d_a_18 = nullptr;
    data_type *d_b_18 = nullptr;
    data_type *d_c_18 = nullptr;
    data_type *d_a_19 = nullptr;
    data_type *d_b_19 = nullptr;
    data_type *d_c_19 = nullptr;
    data_type *d_a_20 = nullptr;
    data_type *d_b_20 = nullptr;
    data_type *d_c_20 = nullptr;
    data_type *d_a_21 = nullptr;
    data_type *d_b_21 = nullptr;
    data_type *d_c_21 = nullptr;
    data_type *d_a_22 = nullptr;
    data_type *d_b_22 = nullptr;
    data_type *d_c_22 = nullptr;
    data_type *d_a_23 = nullptr;
    data_type *d_b_23 = nullptr;
    data_type *d_c_23 = nullptr;
    data_type *d_a_24 = nullptr;
    data_type *d_b_24 = nullptr;
    data_type *d_c_24 = nullptr;
    data_type *d_a_25 = nullptr;
    data_type *d_b_25 = nullptr;
    data_type *d_c_25 = nullptr;
    data_type *d_a_26 = nullptr;
    data_type *d_b_26 = nullptr;
    data_type *d_c_26 = nullptr;
    data_type *d_a_27 = nullptr;
    data_type *d_b_27 = nullptr;
    data_type *d_c_27 = nullptr;
    data_type *d_a_28 = nullptr;
    data_type *d_b_28 = nullptr;
    data_type *d_c_28 = nullptr;
    data_type *d_a_29 = nullptr;
    data_type *d_b_29 = nullptr;
    data_type *d_c_29 = nullptr;
    data_type *d_a_30 = nullptr;
    data_type *d_b_30 = nullptr;
    data_type *d_c_30 = nullptr;
    data_type *d_a_31 = nullptr;
    data_type *d_b_31 = nullptr;
    data_type *d_c_31 = nullptr;
    data_type *d_a_32 = nullptr;
    data_type *d_b_32 = nullptr;
    data_type *d_c_32 = nullptr;
    data_type *d_a_33 = nullptr;
    data_type *d_b_33 = nullptr;
    data_type *d_c_33 = nullptr;
    data_type *d_a_34 = nullptr;
    data_type *d_b_34 = nullptr;
    data_type *d_c_34 = nullptr;
    data_type *d_a_35 = nullptr;
    data_type *d_b_35 = nullptr;
    data_type *d_c_35 = nullptr;
    data_type *d_a_36 = nullptr;
    data_type *d_b_36 = nullptr;
    data_type *d_c_36 = nullptr;
    data_type *d_a_37 = nullptr;
    data_type *d_b_37 = nullptr;
    data_type *d_c_37 = nullptr;
    data_type *d_a_38 = nullptr;
    data_type *d_b_38 = nullptr;
    data_type *d_c_38 = nullptr;
    data_type *d_a_39 = nullptr;
    data_type *d_b_39 = nullptr;
    data_type *d_c_39 = nullptr;
    data_type *d_a_40 = nullptr;
    data_type *d_b_40 = nullptr;
    data_type *d_c_40 = nullptr;
    data_type *d_a_41 = nullptr;
    data_type *d_b_41 = nullptr;
    data_type *d_c_41 = nullptr;
    data_type *d_a_42 = nullptr;
    data_type *d_b_42 = nullptr;
    data_type *d_c_42 = nullptr;
    data_type *d_a_43 = nullptr;
    data_type *d_b_43 = nullptr;
    data_type *d_c_43 = nullptr;
    data_type *d_a_44 = nullptr;
    data_type *d_b_44 = nullptr;
    data_type *d_c_44 = nullptr;
    data_type *d_a_45 = nullptr;
    data_type *d_b_45 = nullptr;
    data_type *d_c_45 = nullptr;
    data_type *d_a_46 = nullptr;
    data_type *d_b_46 = nullptr;
    data_type *d_c_46 = nullptr;
    data_type *d_a_47 = nullptr;
    data_type *d_b_47 = nullptr;
    data_type *d_c_47 = nullptr;
    data_type *d_a_48 = nullptr;
    data_type *d_b_48 = nullptr;
    data_type *d_c_48 = nullptr;
    data_type *d_a_49 = nullptr;
    data_type *d_b_49 = nullptr;
    data_type *d_c_49 = nullptr;
    data_type *d_a_50 = nullptr;
    data_type *d_b_50 = nullptr;
    data_type *d_c_50 = nullptr;
    data_type *d_a_51 = nullptr;
    data_type *d_b_51 = nullptr;
    data_type *d_c_51 = nullptr;
    data_type *d_a_52 = nullptr;
    data_type *d_b_52 = nullptr;
    data_type *d_c_52 = nullptr;
    data_type *d_a_53 = nullptr;
    data_type *d_b_53 = nullptr;
    data_type *d_c_53 = nullptr;
    data_type *d_a_54 = nullptr;
    data_type *d_b_54 = nullptr;
    data_type *d_c_54 = nullptr;
    data_type *d_a_55 = nullptr;
    data_type *d_b_55 = nullptr;
    data_type *d_c_55 = nullptr;
    data_type *d_a_56 = nullptr;
    data_type *d_b_56 = nullptr;
    data_type *d_c_56 = nullptr;
    data_type *d_a_57 = nullptr;
    data_type *d_b_57 = nullptr;
    data_type *d_c_57 = nullptr;
    data_type *d_a_58 = nullptr;
    data_type *d_b_58 = nullptr;
    data_type *d_c_58 = nullptr;
    data_type *d_a_59 = nullptr;
    data_type *d_b_59 = nullptr;
    data_type *d_c_59 = nullptr;
    data_type *d_a_60 = nullptr;
    data_type *d_b_60 = nullptr;
    data_type *d_c_60 = nullptr;
    data_type *d_a_61 = nullptr;
    data_type *d_b_61 = nullptr;
    data_type *d_c_61 = nullptr;
    data_type *d_a_62 = nullptr;
    data_type *d_b_62 = nullptr;
    data_type *d_c_62 = nullptr;
    data_type *d_a_63 = nullptr;
    data_type *d_b_63 = nullptr;
    data_type *d_c_63 = nullptr;
    data_type *d_a_64 = nullptr;
    data_type *d_b_64 = nullptr;
    data_type *d_c_64 = nullptr;
    data_type *d_a_65 = nullptr;
    data_type *d_b_65 = nullptr;
    data_type *d_c_65 = nullptr;
    data_type *d_a_66 = nullptr;
    data_type *d_b_66 = nullptr;
    data_type *d_c_66 = nullptr;
    data_type *d_a_67 = nullptr;
    data_type *d_b_67 = nullptr;
    data_type *d_c_67 = nullptr;
    data_type *d_a_68 = nullptr;
    data_type *d_b_68 = nullptr;
    data_type *d_c_68 = nullptr;
    data_type *d_a_69 = nullptr;
    data_type *d_b_69 = nullptr;
    data_type *d_c_69 = nullptr;
    data_type *d_a_70 = nullptr;
    data_type *d_b_70 = nullptr;
    data_type *d_c_70 = nullptr;
    data_type *d_a_71 = nullptr;
    data_type *d_b_71 = nullptr;
    data_type *d_c_71 = nullptr;
    data_type *d_a_72 = nullptr;
    data_type *d_b_72 = nullptr;
    data_type *d_c_72 = nullptr;
    data_type *d_a_73 = nullptr;
    data_type *d_b_73 = nullptr;
    data_type *d_c_73 = nullptr;
    data_type *d_a_74 = nullptr;
    data_type *d_b_74 = nullptr;
    data_type *d_c_74 = nullptr;
    data_type *d_a_75 = nullptr;
    data_type *d_b_75 = nullptr;
    data_type *d_c_75 = nullptr;
    data_type *d_a_76 = nullptr;
    data_type *d_b_76 = nullptr;
    data_type *d_c_76 = nullptr;
    data_type *d_a_77 = nullptr;
    data_type *d_b_77 = nullptr;
    data_type *d_c_77 = nullptr;
    data_type *d_a_78 = nullptr;
    data_type *d_b_78 = nullptr;
    data_type *d_c_78 = nullptr;
    data_type *d_a_79 = nullptr;
    data_type *d_b_79 = nullptr;
    data_type *d_c_79 = nullptr;
    data_type *d_a_80 = nullptr;
    data_type *d_b_80 = nullptr;
    data_type *d_c_80 = nullptr;
    data_type *d_a_81 = nullptr;
    data_type *d_b_81 = nullptr;
    data_type *d_c_81 = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;


    // step 1: create cublas handle, bind a stream 
    CUBLAS_CHECK(cublasCreate(&cublasH1));CUBLAS_CHECK(cublasCreate(&cublasH2));CUBLAS_CHECK(cublasCreate(&cublasH3));CUBLAS_CHECK(cublasCreate(&cublasH4));CUBLAS_CHECK(cublasCreate(&cublasH5));CUBLAS_CHECK(cublasCreate(&cublasH6));CUBLAS_CHECK(cublasCreate(&cublasH7));CUBLAS_CHECK(cublasCreate(&cublasH8));CUBLAS_CHECK(cublasCreate(&cublasH9));CUBLAS_CHECK(cublasCreate(&cublasH10));CUBLAS_CHECK(cublasCreate(&cublasH11));CUBLAS_CHECK(cublasCreate(&cublasH12));CUBLAS_CHECK(cublasCreate(&cublasH13));CUBLAS_CHECK(cublasCreate(&cublasH14));CUBLAS_CHECK(cublasCreate(&cublasH15));CUBLAS_CHECK(cublasCreate(&cublasH16));CUBLAS_CHECK(cublasCreate(&cublasH17));CUBLAS_CHECK(cublasCreate(&cublasH18));CUBLAS_CHECK(cublasCreate(&cublasH19));CUBLAS_CHECK(cublasCreate(&cublasH20));CUBLAS_CHECK(cublasCreate(&cublasH21));CUBLAS_CHECK(cublasCreate(&cublasH22));CUBLAS_CHECK(cublasCreate(&cublasH23));CUBLAS_CHECK(cublasCreate(&cublasH24));CUBLAS_CHECK(cublasCreate(&cublasH25));CUBLAS_CHECK(cublasCreate(&cublasH26));CUBLAS_CHECK(cublasCreate(&cublasH27));CUBLAS_CHECK(cublasCreate(&cublasH28));CUBLAS_CHECK(cublasCreate(&cublasH29));CUBLAS_CHECK(cublasCreate(&cublasH30));CUBLAS_CHECK(cublasCreate(&cublasH31));CUBLAS_CHECK(cublasCreate(&cublasH32));CUBLAS_CHECK(cublasCreate(&cublasH33));CUBLAS_CHECK(cublasCreate(&cublasH34));CUBLAS_CHECK(cublasCreate(&cublasH35));CUBLAS_CHECK(cublasCreate(&cublasH36));CUBLAS_CHECK(cublasCreate(&cublasH37));CUBLAS_CHECK(cublasCreate(&cublasH38));CUBLAS_CHECK(cublasCreate(&cublasH39));CUBLAS_CHECK(cublasCreate(&cublasH40));CUBLAS_CHECK(cublasCreate(&cublasH41));CUBLAS_CHECK(cublasCreate(&cublasH42));CUBLAS_CHECK(cublasCreate(&cublasH43));CUBLAS_CHECK(cublasCreate(&cublasH44));CUBLAS_CHECK(cublasCreate(&cublasH45));CUBLAS_CHECK(cublasCreate(&cublasH46));CUBLAS_CHECK(cublasCreate(&cublasH47));CUBLAS_CHECK(cublasCreate(&cublasH48));CUBLAS_CHECK(cublasCreate(&cublasH49));CUBLAS_CHECK(cublasCreate(&cublasH50));CUBLAS_CHECK(cublasCreate(&cublasH51));CUBLAS_CHECK(cublasCreate(&cublasH52));CUBLAS_CHECK(cublasCreate(&cublasH53));CUBLAS_CHECK(cublasCreate(&cublasH54));CUBLAS_CHECK(cublasCreate(&cublasH55));CUBLAS_CHECK(cublasCreate(&cublasH56));CUBLAS_CHECK(cublasCreate(&cublasH57));CUBLAS_CHECK(cublasCreate(&cublasH58));CUBLAS_CHECK(cublasCreate(&cublasH59));CUBLAS_CHECK(cublasCreate(&cublasH60));CUBLAS_CHECK(cublasCreate(&cublasH61));CUBLAS_CHECK(cublasCreate(&cublasH62));CUBLAS_CHECK(cublasCreate(&cublasH63));CUBLAS_CHECK(cublasCreate(&cublasH64));CUBLAS_CHECK(cublasCreate(&cublasH65));CUBLAS_CHECK(cublasCreate(&cublasH66));CUBLAS_CHECK(cublasCreate(&cublasH67));CUBLAS_CHECK(cublasCreate(&cublasH68));CUBLAS_CHECK(cublasCreate(&cublasH69));CUBLAS_CHECK(cublasCreate(&cublasH70));CUBLAS_CHECK(cublasCreate(&cublasH71));CUBLAS_CHECK(cublasCreate(&cublasH72));CUBLAS_CHECK(cublasCreate(&cublasH73));CUBLAS_CHECK(cublasCreate(&cublasH74));CUBLAS_CHECK(cublasCreate(&cublasH75));CUBLAS_CHECK(cublasCreate(&cublasH76));CUBLAS_CHECK(cublasCreate(&cublasH77));CUBLAS_CHECK(cublasCreate(&cublasH78));CUBLAS_CHECK(cublasCreate(&cublasH79));CUBLAS_CHECK(cublasCreate(&cublasH80));CUBLAS_CHECK(cublasCreate(&cublasH81));
    

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream4, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream5, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream6, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream7, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream8, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream9, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream10, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream11, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream12, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream13, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream14, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream15, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream16, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream17, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream18, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream19, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream20, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream21, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream22, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream23, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream24, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream25, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream26, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream27, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream28, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream29, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream30, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream31, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream32, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream33, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream34, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream35, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream36, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream37, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream38, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream39, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream40, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream41, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream42, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream43, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream44, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream45, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream46, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream47, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream48, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream49, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream50, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream51, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream52, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream53, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream54, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream55, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream56, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream57, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream58, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream59, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream60, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream61, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream62, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream63, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream64, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream65, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream66, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream67, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream68, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream69, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream70, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream71, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream72, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream73, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream74, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream75, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream76, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream77, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream78, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream79, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream80, cudaStreamNonBlocking));CUDA_CHECK(cudaStreamCreateWithFlags(&stream81, cudaStreamNonBlocking));


    CUBLAS_CHECK(cublasSetStream(cublasH1, stream1));CUBLAS_CHECK(cublasSetStream(cublasH2, stream2));CUBLAS_CHECK(cublasSetStream(cublasH3, stream3));CUBLAS_CHECK(cublasSetStream(cublasH4, stream4));CUBLAS_CHECK(cublasSetStream(cublasH5, stream5));CUBLAS_CHECK(cublasSetStream(cublasH6, stream6));CUBLAS_CHECK(cublasSetStream(cublasH7, stream7));CUBLAS_CHECK(cublasSetStream(cublasH8, stream8));CUBLAS_CHECK(cublasSetStream(cublasH9, stream9));CUBLAS_CHECK(cublasSetStream(cublasH10, stream10));CUBLAS_CHECK(cublasSetStream(cublasH11, stream11));CUBLAS_CHECK(cublasSetStream(cublasH12, stream12));CUBLAS_CHECK(cublasSetStream(cublasH13, stream13));CUBLAS_CHECK(cublasSetStream(cublasH14, stream14));CUBLAS_CHECK(cublasSetStream(cublasH15, stream15));CUBLAS_CHECK(cublasSetStream(cublasH16, stream16));CUBLAS_CHECK(cublasSetStream(cublasH17, stream17));CUBLAS_CHECK(cublasSetStream(cublasH18, stream18));CUBLAS_CHECK(cublasSetStream(cublasH19, stream19));CUBLAS_CHECK(cublasSetStream(cublasH20, stream20));CUBLAS_CHECK(cublasSetStream(cublasH21, stream21));CUBLAS_CHECK(cublasSetStream(cublasH22, stream22));CUBLAS_CHECK(cublasSetStream(cublasH23, stream23));CUBLAS_CHECK(cublasSetStream(cublasH24, stream24));CUBLAS_CHECK(cublasSetStream(cublasH25, stream25));CUBLAS_CHECK(cublasSetStream(cublasH26, stream26));CUBLAS_CHECK(cublasSetStream(cublasH27, stream27));CUBLAS_CHECK(cublasSetStream(cublasH28, stream28));CUBLAS_CHECK(cublasSetStream(cublasH29, stream29));CUBLAS_CHECK(cublasSetStream(cublasH30, stream30));CUBLAS_CHECK(cublasSetStream(cublasH31, stream31));CUBLAS_CHECK(cublasSetStream(cublasH32, stream32));CUBLAS_CHECK(cublasSetStream(cublasH33, stream33));CUBLAS_CHECK(cublasSetStream(cublasH34, stream34));CUBLAS_CHECK(cublasSetStream(cublasH35, stream35));CUBLAS_CHECK(cublasSetStream(cublasH36, stream36));CUBLAS_CHECK(cublasSetStream(cublasH37, stream37));CUBLAS_CHECK(cublasSetStream(cublasH38, stream38));CUBLAS_CHECK(cublasSetStream(cublasH39, stream39));CUBLAS_CHECK(cublasSetStream(cublasH40, stream40));CUBLAS_CHECK(cublasSetStream(cublasH41, stream41));CUBLAS_CHECK(cublasSetStream(cublasH42, stream42));CUBLAS_CHECK(cublasSetStream(cublasH43, stream43));CUBLAS_CHECK(cublasSetStream(cublasH44, stream44));CUBLAS_CHECK(cublasSetStream(cublasH45, stream45));CUBLAS_CHECK(cublasSetStream(cublasH46, stream46));CUBLAS_CHECK(cublasSetStream(cublasH47, stream47));CUBLAS_CHECK(cublasSetStream(cublasH48, stream48));CUBLAS_CHECK(cublasSetStream(cublasH49, stream49));CUBLAS_CHECK(cublasSetStream(cublasH50, stream50));CUBLAS_CHECK(cublasSetStream(cublasH51, stream51));CUBLAS_CHECK(cublasSetStream(cublasH52, stream52));CUBLAS_CHECK(cublasSetStream(cublasH53, stream53));CUBLAS_CHECK(cublasSetStream(cublasH54, stream54));CUBLAS_CHECK(cublasSetStream(cublasH55, stream55));CUBLAS_CHECK(cublasSetStream(cublasH56, stream56));CUBLAS_CHECK(cublasSetStream(cublasH57, stream57));CUBLAS_CHECK(cublasSetStream(cublasH58, stream58));CUBLAS_CHECK(cublasSetStream(cublasH59, stream59));CUBLAS_CHECK(cublasSetStream(cublasH60, stream60));CUBLAS_CHECK(cublasSetStream(cublasH61, stream61));CUBLAS_CHECK(cublasSetStream(cublasH62, stream62));CUBLAS_CHECK(cublasSetStream(cublasH63, stream63));CUBLAS_CHECK(cublasSetStream(cublasH64, stream64));CUBLAS_CHECK(cublasSetStream(cublasH65, stream65));CUBLAS_CHECK(cublasSetStream(cublasH66, stream66));CUBLAS_CHECK(cublasSetStream(cublasH67, stream67));CUBLAS_CHECK(cublasSetStream(cublasH68, stream68));CUBLAS_CHECK(cublasSetStream(cublasH69, stream69));CUBLAS_CHECK(cublasSetStream(cublasH70, stream70));CUBLAS_CHECK(cublasSetStream(cublasH71, stream71));CUBLAS_CHECK(cublasSetStream(cublasH72, stream72));CUBLAS_CHECK(cublasSetStream(cublasH73, stream73));CUBLAS_CHECK(cublasSetStream(cublasH74, stream74));CUBLAS_CHECK(cublasSetStream(cublasH75, stream75));CUBLAS_CHECK(cublasSetStream(cublasH76, stream76));CUBLAS_CHECK(cublasSetStream(cublasH77, stream77));CUBLAS_CHECK(cublasSetStream(cublasH78, stream78));CUBLAS_CHECK(cublasSetStream(cublasH79, stream79));CUBLAS_CHECK(cublasSetStream(cublasH80, stream80));CUBLAS_CHECK(cublasSetStream(cublasH81, stream81));

    // step 2: copy data to device 
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_1), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_1), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_1), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_2), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_2), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_2), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_3), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_3), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_3), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_4), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_4), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_4), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_5), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_5), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_5), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_6), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_6), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_6), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_7), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_7), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_7), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_8), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_8), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_8), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_9), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_9), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_9), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_10), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_10), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_10), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_11), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_11), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_11), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_12), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_12), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_12), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_13), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_13), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_13), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_14), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_14), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_14), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_15), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_15), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_15), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_16), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_16), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_16), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_17), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_17), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_17), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_18), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_18), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_18), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_19), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_19), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_19), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_20), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_20), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_20), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_21), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_21), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_21), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_22), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_22), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_22), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_23), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_23), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_23), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_24), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_24), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_24), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_25), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_25), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_25), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_26), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_26), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_26), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_27), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_27), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_27), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_28), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_28), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_28), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_29), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_29), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_29), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_30), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_30), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_30), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_31), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_31), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_31), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_32), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_32), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_32), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_33), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_33), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_33), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_34), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_34), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_34), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_35), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_35), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_35), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_36), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_36), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_36), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_37), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_37), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_37), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_38), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_38), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_38), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_39), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_39), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_39), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_40), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_40), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_40), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_41), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_41), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_41), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_42), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_42), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_42), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_43), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_43), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_43), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_44), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_44), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_44), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_45), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_45), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_45), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_46), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_46), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_46), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_47), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_47), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_47), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_48), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_48), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_48), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_49), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_49), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_49), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_50), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_50), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_50), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_51), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_51), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_51), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_52), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_52), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_52), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_53), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_53), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_53), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_54), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_54), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_54), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_55), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_55), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_55), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_56), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_56), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_56), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_57), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_57), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_57), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_58), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_58), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_58), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_59), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_59), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_59), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_60), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_60), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_60), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_61), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_61), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_61), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_62), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_62), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_62), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_63), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_63), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_63), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_64), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_64), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_64), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_65), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_65), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_65), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_66), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_66), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_66), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_67), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_67), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_67), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_68), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_68), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_68), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_69), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_69), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_69), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_70), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_70), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_70), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_71), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_71), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_71), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_72), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_72), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_72), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_73), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_73), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_73), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_74), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_74), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_74), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_75), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_75), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_75), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_76), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_76), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_76), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_77), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_77), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_77), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_78), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_78), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_78), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_79), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_79), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_79), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_80), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_80), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_80), sizeof(data_type) * C.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a_81), sizeof(data_type) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b_81), sizeof(data_type) * B.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c_81), sizeof(data_type) * C.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_a_1, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_b_1, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_a_2, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream2));
    CUDA_CHECK(cudaMemcpyAsync(d_b_2, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream2));
    CUDA_CHECK(cudaMemcpyAsync(d_a_3, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream3));
    CUDA_CHECK(cudaMemcpyAsync(d_b_3, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream3));
    CUDA_CHECK(cudaMemcpyAsync(d_a_4, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream4));
    CUDA_CHECK(cudaMemcpyAsync(d_b_4, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream4));
    CUDA_CHECK(cudaMemcpyAsync(d_a_5, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream5));
    CUDA_CHECK(cudaMemcpyAsync(d_b_5, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream5));
    CUDA_CHECK(cudaMemcpyAsync(d_a_6, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream6));
    CUDA_CHECK(cudaMemcpyAsync(d_b_6, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream6));
    CUDA_CHECK(cudaMemcpyAsync(d_a_7, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream7));
    CUDA_CHECK(cudaMemcpyAsync(d_b_7, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream7));
    CUDA_CHECK(cudaMemcpyAsync(d_a_8, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream8));
    CUDA_CHECK(cudaMemcpyAsync(d_b_8, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream8));
    CUDA_CHECK(cudaMemcpyAsync(d_a_9, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream9));
    CUDA_CHECK(cudaMemcpyAsync(d_b_9, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream9));
    CUDA_CHECK(cudaMemcpyAsync(d_a_10, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream10));
    CUDA_CHECK(cudaMemcpyAsync(d_b_10, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream10));
    CUDA_CHECK(cudaMemcpyAsync(d_a_11, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream11));
    CUDA_CHECK(cudaMemcpyAsync(d_b_11, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream11));
    CUDA_CHECK(cudaMemcpyAsync(d_a_12, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream12));
    CUDA_CHECK(cudaMemcpyAsync(d_b_12, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream12));
    CUDA_CHECK(cudaMemcpyAsync(d_a_13, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream13));
    CUDA_CHECK(cudaMemcpyAsync(d_b_13, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream13));
    CUDA_CHECK(cudaMemcpyAsync(d_a_14, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream14));
    CUDA_CHECK(cudaMemcpyAsync(d_b_14, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream14));
    CUDA_CHECK(cudaMemcpyAsync(d_a_15, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream15));
    CUDA_CHECK(cudaMemcpyAsync(d_b_15, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream15));
    CUDA_CHECK(cudaMemcpyAsync(d_a_16, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream16));
    CUDA_CHECK(cudaMemcpyAsync(d_b_16, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream16));
    CUDA_CHECK(cudaMemcpyAsync(d_a_17, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream17));
    CUDA_CHECK(cudaMemcpyAsync(d_b_17, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream17));
    CUDA_CHECK(cudaMemcpyAsync(d_a_18, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream18));
    CUDA_CHECK(cudaMemcpyAsync(d_b_18, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream18));
    CUDA_CHECK(cudaMemcpyAsync(d_a_19, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream19));
    CUDA_CHECK(cudaMemcpyAsync(d_b_19, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream19));
    CUDA_CHECK(cudaMemcpyAsync(d_a_20, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream20));
    CUDA_CHECK(cudaMemcpyAsync(d_b_20, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream20));
    CUDA_CHECK(cudaMemcpyAsync(d_a_21, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream21));
    CUDA_CHECK(cudaMemcpyAsync(d_b_21, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream21));
    CUDA_CHECK(cudaMemcpyAsync(d_a_22, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream22));
    CUDA_CHECK(cudaMemcpyAsync(d_b_22, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream22));
    CUDA_CHECK(cudaMemcpyAsync(d_a_23, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream23));
    CUDA_CHECK(cudaMemcpyAsync(d_b_23, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream23));
    CUDA_CHECK(cudaMemcpyAsync(d_a_24, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream24));
    CUDA_CHECK(cudaMemcpyAsync(d_b_24, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream24));
    CUDA_CHECK(cudaMemcpyAsync(d_a_25, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream25));
    CUDA_CHECK(cudaMemcpyAsync(d_b_25, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream25));
    CUDA_CHECK(cudaMemcpyAsync(d_a_26, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream26));
    CUDA_CHECK(cudaMemcpyAsync(d_b_26, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream26));
    CUDA_CHECK(cudaMemcpyAsync(d_a_27, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream27));
    CUDA_CHECK(cudaMemcpyAsync(d_b_27, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream27));
    CUDA_CHECK(cudaMemcpyAsync(d_a_28, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream28));
    CUDA_CHECK(cudaMemcpyAsync(d_b_28, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream28));
    CUDA_CHECK(cudaMemcpyAsync(d_a_29, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream29));
    CUDA_CHECK(cudaMemcpyAsync(d_b_29, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream29));
    CUDA_CHECK(cudaMemcpyAsync(d_a_30, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream30));
    CUDA_CHECK(cudaMemcpyAsync(d_b_30, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream30));
    CUDA_CHECK(cudaMemcpyAsync(d_a_31, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream31));
    CUDA_CHECK(cudaMemcpyAsync(d_b_31, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream31));
    CUDA_CHECK(cudaMemcpyAsync(d_a_32, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream32));
    CUDA_CHECK(cudaMemcpyAsync(d_b_32, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream32));
    CUDA_CHECK(cudaMemcpyAsync(d_a_33, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream33));
    CUDA_CHECK(cudaMemcpyAsync(d_b_33, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream33));
    CUDA_CHECK(cudaMemcpyAsync(d_a_34, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream34));
    CUDA_CHECK(cudaMemcpyAsync(d_b_34, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream34));
    CUDA_CHECK(cudaMemcpyAsync(d_a_35, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream35));
    CUDA_CHECK(cudaMemcpyAsync(d_b_35, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream35));
    CUDA_CHECK(cudaMemcpyAsync(d_a_36, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream36));
    CUDA_CHECK(cudaMemcpyAsync(d_b_36, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream36));
    CUDA_CHECK(cudaMemcpyAsync(d_a_37, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream37));
    CUDA_CHECK(cudaMemcpyAsync(d_b_37, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream37));
    CUDA_CHECK(cudaMemcpyAsync(d_a_38, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream38));
    CUDA_CHECK(cudaMemcpyAsync(d_b_38, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream38));
    CUDA_CHECK(cudaMemcpyAsync(d_a_39, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream39));
    CUDA_CHECK(cudaMemcpyAsync(d_b_39, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream39));
    CUDA_CHECK(cudaMemcpyAsync(d_a_40, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream40));
    CUDA_CHECK(cudaMemcpyAsync(d_b_40, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream40));
    CUDA_CHECK(cudaMemcpyAsync(d_a_41, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream41));
    CUDA_CHECK(cudaMemcpyAsync(d_b_41, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream41));
    CUDA_CHECK(cudaMemcpyAsync(d_a_42, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream42));
    CUDA_CHECK(cudaMemcpyAsync(d_b_42, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream42));
    CUDA_CHECK(cudaMemcpyAsync(d_a_43, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream43));
    CUDA_CHECK(cudaMemcpyAsync(d_b_43, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream43));
    CUDA_CHECK(cudaMemcpyAsync(d_a_44, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream44));
    CUDA_CHECK(cudaMemcpyAsync(d_b_44, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream44));
    CUDA_CHECK(cudaMemcpyAsync(d_a_45, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream45));
    CUDA_CHECK(cudaMemcpyAsync(d_b_45, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream45));
    CUDA_CHECK(cudaMemcpyAsync(d_a_46, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream46));
    CUDA_CHECK(cudaMemcpyAsync(d_b_46, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream46));
    CUDA_CHECK(cudaMemcpyAsync(d_a_47, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream47));
    CUDA_CHECK(cudaMemcpyAsync(d_b_47, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream47));
    CUDA_CHECK(cudaMemcpyAsync(d_a_48, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream48));
    CUDA_CHECK(cudaMemcpyAsync(d_b_48, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream48));
    CUDA_CHECK(cudaMemcpyAsync(d_a_49, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream49));
    CUDA_CHECK(cudaMemcpyAsync(d_b_49, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream49));
    CUDA_CHECK(cudaMemcpyAsync(d_a_50, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream50));
    CUDA_CHECK(cudaMemcpyAsync(d_b_50, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream50));
    CUDA_CHECK(cudaMemcpyAsync(d_a_51, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream51));
    CUDA_CHECK(cudaMemcpyAsync(d_b_51, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream51));
    CUDA_CHECK(cudaMemcpyAsync(d_a_52, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream52));
    CUDA_CHECK(cudaMemcpyAsync(d_b_52, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream52));
    CUDA_CHECK(cudaMemcpyAsync(d_a_53, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream53));
    CUDA_CHECK(cudaMemcpyAsync(d_b_53, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream53));
    CUDA_CHECK(cudaMemcpyAsync(d_a_54, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream54));
    CUDA_CHECK(cudaMemcpyAsync(d_b_54, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream54));
    CUDA_CHECK(cudaMemcpyAsync(d_a_55, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream55));
    CUDA_CHECK(cudaMemcpyAsync(d_b_55, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream55));
    CUDA_CHECK(cudaMemcpyAsync(d_a_56, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream56));
    CUDA_CHECK(cudaMemcpyAsync(d_b_56, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream56));
    CUDA_CHECK(cudaMemcpyAsync(d_a_57, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream57));
    CUDA_CHECK(cudaMemcpyAsync(d_b_57, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream57));
    CUDA_CHECK(cudaMemcpyAsync(d_a_58, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream58));
    CUDA_CHECK(cudaMemcpyAsync(d_b_58, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream58));
    CUDA_CHECK(cudaMemcpyAsync(d_a_59, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream59));
    CUDA_CHECK(cudaMemcpyAsync(d_b_59, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream59));
    CUDA_CHECK(cudaMemcpyAsync(d_a_60, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream60));
    CUDA_CHECK(cudaMemcpyAsync(d_b_60, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream60));
    CUDA_CHECK(cudaMemcpyAsync(d_a_61, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream61));
    CUDA_CHECK(cudaMemcpyAsync(d_b_61, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream61));
    CUDA_CHECK(cudaMemcpyAsync(d_a_62, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream62));
    CUDA_CHECK(cudaMemcpyAsync(d_b_62, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream62));
    CUDA_CHECK(cudaMemcpyAsync(d_a_63, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream63));
    CUDA_CHECK(cudaMemcpyAsync(d_b_63, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream63));
    CUDA_CHECK(cudaMemcpyAsync(d_a_64, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream64));
    CUDA_CHECK(cudaMemcpyAsync(d_b_64, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream64));
    CUDA_CHECK(cudaMemcpyAsync(d_a_65, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream65));
    CUDA_CHECK(cudaMemcpyAsync(d_b_65, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream65));
    CUDA_CHECK(cudaMemcpyAsync(d_a_66, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream66));
    CUDA_CHECK(cudaMemcpyAsync(d_b_66, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream66));
    CUDA_CHECK(cudaMemcpyAsync(d_a_67, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream67));
    CUDA_CHECK(cudaMemcpyAsync(d_b_67, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream67));
    CUDA_CHECK(cudaMemcpyAsync(d_a_68, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream68));
    CUDA_CHECK(cudaMemcpyAsync(d_b_68, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream68));
    CUDA_CHECK(cudaMemcpyAsync(d_a_69, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream69));
    CUDA_CHECK(cudaMemcpyAsync(d_b_69, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream69));
    CUDA_CHECK(cudaMemcpyAsync(d_a_70, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream70));
    CUDA_CHECK(cudaMemcpyAsync(d_b_70, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream70));
    CUDA_CHECK(cudaMemcpyAsync(d_a_71, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream71));
    CUDA_CHECK(cudaMemcpyAsync(d_b_71, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream71));
    CUDA_CHECK(cudaMemcpyAsync(d_a_72, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream72));
    CUDA_CHECK(cudaMemcpyAsync(d_b_72, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream72));
    CUDA_CHECK(cudaMemcpyAsync(d_a_73, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream73));
    CUDA_CHECK(cudaMemcpyAsync(d_b_73, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream73));
    CUDA_CHECK(cudaMemcpyAsync(d_a_74, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream74));
    CUDA_CHECK(cudaMemcpyAsync(d_b_74, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream74));
    CUDA_CHECK(cudaMemcpyAsync(d_a_75, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream75));
    CUDA_CHECK(cudaMemcpyAsync(d_b_75, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream75));
    CUDA_CHECK(cudaMemcpyAsync(d_a_76, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream76));
    CUDA_CHECK(cudaMemcpyAsync(d_b_76, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream76));
    CUDA_CHECK(cudaMemcpyAsync(d_a_77, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream77));
    CUDA_CHECK(cudaMemcpyAsync(d_b_77, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream77));
    CUDA_CHECK(cudaMemcpyAsync(d_a_78, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream78));
    CUDA_CHECK(cudaMemcpyAsync(d_b_78, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream78));
    CUDA_CHECK(cudaMemcpyAsync(d_a_79, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream79));
    CUDA_CHECK(cudaMemcpyAsync(d_b_79, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream79));
    CUDA_CHECK(cudaMemcpyAsync(d_a_80, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream80));
    CUDA_CHECK(cudaMemcpyAsync(d_b_80, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream80));
    CUDA_CHECK(cudaMemcpyAsync(d_a_81, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice, stream81));
    CUDA_CHECK(cudaMemcpyAsync(d_b_81, B.data(), sizeof(data_type) * B.size(), cudaMemcpyHostToDevice, stream81));
    

    // step 3: compute 
    cublasHgemm(cublasH1, transa, transb, n, m, k, &alpha, d_b_1, ldb, d_a_1, lda, &beta, d_c_1, ldc);cublasHgemm(cublasH2, transa, transb, n, m, k, &alpha, d_b_2, ldb, d_a_2, lda, &beta, d_c_2, ldc);cublasHgemm(cublasH3, transa, transb, n, m, k, &alpha, d_b_3, ldb, d_a_3, lda, &beta, d_c_3, ldc);cublasHgemm(cublasH4, transa, transb, n, m, k, &alpha, d_b_4, ldb, d_a_4, lda, &beta, d_c_4, ldc);cublasHgemm(cublasH5, transa, transb, n, m, k, &alpha, d_b_5, ldb, d_a_5, lda, &beta, d_c_5, ldc);cublasHgemm(cublasH6, transa, transb, n, m, k, &alpha, d_b_6, ldb, d_a_6, lda, &beta, d_c_6, ldc);cublasHgemm(cublasH7, transa, transb, n, m, k, &alpha, d_b_7, ldb, d_a_7, lda, &beta, d_c_7, ldc);cublasHgemm(cublasH8, transa, transb, n, m, k, &alpha, d_b_8, ldb, d_a_8, lda, &beta, d_c_8, ldc);cublasHgemm(cublasH9, transa, transb, n, m, k, &alpha, d_b_9, ldb, d_a_9, lda, &beta, d_c_9, ldc);cublasHgemm(cublasH10, transa, transb, n, m, k, &alpha, d_b_10, ldb, d_a_10, lda, &beta, d_c_10, ldc);cublasHgemm(cublasH11, transa, transb, n, m, k, &alpha, d_b_11, ldb, d_a_11, lda, &beta, d_c_11, ldc);cublasHgemm(cublasH12, transa, transb, n, m, k, &alpha, d_b_12, ldb, d_a_12, lda, &beta, d_c_12, ldc);cublasHgemm(cublasH13, transa, transb, n, m, k, &alpha, d_b_13, ldb, d_a_13, lda, &beta, d_c_13, ldc);cublasHgemm(cublasH14, transa, transb, n, m, k, &alpha, d_b_14, ldb, d_a_14, lda, &beta, d_c_14, ldc);cublasHgemm(cublasH15, transa, transb, n, m, k, &alpha, d_b_15, ldb, d_a_15, lda, &beta, d_c_15, ldc);cublasHgemm(cublasH16, transa, transb, n, m, k, &alpha, d_b_16, ldb, d_a_16, lda, &beta, d_c_16, ldc);cublasHgemm(cublasH17, transa, transb, n, m, k, &alpha, d_b_17, ldb, d_a_17, lda, &beta, d_c_17, ldc);cublasHgemm(cublasH18, transa, transb, n, m, k, &alpha, d_b_18, ldb, d_a_18, lda, &beta, d_c_18, ldc);cublasHgemm(cublasH19, transa, transb, n, m, k, &alpha, d_b_19, ldb, d_a_19, lda, &beta, d_c_19, ldc);cublasHgemm(cublasH20, transa, transb, n, m, k, &alpha, d_b_20, ldb, d_a_20, lda, &beta, d_c_20, ldc);cublasHgemm(cublasH21, transa, transb, n, m, k, &alpha, d_b_21, ldb, d_a_21, lda, &beta, d_c_21, ldc);cublasHgemm(cublasH22, transa, transb, n, m, k, &alpha, d_b_22, ldb, d_a_22, lda, &beta, d_c_22, ldc);cublasHgemm(cublasH23, transa, transb, n, m, k, &alpha, d_b_23, ldb, d_a_23, lda, &beta, d_c_23, ldc);cublasHgemm(cublasH24, transa, transb, n, m, k, &alpha, d_b_24, ldb, d_a_24, lda, &beta, d_c_24, ldc);cublasHgemm(cublasH25, transa, transb, n, m, k, &alpha, d_b_25, ldb, d_a_25, lda, &beta, d_c_25, ldc);cublasHgemm(cublasH26, transa, transb, n, m, k, &alpha, d_b_26, ldb, d_a_26, lda, &beta, d_c_26, ldc);cublasHgemm(cublasH27, transa, transb, n, m, k, &alpha, d_b_27, ldb, d_a_27, lda, &beta, d_c_27, ldc);cublasHgemm(cublasH28, transa, transb, n, m, k, &alpha, d_b_28, ldb, d_a_28, lda, &beta, d_c_28, ldc);cublasHgemm(cublasH29, transa, transb, n, m, k, &alpha, d_b_29, ldb, d_a_29, lda, &beta, d_c_29, ldc);cublasHgemm(cublasH30, transa, transb, n, m, k, &alpha, d_b_30, ldb, d_a_30, lda, &beta, d_c_30, ldc);cublasHgemm(cublasH31, transa, transb, n, m, k, &alpha, d_b_31, ldb, d_a_31, lda, &beta, d_c_31, ldc);cublasHgemm(cublasH32, transa, transb, n, m, k, &alpha, d_b_32, ldb, d_a_32, lda, &beta, d_c_32, ldc);cublasHgemm(cublasH33, transa, transb, n, m, k, &alpha, d_b_33, ldb, d_a_33, lda, &beta, d_c_33, ldc);cublasHgemm(cublasH34, transa, transb, n, m, k, &alpha, d_b_34, ldb, d_a_34, lda, &beta, d_c_34, ldc);cublasHgemm(cublasH35, transa, transb, n, m, k, &alpha, d_b_35, ldb, d_a_35, lda, &beta, d_c_35, ldc);cublasHgemm(cublasH36, transa, transb, n, m, k, &alpha, d_b_36, ldb, d_a_36, lda, &beta, d_c_36, ldc);cublasHgemm(cublasH37, transa, transb, n, m, k, &alpha, d_b_37, ldb, d_a_37, lda, &beta, d_c_37, ldc);cublasHgemm(cublasH38, transa, transb, n, m, k, &alpha, d_b_38, ldb, d_a_38, lda, &beta, d_c_38, ldc);cublasHgemm(cublasH39, transa, transb, n, m, k, &alpha, d_b_39, ldb, d_a_39, lda, &beta, d_c_39, ldc);cublasHgemm(cublasH40, transa, transb, n, m, k, &alpha, d_b_40, ldb, d_a_40, lda, &beta, d_c_40, ldc);cublasHgemm(cublasH41, transa, transb, n, m, k, &alpha, d_b_41, ldb, d_a_41, lda, &beta, d_c_41, ldc);cublasHgemm(cublasH42, transa, transb, n, m, k, &alpha, d_b_42, ldb, d_a_42, lda, &beta, d_c_42, ldc);cublasHgemm(cublasH43, transa, transb, n, m, k, &alpha, d_b_43, ldb, d_a_43, lda, &beta, d_c_43, ldc);cublasHgemm(cublasH44, transa, transb, n, m, k, &alpha, d_b_44, ldb, d_a_44, lda, &beta, d_c_44, ldc);cublasHgemm(cublasH45, transa, transb, n, m, k, &alpha, d_b_45, ldb, d_a_45, lda, &beta, d_c_45, ldc);cublasHgemm(cublasH46, transa, transb, n, m, k, &alpha, d_b_46, ldb, d_a_46, lda, &beta, d_c_46, ldc);cublasHgemm(cublasH47, transa, transb, n, m, k, &alpha, d_b_47, ldb, d_a_47, lda, &beta, d_c_47, ldc);cublasHgemm(cublasH48, transa, transb, n, m, k, &alpha, d_b_48, ldb, d_a_48, lda, &beta, d_c_48, ldc);cublasHgemm(cublasH49, transa, transb, n, m, k, &alpha, d_b_49, ldb, d_a_49, lda, &beta, d_c_49, ldc);cublasHgemm(cublasH50, transa, transb, n, m, k, &alpha, d_b_50, ldb, d_a_50, lda, &beta, d_c_50, ldc);cublasHgemm(cublasH51, transa, transb, n, m, k, &alpha, d_b_51, ldb, d_a_51, lda, &beta, d_c_51, ldc);cublasHgemm(cublasH52, transa, transb, n, m, k, &alpha, d_b_52, ldb, d_a_52, lda, &beta, d_c_52, ldc);cublasHgemm(cublasH53, transa, transb, n, m, k, &alpha, d_b_53, ldb, d_a_53, lda, &beta, d_c_53, ldc);cublasHgemm(cublasH54, transa, transb, n, m, k, &alpha, d_b_54, ldb, d_a_54, lda, &beta, d_c_54, ldc);cublasHgemm(cublasH55, transa, transb, n, m, k, &alpha, d_b_55, ldb, d_a_55, lda, &beta, d_c_55, ldc);cublasHgemm(cublasH56, transa, transb, n, m, k, &alpha, d_b_56, ldb, d_a_56, lda, &beta, d_c_56, ldc);cublasHgemm(cublasH57, transa, transb, n, m, k, &alpha, d_b_57, ldb, d_a_57, lda, &beta, d_c_57, ldc);cublasHgemm(cublasH58, transa, transb, n, m, k, &alpha, d_b_58, ldb, d_a_58, lda, &beta, d_c_58, ldc);cublasHgemm(cublasH59, transa, transb, n, m, k, &alpha, d_b_59, ldb, d_a_59, lda, &beta, d_c_59, ldc);cublasHgemm(cublasH60, transa, transb, n, m, k, &alpha, d_b_60, ldb, d_a_60, lda, &beta, d_c_60, ldc);cublasHgemm(cublasH61, transa, transb, n, m, k, &alpha, d_b_61, ldb, d_a_61, lda, &beta, d_c_61, ldc);cublasHgemm(cublasH62, transa, transb, n, m, k, &alpha, d_b_62, ldb, d_a_62, lda, &beta, d_c_62, ldc);cublasHgemm(cublasH63, transa, transb, n, m, k, &alpha, d_b_63, ldb, d_a_63, lda, &beta, d_c_63, ldc);cublasHgemm(cublasH64, transa, transb, n, m, k, &alpha, d_b_64, ldb, d_a_64, lda, &beta, d_c_64, ldc);cublasHgemm(cublasH65, transa, transb, n, m, k, &alpha, d_b_65, ldb, d_a_65, lda, &beta, d_c_65, ldc);cublasHgemm(cublasH66, transa, transb, n, m, k, &alpha, d_b_66, ldb, d_a_66, lda, &beta, d_c_66, ldc);cublasHgemm(cublasH67, transa, transb, n, m, k, &alpha, d_b_67, ldb, d_a_67, lda, &beta, d_c_67, ldc);cublasHgemm(cublasH68, transa, transb, n, m, k, &alpha, d_b_68, ldb, d_a_68, lda, &beta, d_c_68, ldc);cublasHgemm(cublasH69, transa, transb, n, m, k, &alpha, d_b_69, ldb, d_a_69, lda, &beta, d_c_69, ldc);cublasHgemm(cublasH70, transa, transb, n, m, k, &alpha, d_b_70, ldb, d_a_70, lda, &beta, d_c_70, ldc);cublasHgemm(cublasH71, transa, transb, n, m, k, &alpha, d_b_71, ldb, d_a_71, lda, &beta, d_c_71, ldc);cublasHgemm(cublasH72, transa, transb, n, m, k, &alpha, d_b_72, ldb, d_a_72, lda, &beta, d_c_72, ldc);cublasHgemm(cublasH73, transa, transb, n, m, k, &alpha, d_b_73, ldb, d_a_73, lda, &beta, d_c_73, ldc);cublasHgemm(cublasH74, transa, transb, n, m, k, &alpha, d_b_74, ldb, d_a_74, lda, &beta, d_c_74, ldc);cublasHgemm(cublasH75, transa, transb, n, m, k, &alpha, d_b_75, ldb, d_a_75, lda, &beta, d_c_75, ldc);cublasHgemm(cublasH76, transa, transb, n, m, k, &alpha, d_b_76, ldb, d_a_76, lda, &beta, d_c_76, ldc);cublasHgemm(cublasH77, transa, transb, n, m, k, &alpha, d_b_77, ldb, d_a_77, lda, &beta, d_c_77, ldc);cublasHgemm(cublasH78, transa, transb, n, m, k, &alpha, d_b_78, ldb, d_a_78, lda, &beta, d_c_78, ldc);cublasHgemm(cublasH79, transa, transb, n, m, k, &alpha, d_b_79, ldb, d_a_79, lda, &beta, d_c_79, ldc);cublasHgemm(cublasH80, transa, transb, n, m, k, &alpha, d_b_80, ldb, d_a_80, lda, &beta, d_c_80, ldc);cublasHgemm(cublasH81, transa, transb, n, m, k, &alpha, d_b_81, ldb, d_a_81, lda, &beta, d_c_81, ldc);

    // cublasHgemm111<<<1,1,1,stream1>>>(cublasH1, transa, transb, n, m, k, &alpha, d_b_1, ldb, d_a_1, lda, &beta, d_c_1, ldc);
    // cublasHgemm111<<<2,1,1,stream2>>>(cublasH2, transa, transb, n, m, k, &alpha, d_b_2, ldb, d_a_2, lda, &beta, d_c_2, ldc);
    // cublasHgemm111<<<3,1,1,stream3>>>(cublasH3, transa, transb, n, m, k, &alpha, d_b_3, ldb, d_a_3, lda, &beta, d_c_3, ldc);
    // cublasHgemm111<<<4,1,1,stream4>>>(cublasH4, transa, transb, n, m, k, &alpha, d_b_4, ldb, d_a_4, lda, &beta, d_c_4, ldc);
    // cublasHgemm111<<<5,1,1,stream5>>>(cublasH5, transa, transb, n, m, k, &alpha, d_b_5, ldb, d_a_5, lda, &beta, d_c_5, ldc);
    // cublasHgemm111<<<6,1,1,stream6>>>(cublasH6, transa, transb, n, m, k, &alpha, d_b_6, ldb, d_a_6, lda, &beta, d_c_6, ldc);
    // cublasHgemm111<<<7,1,1,stream7>>>(cublasH7, transa, transb, n, m, k, &alpha, d_b_7, ldb, d_a_7, lda, &beta, d_c_7, ldc);
    // cublasHgemm111<<<8,1,1,stream8>>>(cublasH8, transa, transb, n, m, k, &alpha, d_b_8, ldb, d_a_8, lda, &beta, d_c_8, ldc);
    // cublasHgemm111<<<9,1,1,stream9>>>(cublasH9, transa, transb, n, m, k, &alpha, d_b_9, ldb, d_a_9, lda, &beta, d_c_9, ldc);
    // cublasHgemm111<<<10,1,1,stream10>>>(cublasH10, transa, transb, n, m, k, &alpha, d_b_10, ldb, d_a_10, lda, &beta, d_c_10, ldc);
    // cublasHgemm111<<<11,1,1,stream11>>>(cublasH11, transa, transb, n, m, k, &alpha, d_b_11, ldb, d_a_11, lda, &beta, d_c_11, ldc);
    // cublasHgemm111<<<12,1,1,stream12>>>(cublasH12, transa, transb, n, m, k, &alpha, d_b_12, ldb, d_a_12, lda, &beta, d_c_12, ldc);
    // cublasHgemm111<<<13,1,1,stream13>>>(cublasH13, transa, transb, n, m, k, &alpha, d_b_13, ldb, d_a_13, lda, &beta, d_c_13, ldc);
    // cublasHgemm111<<<14,1,1,stream14>>>(cublasH14, transa, transb, n, m, k, &alpha, d_b_14, ldb, d_a_14, lda, &beta, d_c_14, ldc);
    // cublasHgemm111<<<15,1,1,stream15>>>(cublasH15, transa, transb, n, m, k, &alpha, d_b_15, ldb, d_a_15, lda, &beta, d_c_15, ldc);
    // cublasHgemm111<<<16,1,1,stream16>>>(cublasH16, transa, transb, n, m, k, &alpha, d_b_16, ldb, d_a_16, lda, &beta, d_c_16, ldc);
    // cublasHgemm111<<<17,1,1,stream17>>>(cublasH17, transa, transb, n, m, k, &alpha, d_b_17, ldb, d_a_17, lda, &beta, d_c_17, ldc);
    // cublasHgemm111<<<18,1,1,stream18>>>(cublasH18, transa, transb, n, m, k, &alpha, d_b_18, ldb, d_a_18, lda, &beta, d_c_18, ldc);
    // cublasHgemm111<<<19,1,1,stream19>>>(cublasH19, transa, transb, n, m, k, &alpha, d_b_19, ldb, d_a_19, lda, &beta, d_c_19, ldc);
    // cublasHgemm111<<<20,1,1,stream20>>>(cublasH20, transa, transb, n, m, k, &alpha, d_b_20, ldb, d_a_20, lda, &beta, d_c_20, ldc);
    // cublasHgemm111<<<21,1,1,stream21>>>(cublasH21, transa, transb, n, m, k, &alpha, d_b_21, ldb, d_a_21, lda, &beta, d_c_21, ldc);
    // cublasHgemm111<<<22,1,1,stream22>>>(cublasH22, transa, transb, n, m, k, &alpha, d_b_22, ldb, d_a_22, lda, &beta, d_c_22, ldc);
    // cublasHgemm111<<<23,1,1,stream23>>>(cublasH23, transa, transb, n, m, k, &alpha, d_b_23, ldb, d_a_23, lda, &beta, d_c_23, ldc);
    // cublasHgemm111<<<24,1,1,stream24>>>(cublasH24, transa, transb, n, m, k, &alpha, d_b_24, ldb, d_a_24, lda, &beta, d_c_24, ldc);
    // cublasHgemm111<<<25,1,1,stream25>>>(cublasH25, transa, transb, n, m, k, &alpha, d_b_25, ldb, d_a_25, lda, &beta, d_c_25, ldc);
    // cublasHgemm111<<<26,1,1,stream26>>>(cublasH26, transa, transb, n, m, k, &alpha, d_b_26, ldb, d_a_26, lda, &beta, d_c_26, ldc);
    // cublasHgemm111<<<27,1,1,stream27>>>(cublasH27, transa, transb, n, m, k, &alpha, d_b_27, ldb, d_a_27, lda, &beta, d_c_27, ldc);
    // cublasHgemm111<<<28,1,1,stream28>>>(cublasH28, transa, transb, n, m, k, &alpha, d_b_28, ldb, d_a_28, lda, &beta, d_c_28, ldc);
    // cublasHgemm111<<<29,1,1,stream29>>>(cublasH29, transa, transb, n, m, k, &alpha, d_b_29, ldb, d_a_29, lda, &beta, d_c_29, ldc);
    // cublasHgemm111<<<30,1,1,stream30>>>(cublasH30, transa, transb, n, m, k, &alpha, d_b_30, ldb, d_a_30, lda, &beta, d_c_30, ldc);
    // cublasHgemm111<<<31,1,1,stream31>>>(cublasH31, transa, transb, n, m, k, &alpha, d_b_31, ldb, d_a_31, lda, &beta, d_c_31, ldc);
    // cublasHgemm111<<<32,1,1,stream32>>>(cublasH32, transa, transb, n, m, k, &alpha, d_b_32, ldb, d_a_32, lda, &beta, d_c_32, ldc);
    // cublasHgemm111<<<33,1,1,stream33>>>(cublasH33, transa, transb, n, m, k, &alpha, d_b_33, ldb, d_a_33, lda, &beta, d_c_33, ldc);
    // cublasHgemm111<<<34,1,1,stream34>>>(cublasH34, transa, transb, n, m, k, &alpha, d_b_34, ldb, d_a_34, lda, &beta, d_c_34, ldc);
    // cublasHgemm111<<<35,1,1,stream35>>>(cublasH35, transa, transb, n, m, k, &alpha, d_b_35, ldb, d_a_35, lda, &beta, d_c_35, ldc);
    // cublasHgemm111<<<36,1,1,stream36>>>(cublasH36, transa, transb, n, m, k, &alpha, d_b_36, ldb, d_a_36, lda, &beta, d_c_36, ldc);
    // cublasHgemm111<<<37,1,1,stream37>>>(cublasH37, transa, transb, n, m, k, &alpha, d_b_37, ldb, d_a_37, lda, &beta, d_c_37, ldc);
    // cublasHgemm111<<<38,1,1,stream38>>>(cublasH38, transa, transb, n, m, k, &alpha, d_b_38, ldb, d_a_38, lda, &beta, d_c_38, ldc);
    // cublasHgemm111<<<39,1,1,stream39>>>(cublasH39, transa, transb, n, m, k, &alpha, d_b_39, ldb, d_a_39, lda, &beta, d_c_39, ldc);
    // cublasHgemm111<<<40,1,1,stream40>>>(cublasH40, transa, transb, n, m, k, &alpha, d_b_40, ldb, d_a_40, lda, &beta, d_c_40, ldc);
    // cublasHgemm111<<<41,1,1,stream41>>>(cublasH41, transa, transb, n, m, k, &alpha, d_b_41, ldb, d_a_41, lda, &beta, d_c_41, ldc);
    // cublasHgemm111<<<42,1,1,stream42>>>(cublasH42, transa, transb, n, m, k, &alpha, d_b_42, ldb, d_a_42, lda, &beta, d_c_42, ldc);
    // cublasHgemm111<<<43,1,1,stream43>>>(cublasH43, transa, transb, n, m, k, &alpha, d_b_43, ldb, d_a_43, lda, &beta, d_c_43, ldc);
    // cublasHgemm111<<<44,1,1,stream44>>>(cublasH44, transa, transb, n, m, k, &alpha, d_b_44, ldb, d_a_44, lda, &beta, d_c_44, ldc);
    // cublasHgemm111<<<43,1,1,stream45>>>(cublasH45, transa, transb, n, m, k, &alpha, d_b_45, ldb, d_a_45, lda, &beta, d_c_45, ldc);
    // cublasHgemm111<<<42,1,1,stream46>>>(cublasH46, transa, transb, n, m, k, &alpha, d_b_46, ldb, d_a_46, lda, &beta, d_c_46, ldc);
    // cublasHgemm111<<<41,1,1,stream47>>>(cublasH47, transa, transb, n, m, k, &alpha, d_b_47, ldb, d_a_47, lda, &beta, d_c_47, ldc);
    // cublasHgemm111<<<40,1,1,stream48>>>(cublasH48, transa, transb, n, m, k, &alpha, d_b_48, ldb, d_a_48, lda, &beta, d_c_48, ldc);
    // cublasHgemm111<<<39,1,1,stream49>>>(cublasH49, transa, transb, n, m, k, &alpha, d_b_49, ldb, d_a_49, lda, &beta, d_c_49, ldc);
    // cublasHgemm111<<<38,1,1,stream50>>>(cublasH50, transa, transb, n, m, k, &alpha, d_b_50, ldb, d_a_50, lda, &beta, d_c_50, ldc);
    // cublasHgemm111<<<37,1,1,stream51>>>(cublasH51, transa, transb, n, m, k, &alpha, d_b_51, ldb, d_a_51, lda, &beta, d_c_51, ldc);
    // cublasHgemm111<<<36,1,1,stream52>>>(cublasH52, transa, transb, n, m, k, &alpha, d_b_52, ldb, d_a_52, lda, &beta, d_c_52, ldc);
    // cublasHgemm111<<<35,1,1,stream53>>>(cublasH53, transa, transb, n, m, k, &alpha, d_b_53, ldb, d_a_53, lda, &beta, d_c_53, ldc);
    // cublasHgemm111<<<34,1,1,stream54>>>(cublasH54, transa, transb, n, m, k, &alpha, d_b_54, ldb, d_a_54, lda, &beta, d_c_54, ldc);
    // cublasHgemm111<<<33,1,1,stream55>>>(cublasH55, transa, transb, n, m, k, &alpha, d_b_55, ldb, d_a_55, lda, &beta, d_c_55, ldc);
    // cublasHgemm111<<<32,1,1,stream56>>>(cublasH56, transa, transb, n, m, k, &alpha, d_b_56, ldb, d_a_56, lda, &beta, d_c_56, ldc);
    // cublasHgemm111<<<31,1,1,stream57>>>(cublasH57, transa, transb, n, m, k, &alpha, d_b_57, ldb, d_a_57, lda, &beta, d_c_57, ldc);
    // cublasHgemm111<<<30,1,1,stream58>>>(cublasH58, transa, transb, n, m, k, &alpha, d_b_58, ldb, d_a_58, lda, &beta, d_c_58, ldc);
    // cublasHgemm111<<<29,1,1,stream59>>>(cublasH59, transa, transb, n, m, k, &alpha, d_b_59, ldb, d_a_59, lda, &beta, d_c_59, ldc);
    // cublasHgemm111<<<28,1,1,stream60>>>(cublasH60, transa, transb, n, m, k, &alpha, d_b_60, ldb, d_a_60, lda, &beta, d_c_60, ldc);
    // cublasHgemm111<<<27,1,1,stream61>>>(cublasH61, transa, transb, n, m, k, &alpha, d_b_61, ldb, d_a_61, lda, &beta, d_c_61, ldc);
    // cublasHgemm111<<<26,1,1,stream62>>>(cublasH62, transa, transb, n, m, k, &alpha, d_b_62, ldb, d_a_62, lda, &beta, d_c_62, ldc);
    // cublasHgemm111<<<25,1,1,stream63>>>(cublasH63, transa, transb, n, m, k, &alpha, d_b_63, ldb, d_a_63, lda, &beta, d_c_63, ldc);
    // cublasHgemm111<<<24,1,1,stream64>>>(cublasH64, transa, transb, n, m, k, &alpha, d_b_64, ldb, d_a_64, lda, &beta, d_c_64, ldc);
    // cublasHgemm111<<<23,1,1,stream65>>>(cublasH65, transa, transb, n, m, k, &alpha, d_b_65, ldb, d_a_65, lda, &beta, d_c_65, ldc);
    // cublasHgemm111<<<22,1,1,stream66>>>(cublasH66, transa, transb, n, m, k, &alpha, d_b_66, ldb, d_a_66, lda, &beta, d_c_66, ldc);
    // cublasHgemm111<<<21,1,1,stream67>>>(cublasH67, transa, transb, n, m, k, &alpha, d_b_67, ldb, d_a_67, lda, &beta, d_c_67, ldc);
    // cublasHgemm111<<<20,1,1,stream68>>>(cublasH68, transa, transb, n, m, k, &alpha, d_b_68, ldb, d_a_68, lda, &beta, d_c_68, ldc);
    // cublasHgemm111<<<19,1,1,stream69>>>(cublasH69, transa, transb, n, m, k, &alpha, d_b_69, ldb, d_a_69, lda, &beta, d_c_69, ldc);
    // cublasHgemm111<<<18,1,1,stream70>>>(cublasH70, transa, transb, n, m, k, &alpha, d_b_70, ldb, d_a_70, lda, &beta, d_c_70, ldc);
    // cublasHgemm111<<<17,1,1,stream71>>>(cublasH71, transa, transb, n, m, k, &alpha, d_b_71, ldb, d_a_71, lda, &beta, d_c_71, ldc);
    // cublasHgemm111<<<16,1,1,stream72>>>(cublasH72, transa, transb, n, m, k, &alpha, d_b_72, ldb, d_a_72, lda, &beta, d_c_72, ldc);
    // cublasHgemm111<<<15,1,1,stream73>>>(cublasH73, transa, transb, n, m, k, &alpha, d_b_73, ldb, d_a_73, lda, &beta, d_c_73, ldc);
    // cublasHgemm111<<<14,1,1,stream74>>>(cublasH74, transa, transb, n, m, k, &alpha, d_b_74, ldb, d_a_74, lda, &beta, d_c_74, ldc);
    // cublasHgemm111<<<13,1,1,stream75>>>(cublasH75, transa, transb, n, m, k, &alpha, d_b_75, ldb, d_a_75, lda, &beta, d_c_75, ldc);
    // cublasHgemm111<<<12,1,1,stream76>>>(cublasH76, transa, transb, n, m, k, &alpha, d_b_76, ldb, d_a_76, lda, &beta, d_c_76, ldc);
    // cublasHgemm111<<<11,1,1,stream77>>>(cublasH77, transa, transb, n, m, k, &alpha, d_b_77, ldb, d_a_77, lda, &beta, d_c_77, ldc);
    // cublasHgemm111<<<10,1,1,stream78>>>(cublasH78, transa, transb, n, m, k, &alpha, d_b_78, ldb, d_a_78, lda, &beta, d_c_78, ldc);
    // cublasHgemm111<<<9,1,1,stream79>>>(cublasH79, transa, transb, n, m, k, &alpha, d_b_79, ldb, d_a_79, lda, &beta, d_c_79, ldc);
    // cublasHgemm111<<<8,1,1,stream80>>>(cublasH80, transa, transb, n, m, k, &alpha, d_b_80, ldb, d_a_80, lda, &beta, d_c_80, ldc);
    // cublasHgemm111<<<7,1,1,stream81>>>(cublasH81, transa, transb, n, m, k, &alpha, d_b_81, ldb, d_a_81, lda, &beta, d_c_81, ldc);
    


    // step 4: copy data to host 
    // CUDA_CHECK(cudaMemcpyAsync(C.data(), d_c_1, sizeof(data_type) * C.size(), cudaMemcpyDeviceToHost,
    //                            stream));

    // cudaStreamSynchronize 71 streams
    CUDA_CHECK(cudaStreamSynchronize(stream1));CUDA_CHECK(cudaStreamSynchronize(stream2));CUDA_CHECK(cudaStreamSynchronize(stream3));CUDA_CHECK(cudaStreamSynchronize(stream4));CUDA_CHECK(cudaStreamSynchronize(stream5));CUDA_CHECK(cudaStreamSynchronize(stream6));CUDA_CHECK(cudaStreamSynchronize(stream7));CUDA_CHECK(cudaStreamSynchronize(stream8));CUDA_CHECK(cudaStreamSynchronize(stream9));CUDA_CHECK(cudaStreamSynchronize(stream10));CUDA_CHECK(cudaStreamSynchronize(stream11));CUDA_CHECK(cudaStreamSynchronize(stream12));CUDA_CHECK(cudaStreamSynchronize(stream13));CUDA_CHECK(cudaStreamSynchronize(stream14));CUDA_CHECK(cudaStreamSynchronize(stream15));CUDA_CHECK(cudaStreamSynchronize(stream16));CUDA_CHECK(cudaStreamSynchronize(stream17));CUDA_CHECK(cudaStreamSynchronize(stream18));CUDA_CHECK(cudaStreamSynchronize(stream19));CUDA_CHECK(cudaStreamSynchronize(stream20));CUDA_CHECK(cudaStreamSynchronize(stream21));CUDA_CHECK(cudaStreamSynchronize(stream22));CUDA_CHECK(cudaStreamSynchronize(stream23));CUDA_CHECK(cudaStreamSynchronize(stream24));CUDA_CHECK(cudaStreamSynchronize(stream25));CUDA_CHECK(cudaStreamSynchronize(stream26));CUDA_CHECK(cudaStreamSynchronize(stream27));CUDA_CHECK(cudaStreamSynchronize(stream28));CUDA_CHECK(cudaStreamSynchronize(stream29));CUDA_CHECK(cudaStreamSynchronize(stream30));CUDA_CHECK(cudaStreamSynchronize(stream31));CUDA_CHECK(cudaStreamSynchronize(stream32));CUDA_CHECK(cudaStreamSynchronize(stream33));CUDA_CHECK(cudaStreamSynchronize(stream34));CUDA_CHECK(cudaStreamSynchronize(stream35));CUDA_CHECK(cudaStreamSynchronize(stream36));CUDA_CHECK(cudaStreamSynchronize(stream37));CUDA_CHECK(cudaStreamSynchronize(stream38));CUDA_CHECK(cudaStreamSynchronize(stream39));CUDA_CHECK(cudaStreamSynchronize(stream40));CUDA_CHECK(cudaStreamSynchronize(stream41));CUDA_CHECK(cudaStreamSynchronize(stream42));CUDA_CHECK(cudaStreamSynchronize(stream43));CUDA_CHECK(cudaStreamSynchronize(stream44));CUDA_CHECK(cudaStreamSynchronize(stream45));CUDA_CHECK(cudaStreamSynchronize(stream46));CUDA_CHECK(cudaStreamSynchronize(stream47));CUDA_CHECK(cudaStreamSynchronize(stream48));CUDA_CHECK(cudaStreamSynchronize(stream49));CUDA_CHECK(cudaStreamSynchronize(stream50));CUDA_CHECK(cudaStreamSynchronize(stream51));CUDA_CHECK(cudaStreamSynchronize(stream52));CUDA_CHECK(cudaStreamSynchronize(stream53));CUDA_CHECK(cudaStreamSynchronize(stream54));CUDA_CHECK(cudaStreamSynchronize(stream55));CUDA_CHECK(cudaStreamSynchronize(stream56));CUDA_CHECK(cudaStreamSynchronize(stream57));CUDA_CHECK(cudaStreamSynchronize(stream58));CUDA_CHECK(cudaStreamSynchronize(stream59));CUDA_CHECK(cudaStreamSynchronize(stream60));CUDA_CHECK(cudaStreamSynchronize(stream61));CUDA_CHECK(cudaStreamSynchronize(stream62));CUDA_CHECK(cudaStreamSynchronize(stream63));CUDA_CHECK(cudaStreamSynchronize(stream64));CUDA_CHECK(cudaStreamSynchronize(stream65));CUDA_CHECK(cudaStreamSynchronize(stream66));CUDA_CHECK(cudaStreamSynchronize(stream67));CUDA_CHECK(cudaStreamSynchronize(stream68));CUDA_CHECK(cudaStreamSynchronize(stream69));CUDA_CHECK(cudaStreamSynchronize(stream70));CUDA_CHECK(cudaStreamSynchronize(stream71));CUDA_CHECK(cudaStreamSynchronize(stream72));CUDA_CHECK(cudaStreamSynchronize(stream73));CUDA_CHECK(cudaStreamSynchronize(stream74));CUDA_CHECK(cudaStreamSynchronize(stream75));CUDA_CHECK(cudaStreamSynchronize(stream76));CUDA_CHECK(cudaStreamSynchronize(stream77));CUDA_CHECK(cudaStreamSynchronize(stream78));CUDA_CHECK(cudaStreamSynchronize(stream79));CUDA_CHECK(cudaStreamSynchronize(stream80));CUDA_CHECK(cudaStreamSynchronize(stream81));

    //
    //   C = | 19.0 | 22.0 |
    //       | 43.0 | 50.0 |
    //
    
    // free resources 
    CUDA_CHECK(cudaFree(d_a_1));
    CUDA_CHECK(cudaFree(d_b_1));
    CUDA_CHECK(cudaFree(d_c_1));
    CUDA_CHECK(cudaFree(d_a_2));
    CUDA_CHECK(cudaFree(d_b_2));
    CUDA_CHECK(cudaFree(d_c_2));
    CUDA_CHECK(cudaFree(d_a_3));
    CUDA_CHECK(cudaFree(d_b_3));
    CUDA_CHECK(cudaFree(d_c_3));
    CUDA_CHECK(cudaFree(d_a_4));
    CUDA_CHECK(cudaFree(d_b_4));
    CUDA_CHECK(cudaFree(d_c_4));
    CUDA_CHECK(cudaFree(d_a_5));
    CUDA_CHECK(cudaFree(d_b_5));
    CUDA_CHECK(cudaFree(d_c_5));
    CUDA_CHECK(cudaFree(d_a_6));
    CUDA_CHECK(cudaFree(d_b_6));
    CUDA_CHECK(cudaFree(d_c_6));
    CUDA_CHECK(cudaFree(d_a_7));
    CUDA_CHECK(cudaFree(d_b_7));
    CUDA_CHECK(cudaFree(d_c_7));
    CUDA_CHECK(cudaFree(d_a_8));
    CUDA_CHECK(cudaFree(d_b_8));
    CUDA_CHECK(cudaFree(d_c_8));
    CUDA_CHECK(cudaFree(d_a_9));
    CUDA_CHECK(cudaFree(d_b_9));
    CUDA_CHECK(cudaFree(d_c_9));
    CUDA_CHECK(cudaFree(d_a_10));
    CUDA_CHECK(cudaFree(d_b_10));
    CUDA_CHECK(cudaFree(d_c_10));
    CUDA_CHECK(cudaFree(d_a_11));
    CUDA_CHECK(cudaFree(d_b_11));
    CUDA_CHECK(cudaFree(d_c_11));
    CUDA_CHECK(cudaFree(d_a_12));
    CUDA_CHECK(cudaFree(d_b_12));
    CUDA_CHECK(cudaFree(d_c_12));
    CUDA_CHECK(cudaFree(d_a_13));
    CUDA_CHECK(cudaFree(d_b_13));
    CUDA_CHECK(cudaFree(d_c_13));
    CUDA_CHECK(cudaFree(d_a_14));
    CUDA_CHECK(cudaFree(d_b_14));
    CUDA_CHECK(cudaFree(d_c_14));
    CUDA_CHECK(cudaFree(d_a_15));
    CUDA_CHECK(cudaFree(d_b_15));
    CUDA_CHECK(cudaFree(d_c_15));
    CUDA_CHECK(cudaFree(d_a_16));
    CUDA_CHECK(cudaFree(d_b_16));
    CUDA_CHECK(cudaFree(d_c_16));
    CUDA_CHECK(cudaFree(d_a_17));
    CUDA_CHECK(cudaFree(d_b_17));
    CUDA_CHECK(cudaFree(d_c_17));
    CUDA_CHECK(cudaFree(d_a_18));
    CUDA_CHECK(cudaFree(d_b_18));
    CUDA_CHECK(cudaFree(d_c_18));
    CUDA_CHECK(cudaFree(d_a_19));
    CUDA_CHECK(cudaFree(d_b_19));
    CUDA_CHECK(cudaFree(d_c_19));
    CUDA_CHECK(cudaFree(d_a_20));
    CUDA_CHECK(cudaFree(d_b_20));
    CUDA_CHECK(cudaFree(d_c_20));
    CUDA_CHECK(cudaFree(d_a_21));
    CUDA_CHECK(cudaFree(d_b_21));
    CUDA_CHECK(cudaFree(d_c_21));
    CUDA_CHECK(cudaFree(d_a_22));
    CUDA_CHECK(cudaFree(d_b_22));
    CUDA_CHECK(cudaFree(d_c_22));
    CUDA_CHECK(cudaFree(d_a_23));
    CUDA_CHECK(cudaFree(d_b_23));
    CUDA_CHECK(cudaFree(d_c_23));
    CUDA_CHECK(cudaFree(d_a_24));
    CUDA_CHECK(cudaFree(d_b_24));
    CUDA_CHECK(cudaFree(d_c_24));
    CUDA_CHECK(cudaFree(d_a_25));
    CUDA_CHECK(cudaFree(d_b_25));
    CUDA_CHECK(cudaFree(d_c_25));
    CUDA_CHECK(cudaFree(d_a_26));
    CUDA_CHECK(cudaFree(d_b_26));
    CUDA_CHECK(cudaFree(d_c_26));
    CUDA_CHECK(cudaFree(d_a_27));
    CUDA_CHECK(cudaFree(d_b_27));
    CUDA_CHECK(cudaFree(d_c_27));
    CUDA_CHECK(cudaFree(d_a_28));
    CUDA_CHECK(cudaFree(d_b_28));
    CUDA_CHECK(cudaFree(d_c_28));
    CUDA_CHECK(cudaFree(d_a_29));
    CUDA_CHECK(cudaFree(d_b_29));
    CUDA_CHECK(cudaFree(d_c_29));
    CUDA_CHECK(cudaFree(d_a_30));
    CUDA_CHECK(cudaFree(d_b_30));
    CUDA_CHECK(cudaFree(d_c_30));
    CUDA_CHECK(cudaFree(d_a_31));
    CUDA_CHECK(cudaFree(d_b_31));
    CUDA_CHECK(cudaFree(d_c_31));
    CUDA_CHECK(cudaFree(d_a_32));
    CUDA_CHECK(cudaFree(d_b_32));
    CUDA_CHECK(cudaFree(d_c_32));
    CUDA_CHECK(cudaFree(d_a_33));
    CUDA_CHECK(cudaFree(d_b_33));
    CUDA_CHECK(cudaFree(d_c_33));
    CUDA_CHECK(cudaFree(d_a_34));
    CUDA_CHECK(cudaFree(d_b_34));
    CUDA_CHECK(cudaFree(d_c_34));
    CUDA_CHECK(cudaFree(d_a_35));
    CUDA_CHECK(cudaFree(d_b_35));
    CUDA_CHECK(cudaFree(d_c_35));
    CUDA_CHECK(cudaFree(d_a_36));
    CUDA_CHECK(cudaFree(d_b_36));
    CUDA_CHECK(cudaFree(d_c_36));
    CUDA_CHECK(cudaFree(d_a_37));
    CUDA_CHECK(cudaFree(d_b_37));
    CUDA_CHECK(cudaFree(d_c_37));
    CUDA_CHECK(cudaFree(d_a_38));
    CUDA_CHECK(cudaFree(d_b_38));
    CUDA_CHECK(cudaFree(d_c_38));
    CUDA_CHECK(cudaFree(d_a_39));
    CUDA_CHECK(cudaFree(d_b_39));
    CUDA_CHECK(cudaFree(d_c_39));
    CUDA_CHECK(cudaFree(d_a_40));
    CUDA_CHECK(cudaFree(d_b_40));
    CUDA_CHECK(cudaFree(d_c_40));
    CUDA_CHECK(cudaFree(d_a_41));
    CUDA_CHECK(cudaFree(d_b_41));
    CUDA_CHECK(cudaFree(d_c_41));
    CUDA_CHECK(cudaFree(d_a_42));
    CUDA_CHECK(cudaFree(d_b_42));
    CUDA_CHECK(cudaFree(d_c_42));
    CUDA_CHECK(cudaFree(d_a_43));
    CUDA_CHECK(cudaFree(d_b_43));
    CUDA_CHECK(cudaFree(d_c_43));
    CUDA_CHECK(cudaFree(d_a_44));
    CUDA_CHECK(cudaFree(d_b_44));
    CUDA_CHECK(cudaFree(d_c_44));
    CUDA_CHECK(cudaFree(d_a_45));
    CUDA_CHECK(cudaFree(d_b_45));
    CUDA_CHECK(cudaFree(d_c_45));
    CUDA_CHECK(cudaFree(d_a_46));
    CUDA_CHECK(cudaFree(d_b_46));
    CUDA_CHECK(cudaFree(d_c_46));
    CUDA_CHECK(cudaFree(d_a_47));
    CUDA_CHECK(cudaFree(d_b_47));
    CUDA_CHECK(cudaFree(d_c_47));
    CUDA_CHECK(cudaFree(d_a_48));
    CUDA_CHECK(cudaFree(d_b_48));
    CUDA_CHECK(cudaFree(d_c_48));
    CUDA_CHECK(cudaFree(d_a_49));
    CUDA_CHECK(cudaFree(d_b_49));
    CUDA_CHECK(cudaFree(d_c_49));
    CUDA_CHECK(cudaFree(d_a_50));
    CUDA_CHECK(cudaFree(d_b_50));
    CUDA_CHECK(cudaFree(d_c_50));
    CUDA_CHECK(cudaFree(d_a_51));
    CUDA_CHECK(cudaFree(d_b_51));
    CUDA_CHECK(cudaFree(d_c_51));
    CUDA_CHECK(cudaFree(d_a_52));
    CUDA_CHECK(cudaFree(d_b_52));
    CUDA_CHECK(cudaFree(d_c_52));
    CUDA_CHECK(cudaFree(d_a_53));
    CUDA_CHECK(cudaFree(d_b_53));
    CUDA_CHECK(cudaFree(d_c_53));
    CUDA_CHECK(cudaFree(d_a_54));
    CUDA_CHECK(cudaFree(d_b_54));
    CUDA_CHECK(cudaFree(d_c_54));
    CUDA_CHECK(cudaFree(d_a_55));
    CUDA_CHECK(cudaFree(d_b_55));
    CUDA_CHECK(cudaFree(d_c_55));
    CUDA_CHECK(cudaFree(d_a_56));
    CUDA_CHECK(cudaFree(d_b_56));
    CUDA_CHECK(cudaFree(d_c_56));
    CUDA_CHECK(cudaFree(d_a_57));
    CUDA_CHECK(cudaFree(d_b_57));
    CUDA_CHECK(cudaFree(d_c_57));
    CUDA_CHECK(cudaFree(d_a_58));
    CUDA_CHECK(cudaFree(d_b_58));
    CUDA_CHECK(cudaFree(d_c_58));
    CUDA_CHECK(cudaFree(d_a_59));
    CUDA_CHECK(cudaFree(d_b_59));
    CUDA_CHECK(cudaFree(d_c_59));
    CUDA_CHECK(cudaFree(d_a_60));
    CUDA_CHECK(cudaFree(d_b_60));
    CUDA_CHECK(cudaFree(d_c_60));
    CUDA_CHECK(cudaFree(d_a_61));
    CUDA_CHECK(cudaFree(d_b_61));
    CUDA_CHECK(cudaFree(d_c_61));
    CUDA_CHECK(cudaFree(d_a_62));
    CUDA_CHECK(cudaFree(d_b_62));
    CUDA_CHECK(cudaFree(d_c_62));
    CUDA_CHECK(cudaFree(d_a_63));
    CUDA_CHECK(cudaFree(d_b_63));
    CUDA_CHECK(cudaFree(d_c_63));
    CUDA_CHECK(cudaFree(d_a_64));
    CUDA_CHECK(cudaFree(d_b_64));
    CUDA_CHECK(cudaFree(d_c_64));
    CUDA_CHECK(cudaFree(d_a_65));
    CUDA_CHECK(cudaFree(d_b_65));
    CUDA_CHECK(cudaFree(d_c_65));
    CUDA_CHECK(cudaFree(d_a_66));
    CUDA_CHECK(cudaFree(d_b_66));
    CUDA_CHECK(cudaFree(d_c_66));
    CUDA_CHECK(cudaFree(d_a_67));
    CUDA_CHECK(cudaFree(d_b_67));
    CUDA_CHECK(cudaFree(d_c_67));
    CUDA_CHECK(cudaFree(d_a_68));
    CUDA_CHECK(cudaFree(d_b_68));
    CUDA_CHECK(cudaFree(d_c_68));
    CUDA_CHECK(cudaFree(d_a_69));
    CUDA_CHECK(cudaFree(d_b_69));
    CUDA_CHECK(cudaFree(d_c_69));
    CUDA_CHECK(cudaFree(d_a_70));
    CUDA_CHECK(cudaFree(d_b_70));
    CUDA_CHECK(cudaFree(d_c_70));
    CUDA_CHECK(cudaFree(d_a_71));
    CUDA_CHECK(cudaFree(d_b_71));
    CUDA_CHECK(cudaFree(d_c_71));
    CUDA_CHECK(cudaFree(d_a_72));
    CUDA_CHECK(cudaFree(d_b_72));
    CUDA_CHECK(cudaFree(d_c_72));
    CUDA_CHECK(cudaFree(d_a_73));
    CUDA_CHECK(cudaFree(d_b_73));
    CUDA_CHECK(cudaFree(d_c_73));
    CUDA_CHECK(cudaFree(d_a_74));
    CUDA_CHECK(cudaFree(d_b_74));
    CUDA_CHECK(cudaFree(d_c_74));
    CUDA_CHECK(cudaFree(d_a_75));
    CUDA_CHECK(cudaFree(d_b_75));
    CUDA_CHECK(cudaFree(d_c_75));
    CUDA_CHECK(cudaFree(d_a_76));
    CUDA_CHECK(cudaFree(d_b_76));
    CUDA_CHECK(cudaFree(d_c_76));
    CUDA_CHECK(cudaFree(d_a_77));
    CUDA_CHECK(cudaFree(d_b_77));
    CUDA_CHECK(cudaFree(d_c_77));
    CUDA_CHECK(cudaFree(d_a_78));
    CUDA_CHECK(cudaFree(d_b_78));
    CUDA_CHECK(cudaFree(d_c_78));
    CUDA_CHECK(cudaFree(d_a_79));
    CUDA_CHECK(cudaFree(d_b_79));
    CUDA_CHECK(cudaFree(d_c_79));
    CUDA_CHECK(cudaFree(d_a_80));
    CUDA_CHECK(cudaFree(d_b_80));
    CUDA_CHECK(cudaFree(d_c_80));
    CUDA_CHECK(cudaFree(d_a_81));
    CUDA_CHECK(cudaFree(d_b_81));
    CUDA_CHECK(cudaFree(d_c_81));

    CUBLAS_CHECK(cublasDestroy(cublasH1));CUBLAS_CHECK(cublasDestroy(cublasH2));CUBLAS_CHECK(cublasDestroy(cublasH3));CUBLAS_CHECK(cublasDestroy(cublasH4));CUBLAS_CHECK(cublasDestroy(cublasH5));CUBLAS_CHECK(cublasDestroy(cublasH6));CUBLAS_CHECK(cublasDestroy(cublasH7));CUBLAS_CHECK(cublasDestroy(cublasH8));CUBLAS_CHECK(cublasDestroy(cublasH9));CUBLAS_CHECK(cublasDestroy(cublasH10));CUBLAS_CHECK(cublasDestroy(cublasH11));CUBLAS_CHECK(cublasDestroy(cublasH12));CUBLAS_CHECK(cublasDestroy(cublasH13));CUBLAS_CHECK(cublasDestroy(cublasH14));CUBLAS_CHECK(cublasDestroy(cublasH15));CUBLAS_CHECK(cublasDestroy(cublasH16));CUBLAS_CHECK(cublasDestroy(cublasH17));CUBLAS_CHECK(cublasDestroy(cublasH18));CUBLAS_CHECK(cublasDestroy(cublasH19));CUBLAS_CHECK(cublasDestroy(cublasH20));CUBLAS_CHECK(cublasDestroy(cublasH21));CUBLAS_CHECK(cublasDestroy(cublasH22));CUBLAS_CHECK(cublasDestroy(cublasH23));CUBLAS_CHECK(cublasDestroy(cublasH24));CUBLAS_CHECK(cublasDestroy(cublasH25));CUBLAS_CHECK(cublasDestroy(cublasH26));CUBLAS_CHECK(cublasDestroy(cublasH27));CUBLAS_CHECK(cublasDestroy(cublasH28));CUBLAS_CHECK(cublasDestroy(cublasH29));CUBLAS_CHECK(cublasDestroy(cublasH30));CUBLAS_CHECK(cublasDestroy(cublasH31));CUBLAS_CHECK(cublasDestroy(cublasH32));CUBLAS_CHECK(cublasDestroy(cublasH33));CUBLAS_CHECK(cublasDestroy(cublasH34));CUBLAS_CHECK(cublasDestroy(cublasH35));CUBLAS_CHECK(cublasDestroy(cublasH36));CUBLAS_CHECK(cublasDestroy(cublasH37));CUBLAS_CHECK(cublasDestroy(cublasH38));CUBLAS_CHECK(cublasDestroy(cublasH39));CUBLAS_CHECK(cublasDestroy(cublasH40));CUBLAS_CHECK(cublasDestroy(cublasH41));CUBLAS_CHECK(cublasDestroy(cublasH42));CUBLAS_CHECK(cublasDestroy(cublasH43));CUBLAS_CHECK(cublasDestroy(cublasH44));CUBLAS_CHECK(cublasDestroy(cublasH45));CUBLAS_CHECK(cublasDestroy(cublasH46));CUBLAS_CHECK(cublasDestroy(cublasH47));CUBLAS_CHECK(cublasDestroy(cublasH48));CUBLAS_CHECK(cublasDestroy(cublasH49));CUBLAS_CHECK(cublasDestroy(cublasH50));CUBLAS_CHECK(cublasDestroy(cublasH51));CUBLAS_CHECK(cublasDestroy(cublasH52));CUBLAS_CHECK(cublasDestroy(cublasH53));CUBLAS_CHECK(cublasDestroy(cublasH54));CUBLAS_CHECK(cublasDestroy(cublasH55));CUBLAS_CHECK(cublasDestroy(cublasH56));CUBLAS_CHECK(cublasDestroy(cublasH57));CUBLAS_CHECK(cublasDestroy(cublasH58));CUBLAS_CHECK(cublasDestroy(cublasH59));CUBLAS_CHECK(cublasDestroy(cublasH60));CUBLAS_CHECK(cublasDestroy(cublasH61));CUBLAS_CHECK(cublasDestroy(cublasH62));CUBLAS_CHECK(cublasDestroy(cublasH63));CUBLAS_CHECK(cublasDestroy(cublasH64));CUBLAS_CHECK(cublasDestroy(cublasH65));CUBLAS_CHECK(cublasDestroy(cublasH66));CUBLAS_CHECK(cublasDestroy(cublasH67));CUBLAS_CHECK(cublasDestroy(cublasH68));CUBLAS_CHECK(cublasDestroy(cublasH69));CUBLAS_CHECK(cublasDestroy(cublasH70));CUBLAS_CHECK(cublasDestroy(cublasH71));CUBLAS_CHECK(cublasDestroy(cublasH72));CUBLAS_CHECK(cublasDestroy(cublasH73));CUBLAS_CHECK(cublasDestroy(cublasH74));CUBLAS_CHECK(cublasDestroy(cublasH75));CUBLAS_CHECK(cublasDestroy(cublasH76));CUBLAS_CHECK(cublasDestroy(cublasH77));CUBLAS_CHECK(cublasDestroy(cublasH78));CUBLAS_CHECK(cublasDestroy(cublasH79));CUBLAS_CHECK(cublasDestroy(cublasH80));CUBLAS_CHECK(cublasDestroy(cublasH81));

    CUDA_CHECK(cudaStreamDestroy(stream1));CUDA_CHECK(cudaStreamDestroy(stream2));CUDA_CHECK(cudaStreamDestroy(stream3));CUDA_CHECK(cudaStreamDestroy(stream4));CUDA_CHECK(cudaStreamDestroy(stream5));CUDA_CHECK(cudaStreamDestroy(stream6));CUDA_CHECK(cudaStreamDestroy(stream7));CUDA_CHECK(cudaStreamDestroy(stream8));CUDA_CHECK(cudaStreamDestroy(stream9));CUDA_CHECK(cudaStreamDestroy(stream10));CUDA_CHECK(cudaStreamDestroy(stream11));CUDA_CHECK(cudaStreamDestroy(stream12));CUDA_CHECK(cudaStreamDestroy(stream13));CUDA_CHECK(cudaStreamDestroy(stream14));CUDA_CHECK(cudaStreamDestroy(stream15));CUDA_CHECK(cudaStreamDestroy(stream16));CUDA_CHECK(cudaStreamDestroy(stream17));CUDA_CHECK(cudaStreamDestroy(stream18));CUDA_CHECK(cudaStreamDestroy(stream19));CUDA_CHECK(cudaStreamDestroy(stream20));CUDA_CHECK(cudaStreamDestroy(stream21));CUDA_CHECK(cudaStreamDestroy(stream22));CUDA_CHECK(cudaStreamDestroy(stream23));CUDA_CHECK(cudaStreamDestroy(stream24));CUDA_CHECK(cudaStreamDestroy(stream25));CUDA_CHECK(cudaStreamDestroy(stream26));CUDA_CHECK(cudaStreamDestroy(stream27));CUDA_CHECK(cudaStreamDestroy(stream28));CUDA_CHECK(cudaStreamDestroy(stream29));CUDA_CHECK(cudaStreamDestroy(stream30));CUDA_CHECK(cudaStreamDestroy(stream31));CUDA_CHECK(cudaStreamDestroy(stream32));CUDA_CHECK(cudaStreamDestroy(stream33));CUDA_CHECK(cudaStreamDestroy(stream34));CUDA_CHECK(cudaStreamDestroy(stream35));CUDA_CHECK(cudaStreamDestroy(stream36));CUDA_CHECK(cudaStreamDestroy(stream37));CUDA_CHECK(cudaStreamDestroy(stream38));CUDA_CHECK(cudaStreamDestroy(stream39));CUDA_CHECK(cudaStreamDestroy(stream40));CUDA_CHECK(cudaStreamDestroy(stream41));CUDA_CHECK(cudaStreamDestroy(stream42));CUDA_CHECK(cudaStreamDestroy(stream43));CUDA_CHECK(cudaStreamDestroy(stream44));CUDA_CHECK(cudaStreamDestroy(stream45));CUDA_CHECK(cudaStreamDestroy(stream46));CUDA_CHECK(cudaStreamDestroy(stream47));CUDA_CHECK(cudaStreamDestroy(stream48));CUDA_CHECK(cudaStreamDestroy(stream49));CUDA_CHECK(cudaStreamDestroy(stream50));CUDA_CHECK(cudaStreamDestroy(stream51));CUDA_CHECK(cudaStreamDestroy(stream52));CUDA_CHECK(cudaStreamDestroy(stream53));CUDA_CHECK(cudaStreamDestroy(stream54));CUDA_CHECK(cudaStreamDestroy(stream55));CUDA_CHECK(cudaStreamDestroy(stream56));CUDA_CHECK(cudaStreamDestroy(stream57));CUDA_CHECK(cudaStreamDestroy(stream58));CUDA_CHECK(cudaStreamDestroy(stream59));CUDA_CHECK(cudaStreamDestroy(stream60));CUDA_CHECK(cudaStreamDestroy(stream61));CUDA_CHECK(cudaStreamDestroy(stream62));CUDA_CHECK(cudaStreamDestroy(stream63));CUDA_CHECK(cudaStreamDestroy(stream64));CUDA_CHECK(cudaStreamDestroy(stream65));CUDA_CHECK(cudaStreamDestroy(stream66));CUDA_CHECK(cudaStreamDestroy(stream67));CUDA_CHECK(cudaStreamDestroy(stream68));CUDA_CHECK(cudaStreamDestroy(stream69));CUDA_CHECK(cudaStreamDestroy(stream70));CUDA_CHECK(cudaStreamDestroy(stream71));CUDA_CHECK(cudaStreamDestroy(stream72));CUDA_CHECK(cudaStreamDestroy(stream73));CUDA_CHECK(cudaStreamDestroy(stream74));CUDA_CHECK(cudaStreamDestroy(stream75));CUDA_CHECK(cudaStreamDestroy(stream76));CUDA_CHECK(cudaStreamDestroy(stream77));CUDA_CHECK(cudaStreamDestroy(stream78));CUDA_CHECK(cudaStreamDestroy(stream79));CUDA_CHECK(cudaStreamDestroy(stream80));CUDA_CHECK(cudaStreamDestroy(stream81));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}


/**/