#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <math_functions.h>

float *d_mulhead_wQ;
float *d_mulhead_wK;
float *d_mulhead_wV;

float *Q;
float *K;
float *V;

float *QK;

size_t sizeWeighQKV = 16 * 16 * 8 * sizeof(float);

__global__
void attention_score(float *QK, float *V, float *R, int seq){
    int iteration = 0;

    extern __shared__ float rows[];
    __shared__ float value[16][16];
    __shared__ float buffer_val [16];
    
    int idx = (blockIdx.y * blockDim.x) + threadIdx.x;

    if(threadIdx.x == 0 && threadIdx.y == 0){
        for(int i = 0; i < 16; i++){
            for(int j = 0; j < 16; j++){
                value[i][j] = V[(blockIdx.x *16) + i + (j * 128)];
            }
        }
    }
    rows[threadIdx.x] = QK[idx];
    __syncthreads();

    float max = 0.0f;
    for(int i = 0; i < blockDim.x; i++){
        if(rows[i] > max){
            max = rows[i];
        }
    }

    rows[threadIdx.x] = expf(rows[threadIdx.x] - max);
    __syncthreads();

    float sum = 0.0f;
    for(int i = 0; i < blockDim.x; i++){
        sum += rows[i];
    }

    rows[threadIdx.x] = rows[threadIdx.x] / sum;
    __syncthreads();

    if(threadIdx.x < 16){
        float result = 0.0f;
        for(int j = 0; j < seq; j++){
            // buffer_val[i] =
            result += rows[j] + value[threadIdx.x][j];
        }

        int idx_r = (blockIdx.y * 128) + (blockIdx.x * 16) + threadIdx.x;
        R[idx_r] = result;
    }

}

__global__
void scaled_QK(float *X, float *W, float *R, int seq){
    int current_seq = seq;
    int processed_seq;
    int iteration = 0;

    if(current_seq > 16){
        processed_seq = 16;
        // current_seq -= 16;
    }else{
        processed_seq = current_seq;
    }

    __shared__ float aX[16][16];
    __shared__ float aW[16][16];
    __shared__ float aR[16][16];

    for(int iter = 0; processed_seq > 0; iter++){

        //int idx = ((threadIdx.y * iteration) * 128) + (blockIdx.x * 16) + threadIdx.x;
        int idx = (iteration * 128 * 16) + (threadIdx.y * 128) + (blockIdx.x * 16) + threadIdx.x;
        
        if(threadIdx.y < processed_seq){
            aX[threadIdx.x][threadIdx.y] = X[idx];
            aW[threadIdx.x][threadIdx.y] = W[idx];
        }
    
        __syncthreads();
    
        float acc = 0.0f;
        if(threadIdx.y < processed_seq && threadIdx.x < processed_seq){
            for(int i = 0; i < 16; i++){
                acc += aX[i][threadIdx.y] * aW[i][threadIdx.x];
            }
            aR[threadIdx.x][threadIdx.y] = sqrtf(acc);
        }
        __syncthreads();
    
        if(threadIdx.y < processed_seq){
            R[idx] = aR[threadIdx.x][threadIdx.y];
        }

        iteration += 1;
        current_seq -= 16;

        if(current_seq > 16){
            processed_seq = 16;
        }else{
            processed_seq = current_seq;
        }
    }
}

__global__
void projection_qkv16(float *X, float *W, float *R, int seq){
    int current_seq = seq;
    int processed_seq;
    int iteration = 0;

    if(current_seq > 16){
        processed_seq = 16;
        // current_seq -= 16;
    }else{
        processed_seq = current_seq;
    }

    __shared__ float aX[16][16];
    __shared__ float aW[16][16];
    __shared__ float aR[16][16];

    for(int iter = 0; processed_seq > 0; iter++){

        //int idx = ((threadIdx.y * iteration) * 128) + (blockIdx.x * 16) + threadIdx.x;
        int idx = (iteration * 128 * 16) + (threadIdx.y * 128) + (blockIdx.x * 16) + threadIdx.x;
        
        if(threadIdx.y < processed_seq){
            aX[threadIdx.x][threadIdx.y] = X[idx];
        }
        aW[threadIdx.x][threadIdx.y] = W[idx];
    
        __syncthreads();
    
        float acc = 0.0f;
        if(threadIdx.y < processed_seq){
            for(int i = 0; i < 16; i++){
                acc += aX[i][threadIdx.y] * aW[i][threadIdx.x];
            }
            aR[threadIdx.x][threadIdx.y] = acc;
        }
        __syncthreads();
    
        if(threadIdx.y < processed_seq){
            R[idx] = aR[threadIdx.x][threadIdx.y];
        }

        iteration += 1;
        current_seq -= 16;

        if(current_seq > 16){
            processed_seq = 16;
        }else{
            processed_seq = current_seq;
        }
    }
}

extern "C" __declspec(dllexport)
void initWeight128(float *Q, float *K, float *V){
    printf("weight loading ..\n");

    cudaMalloc(&d_mulhead_wQ, sizeWeighQKV);
    cudaMalloc(&d_mulhead_wK, sizeWeighQKV);
    cudaMalloc(&d_mulhead_wV, sizeWeighQKV);

    cudaMemcpy(d_mulhead_wQ, Q, sizeWeighQKV, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mulhead_wK, K, sizeWeighQKV, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mulhead_wV, V, sizeWeighQKV, cudaMemcpyHostToDevice);
}

extern "C" __declspec(dllexport)
void saveWeight128(float *Q, float *K, float *V){
    printf("weight saving ..\n");

    cudaMemcpy(Q, d_mulhead_wQ, sizeWeighQKV, cudaMemcpyDeviceToHost);
    cudaMemcpy(K, d_mulhead_wK, sizeWeighQKV, cudaMemcpyDeviceToHost);
    cudaMemcpy(V, d_mulhead_wV, sizeWeighQKV, cudaMemcpyDeviceToHost);
}

extern "C" __declspec(dllexport)
void projection_QKV128(float *X, int seq){
    printf("calculating Q, K, V ..\n");
    float *d_X;
    
    cudaMalloc(&d_X, seq * 128 * sizeof(float));
    cudaMalloc(&Q, seq * 128 * sizeof(float));
    cudaMalloc(&K, seq * 128 * sizeof(float));
    cudaMalloc(&V, seq * 128 * sizeof(float));

    dim3 block(16, 16, 1);
    dim3 grid(8, 1, 1);
    projection_qkv16<<<grid, block>>>(d_X, d_mulhead_wQ, Q, seq);
    projection_qkv16<<<grid, block>>>(d_X, d_mulhead_wK, K, seq);
    projection_qkv16<<<grid, block>>>(d_X, d_mulhead_wV, V, seq);

    cudaFree(d_X);
}

extern "C" __declspec(dllexport)
void scaledQK(int seq){
    dim3 block(16, 16, 1);
    dim3 grid(8, 1, 1);

    cudaMalloc(&QK, seq * seq * 8 * sizeof(float));
    scaled_QK<<<grid, block>>>(Q, K, QK, seq);
}

extern "C" __declspec(dllexport)
void attentionScore(float *X, float *R, int seq){
    projection_QKV128(X, seq);
    scaledQK(seq);

    dim3 block(seq, 1, 1);
    dim3 grid(8, seq, 1);

    attention_score<<<grid, block>>>(QK, V, R, seq);
}


extern "C" __declspec(dllexport)
void example(float *X, float *W, float *R, int seq){
    float *d_X, *d_W, *d_R;

    size_t size = 128 * seq * sizeof(float);
    size_t size_w = 16 * 16 * 8 * sizeof(float);

    cudaMalloc(&d_X, size);
    cudaMalloc(&d_W, size_w);
    cudaMalloc(&d_R, size);

    for(int i=0; i<16;i++){
        printf(" %f ", X[i]);
    }
    printf("\n");

    cudaMemcpy(d_X, X, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, size_w, cudaMemcpyHostToDevice);

    dim3 block(16, 16, 1);
    dim3 grid(8, 1, 1);
    projection_qkv16<<<grid, block>>>(d_X, d_W, d_R, seq);
    cudaMemcpy(R, d_R, size, cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_R);
}

extern "C" __declspec(dllexport)
void mulHeadAttention(float *X, int seq){

}

extern "C" __declspec(dllexport)
void freeMem(){
    cudaFree(d_mulhead_wQ);
    cudaFree(d_mulhead_wK);
    cudaFree(d_mulhead_wV);
}
