// lib.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

__global__ void add_kernel(float* a, float* b, float* result, int n) {
    int i = threadIdx.x;
    if (i < n)
        result[i] = a[i] + b[i];
}

__global__ void mult_kernel(float* a, float* b, float* result, int n) {
    int i = threadIdx.x;
    if (i < n)
        result[i] = a[i] + b[i];
}

__global__ void make_corpus(char* data, int n) {
    int i = threadIdx.x;
    if (i < n)
        result[i] = a[i] + b[i];
}

extern "C" __declspec(dllexport)
void add(float* a, float* b, float* result, int n) {
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_result, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    add_kernel<<<1, n>>>(d_a, d_b, d_result, n);

    cudaMemcpy(result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

extern "C" __declspec(dllexport)
void mult(float* a, float* b, float* result, int n) {
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_result, n * sizeof(float));

    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    mult_kernel<<<1, n>>>(d_a, d_b, d_result, n);

    cudaMemcpy(result, d_result, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

extern "C" __declspec(dllexport)
void tfidf(float* data, float* tfidf, int len){
    printf("length: %i \n", len);
    for(int i = 0; i < len; i++){
        tfidf[i] = data[i];
    }
}

void tfidf2(char* data[], int len){
    int data_len;
    printf("word lengt: %zu", sizeof(data));
}

extern "C" __declspec(dllexport)
char **cu_make_corpus(char** data, int len, int *r_len){

    char **d_data;
    cudaMalloc(&d_data, len * sizeof(char));
    cudaMemcpy(d_data, data, len * sizeof(char), cudaMemcpyHostToDevice);

    printf("make corpus..");
    char** corpus = (char **)malloc(len * sizeof(char *));
    int corpus_count = 0;
    bool isUnique = false;
    
    for(int i = 0; i < len; i++){
        printf("\r%i/%i data processed..", i, len);
        for(int j = 0; j < corpus_count; j++){
            if(strcmp(corpus[j], data[i]) == 0){
                isUnique = true;
                break;
            }
        }

        if(isUnique == false){
            corpus[corpus_count] = strdup(data[i]);
            corpus_count += 1;
        }else{
            isUnique = false;
        }
    }
    *r_len = corpus_count;

    return corpus;
}

extern "C" __declspec(dllexport)
char **make_corpus(char** data, int len, int *r_len){
    printf("make corpus..");
    char** corpus = (char **)malloc(len * sizeof(char *));
    int corpus_count = 0;
    bool isUnique = false;
    
    for(int i = 0; i < len; i++){
        printf("\r%i/%i data processed..", i, len);
        for(int j = 0; j < corpus_count; j++){
            if(strcmp(corpus[j], data[i]) == 0){
                isUnique = true;
                break;
            }
        }

        if(isUnique == false){
            corpus[corpus_count] = strdup(data[i]);
            corpus_count += 1;
        }else{
            isUnique = false;
        }
    }
    *r_len = corpus_count;

    return corpus;
}

extern "C" __declspec(dllexport)
float *countTf(char **corpus, int count_cor, char **document, int count_doc, int doc_length, int* docindex){
    float* tf = (float *)calloc(count_cor * doc_length, sizeof(float));
    int doc_offside = 0;
    printf("\ncount tf data..\n");

    for(int i = 0; i < doc_length; i++){
        printf("\r%i/%i data counted..", i, doc_length);
        for(int w = 0; w < docindex[i]; w++){
            for(int j = 0; j < count_cor; j++){
                if(strcmp(document[doc_offside+w], corpus[j]) == 0){
                    tf[(i*count_cor)+j] += 1;
                }
            }
        }
        doc_offside += docindex[i];
    }

    return tf;
}

extern "C" __declspec(dllexport)
float *countIdf(char **corpus, int count_cor, char **document, int count_doc, int doc_length, int *docindex){
    float *idf = (float *)calloc(count_cor, sizeof(float));
    int current_doc_idx = 0;
    bool isContinue = false;
    printf("\ncount data idf..\n");
    
    for(int i = 0; i < count_cor; i++){
        printf("\r%i/%i data counted..", i, count_cor);
        for(int j = 0; j < doc_length; j++){
            for(int k = 0; k < docindex[j]; k++){
                if(strcmp(corpus[i], document[current_doc_idx]) == 0){
                    idf[i] += 1;
                    current_doc_idx = current_doc_idx + (docindex[j] - k);
                    isContinue = true;
                    break;
                }
                current_doc_idx += 1;
            }
            if(isContinue){
                isContinue = false;
                continue;
            }
        }
        current_doc_idx = 0;
        idf[i] = log10f((float)(doc_length) / (float)(1 + idf[i]));
    }

    return idf;
}

extern "C" __declspec(dllexport)
float *countTfIdf(char** data, int count, int doc_len, int *index, int *r_corpus_len){
    int corpus_length = 0;
    char **corpus = make_corpus(data, count, &corpus_length);
    float *tfidf = (float *)calloc(corpus_length*doc_len, sizeof(float));

    float *tf = countTf(corpus, corpus_length, data, count, doc_len, index);
    float *idf = countIdf(corpus, corpus_length, data, count, doc_len, index);

    printf("\ncount TF-IDF data..\n");

    for(int i = 0; i < doc_len; i++){
        printf("\r%i/%i data counted..", i, doc_len);
        for(int x = 0; x < corpus_length; x++){
            tfidf[(i * corpus_length) + x] = tf[(i * corpus_length) + x] * idf[x];
        }
    }
    printf("corpus length: %i  doc length: %i", corpus_length, doc_len);
    for(int t = 0; t < 1000; t++){
        printf(" || %s: %f || ", corpus[t], tfidf[t]);
    }

    return tfidf;
}

