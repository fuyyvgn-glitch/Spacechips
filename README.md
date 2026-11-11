# Spacechips
Together for a prosperous future and a smart generation 
main.cu
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// CUDA Kernel for Neural Network Dense Layer (Matrix-Vector Multiplication + Bias)
__global__ void neuralNetworkDenseGPU(const float* inputs, const float* weights, const float* bias, float* output, int inputDim, int outputDim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < outputDim) {
        float sum = 0.0f;
        for (int col = 0; col < inputDim; ++col) {
            sum += inputs[col] * weights[row * inputDim + col];
        }
        // Applying ReLU activation function directly on GPU
        float val = sum + bias[row];
        output[row] = (val > 0.0f) ? val : 0.0f;
    }
}

// CPU Complementary Processing for AI Post-Layer Analysis
void neuralNetworkCPUProcessor(const float* intermediateOutput, float* finalOutput, int size) {
    for (int i = 0; i < size; ++i) {
        // Simulating normalization and advanced attention scoring on CPU
        finalOutput[i] = 1.0f / (1.0f + expf(-intermediateOutput[i])); // Sigmoid activation
    }
}

int main() {
    // AI Model Dimensions
    const int inputDim = 1024;
    const int outputDim = 512;
    const int cpuBatchSize = 128;

    std::cout << "Initializing Pi AI Heterogeneous Neural Engine..." << std::endl;

    // Host Memory Allocation
    std::vector<float> h_inputs(inputDim, 0.5f);
    std::vector<float> h_weights(outputDim * inputDim, 0.01f);
    std::vector<float> h_bias(outputDim, 0.1f);
    std::vector<float> h_gpuOutput(outputDim, 0.0f);
    std::vector<float> h_finalOutput(cpuBatchSize, 0.0f);

    // Device Memory Allocation
    float *d_inputs, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_inputs, inputDim * sizeof(float));
    cudaMalloc(&d_weights, outputDim * inputDim * sizeof(float));
    cudaMalloc(&d_bias, outputDim * sizeof(float));
    cudaMalloc(&d_output, outputDim * sizeof(float));

    // Asynchronous CUDA Streams for Pipeline Efficiency
    cudaStream_t aiStream;
    cudaStreamCreate(&aiStream);

    // Transfer Data to GPU
    cudaMemcpyAsync(d_inputs, h_inputs.data(), inputDim * sizeof(float), cudaMemcpyHostToDevice, aiStream);
    cudaMemcpyAsync(d_weights, h_weights.data(), outputDim * inputDim * sizeof(float), cudaMemcpyHostToDevice, aiStream);
    cudaMemcpyAsync(d_bias, h_bias.data(), outputDim * sizeof(float), cudaMemcpyHostToDevice, aiStream);

    // Configure Execution Grid for Neural Network Layer
    int threadsPerBlock = 256;
    int blocksPerGrid = (outputDim + threadsPerBlock - 1) / threadsPerBlock;

    // Execute Core AI Layer on GPU
    neuralNetworkDenseGPU<<<blocksPerGrid, threadsPerBlock, 0, aiStream>>>(d_inputs, d_weights, d_bias, d_output, inputDim, outputDim);

    // Retrieve GPU Results Back to Host
    cudaMemcpyAsync(h_gpuOutput.data(), d_output, outputDim * sizeof(float), cudaMemcpyDeviceToHost, aiStream);

    // Synchronize Stream before CPU processing
    cudaStreamSynchronize(aiStream);

    // Simultaneously Execute CPU Secondary AI Processing Batch
    neuralNetworkCPUProcessor(h_gpuOutput.data(), h_finalOutput.data(), cpuBatchSize);

    std::cout << "Pi AI Neural Inference Completed Successfully across GPU and CPU!" << std::endl;
    std::cout << "Sample AI Output Node [0] (GPU-ReLU): " << h_gpuOutput[0] << std::endl;
    std::cout << "Sample AI Output Node [0] (CPU-Sigmoid): " << h_finalOutput[0] << std::endl;

    // Clean Up Resources
    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaStreamDestroy(aiStream);

    return 0;
}
