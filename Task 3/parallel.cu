#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << "\n";\
        exit(1);\
    }\
}

// Отдельные ядра для каждой операции
__global__ void add_kernel(const float* a, const float* b, float* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] + b[idx];
}

__global__ void sub_kernel(const float* a, const float* b, float* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] - b[idx];
}

__global__ void mul_kernel(const float* a, const float* b, float* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] * b[idx];
}

__global__ void div_kernel(const float* a, const float* b, float* result, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) result[idx] = a[idx] / b[idx];
}

void benchmark_cuda_operations(const float* a, const float* b,
    float* add, float* sub,
    float* mul, float* div,
    size_t n, int block_size,
    size_t iterations) {
    float* d_a, * d_b, * d_add, * d_sub, * d_mul, * d_div;

    // Выделение памяти на устройстве
    CHECK_CUDA(cudaMalloc(&d_a, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_add, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sub, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mul, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_div, n * sizeof(float)));

    // Копирование данных на устройство
    CHECK_CUDA(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(block_size);
    dim3 grid((n + block_size - 1) / block_size);

    // Временные переменные
    double add_time = 0, sub_time = 0, mul_time = 0, div_time = 0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (size_t iter = 0; iter < iterations; ++iter) {
        // Сложение
        CHECK_CUDA(cudaEventRecord(start));
        add_kernel << <grid, block >> > (d_a, d_b, d_add, n);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        add_time += ms;

        // Вычитание
        CHECK_CUDA(cudaEventRecord(start));
        sub_kernel << <grid, block >> > (d_a, d_b, d_sub, n);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        sub_time += ms;

        // Умножение
        CHECK_CUDA(cudaEventRecord(start));
        mul_kernel << <grid, block >> > (d_a, d_b, d_mul, n);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        mul_time += ms;

        // Деление
        CHECK_CUDA(cudaEventRecord(start));
        div_kernel << <grid, block >> > (d_a, d_b, d_div, n);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        div_time += ms;
    }

    // Копирование результатов обратно
    CHECK_CUDA(cudaMemcpy(add, d_add, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(sub, d_sub, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(mul, d_mul, n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(div, d_div, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Освобождение памяти
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_add));
    CHECK_CUDA(cudaFree(d_sub));
    CHECK_CUDA(cudaFree(d_mul));
    CHECK_CUDA(cudaFree(d_div));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // Вывод результатов
    std::cout << "Block size: " << block_size << "\n";
    std::cout << "Average times over " << iterations << " iterations (ms):\n";
    std::cout << "Addition: " << add_time / iterations << "\n";
    std::cout << "Subtraction: " << sub_time / iterations << "\n";
    std::cout << "Multiplication: " << mul_time / iterations << "\n";
    std::cout << "Division: " << div_time / iterations << "\n";
}

int main() {
    const size_t N = 10000000;
    const size_t iterations = 100;

    std::vector<float> a(N), b(N);
    std::vector<float> add(N), sub(N), mul(N), div(N);

    // Генерация данных
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 10.0f);
    for (size_t i = 0; i < N; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }

    // Тестирование разных размеров блоков
    std::vector<int> block_sizes = { 32, 64, 128, 256, 512, 1024 };

    for (int block_size : block_sizes) {
        benchmark_cuda_operations(a.data(), b.data(),
            add.data(), sub.data(),
            mul.data(), div.data(),
            N, block_size, iterations);

        // Проверка корректности для первого блока
        std::cout << "\nSample results (first 5 elements):\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << a[i] << " + " << b[i] << " = " << add[i] << "\n";
            std::cout << a[i] << " - " << b[i] << " = " << sub[i] << "\n";
            std::cout << a[i] << " * " << b[i] << " = " << mul[i] << "\n";
            std::cout << a[i] << " / " << b[i] << " = " << div[i] << "\n\n";
        }
        std::cout << "----------------------------------------\n";
    }

    return 0;
}