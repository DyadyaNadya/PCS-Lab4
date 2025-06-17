#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <numeric>

void generate_arrays(std::vector<float>& a, std::vector<float>& b) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 10.0f);

    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }
}

void benchmark_operations(const std::vector<float>& a, const std::vector<float>& b,
    std::vector<float>& add, std::vector<float>& sub,
    std::vector<float>& mul, std::vector<float>& div,
    size_t iterations) {
    // Временные переменные для замеров
    double add_time = 0, sub_time = 0, mul_time = 0, div_time = 0;
    size_t n = a.size();

    for (size_t iter = 0; iter < iterations; ++iter) {
        // Сложение
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; ++i) add[i] = a[i] + b[i];
        auto end = std::chrono::high_resolution_clock::now();
        add_time += std::chrono::duration<double>(end - start).count();

        // Вычитание
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; ++i) sub[i] = a[i] - b[i];
        end = std::chrono::high_resolution_clock::now();
        sub_time += std::chrono::duration<double>(end - start).count();

        // Умножение
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; ++i) mul[i] = a[i] * b[i];
        end = std::chrono::high_resolution_clock::now();
        mul_time += std::chrono::duration<double>(end - start).count();

        // Деление
        start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; ++i) div[i] = a[i] / b[i];
        end = std::chrono::high_resolution_clock::now();
        div_time += std::chrono::duration<double>(end - start).count();
    }

    // Вывод среднего времени
    std::cout << "Average times over " << iterations << " iterations (ms):\n";
    std::cout << "Addition: " << (add_time / iterations) * 1000 << "\n";
    std::cout << "Subtraction: " << (sub_time / iterations) * 1000 << "\n";
    std::cout << "Multiplication: " << (mul_time / iterations) * 1000 << "\n";
    std::cout << "Division: " << (div_time / iterations) * 1000 << "\n";
}

int main() {
    const size_t N = 10000000;
    const size_t iterations = 100;

    std::vector<float> a(N), b(N);
    std::vector<float> add(N), sub(N), mul(N), div(N);

    generate_arrays(a, b);
    benchmark_operations(a, b, add, sub, mul, div, iterations);

    // Проверка корректности
    std::cout << "\nSample results (first 5 elements):\n";
    for (int i = 0; i < 5; ++i) {
        std::cout << a[i] << " + " << b[i] << " = " << add[i] << "\n";
        std::cout << a[i] << " - " << b[i] << " = " << sub[i] << "\n";
        std::cout << a[i] << " * " << b[i] << " = " << mul[i] << "\n";
        std::cout << a[i] << " / " << b[i] << " = " << div[i] << "\n\n";
    }

    return 0;
}