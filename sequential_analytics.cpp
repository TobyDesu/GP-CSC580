// ============================================================================
// sequential.cpp (Full Code with Named Excel Exporter)
// ============================================================================
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <fstream>
#include "data_gen.hpp"

using Clock = std::chrono::high_resolution_clock;

struct Stats {
    double mean, variance, stddev, minVal, maxVal, sum;
};

Stats compute_stats(const std::vector<double>& data) {
    double sum = 0.0, sumSq = 0.0;
    double mn = data[0], mx = data[0];
    for (double v : data) {
        sum += v;
        sumSq += v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    double n = static_cast<double>(data.size());
    double mean = sum / n;
    double variance = (sumSq / n) - (mean * mean);
    double stddev = std::sqrt(variance);
    return {mean, variance, stddev, mn, mx, sum};
}

std::vector<uint64_t> compute_histogram(const std::vector<double>& data, int bins,
                                         double lo = DATA_MIN, double hi = DATA_MAX) {
    std::vector<uint64_t> hist(bins, 0);
    double binWidth = (hi - lo) / bins;
    for (double v : data) {
        int idx = static_cast<int>((v - lo) / binWidth);
        if (idx < 0) idx = 0;
        if (idx >= bins) idx = bins - 1;
        hist[idx]++;
    }
    return hist;
}

std::vector<double> sort_data(std::vector<double> data) {
    std::sort(data.begin(), data.end());
    return data;
}

double pearson_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    double n = static_cast<double>(x.size());
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
        sumY2 += y[i] * y[i];
    }
    double numerator = n * sumXY - sumX * sumY;
    double denominator = std::sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    return numerator / denominator;
}

std::vector<double> moving_average(const std::vector<double>& data, int window) {
    std::vector<double> result(data.size());
    double runningSum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        runningSum += data[i];
        if (i >= static_cast<size_t>(window)) {
            runningSum -= data[i - window];
            result[i] = runningSum / window;
        } else {
            result[i] = runningSum / (i + 1);
        }
    }
    return result;
}

uint64_t detect_outliers(const std::vector<double>& data, double mean, double stddev) {
    uint64_t count = 0;
    for (double v : data) {
        double z = (v - mean) / stddev;
        if (std::fabs(z) > 3.0) count++;
    }
    return count;
}

template <typename Func>
double time_it(const std::string& label, Func f) {
    auto t0 = Clock::now();
    f();
    auto t1 = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << std::left << std::setw(28) << label
              << std::right << std::setw(12) << std::fixed << std::setprecision(3)
              << ms << " ms\n";
    return ms;
}

int main(int argc, char** argv) {
    uint64_t N = 10'000'000ULL;
    int bins = 20;
    int window = 100;

    if (argc >= 2) N = std::stoull(argv[1]);
    if (argc >= 3) bins = std::stoi(argv[2]);
    if (argc >= 4) window = std::stoi(argv[3]);

    std::string runSize = "Custom";
    if (N == 1'000'000ULL) runSize = "Small";
    else if (N == 10'000'000ULL) runSize = "Medium";
    else if (N == 100'000'000ULL) runSize = "Large";

    std::cout << "=== SEQUENTIAL RUN (" << runSize << ") ===\n";
    std::cout << "N = " << N << ", bins = " << bins << ", window = " << window << "\n\n";

    std::vector<double> colA, colB;
    double t_gen = time_it("Data generation", [&]() {
        generate_dataset(colA, colB, N);
    });

    Stats stats{};
    double totalTime = 0.0;

    double t1 = time_it("1. Basic Statistics", [&]() { stats = compute_stats(colA); });
    double t2 = time_it("2. Histogram", [&]() { compute_histogram(colA, bins); });
    double t3 = time_it("3. Sorting", [&]() { sort_data(colA); });
    double t4 = time_it("4. Pearson Correlation", [&]() { pearson_correlation(colA, colB); });
    double t5 = time_it("5. Moving Average", [&]() { moving_average(colA, window); });
    double t6 = time_it("6. Outlier Detection", [&]() { detect_outliers(colA, stats.mean, stats.stddev); });

    totalTime = t1 + t2 + t3 + t4 + t5 + t6;
    std::cout << "\nTotal compute time (excludes data gen): " << totalTime << " ms\n";

    // Set clear names matching both the size and execution type
    std::string resultFilename = "custom_sequential_result.csv";
    if (runSize == "Small") resultFilename = "small_sequential_result.csv";
    else if (runSize == "Medium") resultFilename = "medium_sequential_result.csv";
    else if (runSize == "Large") resultFilename = "large_sequential_result.csv";

    // Write to the specific Sequential Excel file
    std::ifstream checkFile(resultFilename);
    bool exists = checkFile.good();
    checkFile.close();

    std::ofstream csvFile(resultFilename, std::ios::app);
    if (!exists) {
        csvFile << "Execution Type,N,Bins,Window,Data Gen (ms),Task 1 (ms),Task 2 (ms),Task 3 (ms),Task 4 (ms),Task 5 (ms),Task 6 (ms),Total Compute (ms)\n";
    }
    csvFile << "Sequential," << N << "," << bins << "," << window << ","
            << t_gen << "," << t1 << "," << t2 << "," << t3 << ","
            << t4 << "," << t5 << "," << t6 << "," << totalTime << "\n";
    csvFile.close();
    
    std::cout << "--> Results saved to '" << resultFilename << "'\n";
    return 0;
}