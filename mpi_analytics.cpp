#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default data scale size (10 Million points as recommended for Medium size)
    long long N = 10000000; 
    if (argc > 1) {
        N = std::atoll(argv[1]);
    }

    // Every node needs to know the total size N
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Calculate how many elements each of the 4 nodes will process
    long long local_N = N / size;
    std::vector<double> local_data(local_N);

    // Master Node (Rank 0) handles original dataset generation
    std::vector<double> global_data;
    if (rank == 0) {
        global_data.resize(N);
        // Fixed seed configuration for reproducibility across benchmarks
        std::mt19937_64 rng(42); 
        std::uniform_real_distribution<double> dist(0.0, 10000.0);

        std::cout << "Master Node: Generating " << N << " data points..." << std::endl;
        for (long long i = 0; i < N; ++i) {
            global_data[i] = dist(rng);
        }
    }

    // Start timing synchronization across your network
    double start_time = MPI_Wtime();

    // Distribute chunks of data seamlessly over the network to all laptops
    MPI_Scatter(global_data.data(), local_N, MPI_DOUBLE, 
                local_data.data(), local_N, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    // --- Task 1: Basic Statistics (Local Processing) ---
    double local_sum = 0.0;
    double local_min = local_data[0];
    double local_max = local_data[0];

    for (long long i = 0; i < local_N; ++i) {
        local_sum += local_data[i];
        if (local_data[i] < local_min) local_min = local_data[i];
        if (local_data[i] > local_max) local_max = local_data[i];
    }

    // --- Aggregating Data Back to Master ---
    double global_sum = 0.0;
    double global_min = 0.0;
    double global_max = 0.0;

    // Use MPI Collective operations to find true answers across all nodes
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Stop timing
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // Output Final Combined Statistics on Master Node
    if (rank == 0) {
        double global_mean = global_sum / N;
        std::cout << "\n=====================================" << std::endl;
        std::cout << "📊 PARALLEL STATISTICS RESULTS" << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "Data Points processed: " << N << std::endl;
        std::cout << "Global Mean          : " << global_mean << std::endl;
        std::cout << "Global Min           : " << global_min << std::endl;
        std::cout << "Global Max           : " << global_max << std::endl;
        std::cout << "Execution Time       : " << elapsed_time << " seconds" << std::endl;
        std::cout << "=====================================" << std::endl;
    }

    MPI_Finalize();
    return 0;
}