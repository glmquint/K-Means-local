#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <thread>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <limits>
#include <string>
#include <chrono>
#include <math.h>
#include <time.h>
#include <cassert>

#define PRINT_CENTERS
#define PREALLOC_OPTIMIZE_OFF

#define POINT_DIMENSION 2
#define PRECISION float
#define MAX_PRECISION 3.40282e+38
#define STOPPING_ERROR 1e-2

using namespace std;
clock_t tic, toc;

int THREADS = 1;
bool CONVERGED = false;
int NUM_CLUSTERS = 5;
int DATASET_SIZE;
int THREADS_PER_BLOCK;

struct KMeansData_s
{
	// clustered_Point
	PRECISION *points_p_x;
	PRECISION *points_p_y;
	int *points_k;

	// Centroid
	PRECISION *centroids_p_x;
	PRECISION *centroids_p_y;
	PRECISION *centroids_sum_x;
	PRECISION *centroids_sum_y;
	int *centroids_partition_lengths;

	// clustered_Point
	PRECISION *d_points_p_x;
	PRECISION *d_points_p_y;
	int *d_points_k;

	// Centroid
	PRECISION *d_centroids_p_x;
	PRECISION *d_centroids_p_y;
	PRECISION *d_centroids_sums_x;
	PRECISION *d_centroids_sums_y;
	int *d_centroids_partition_lengths;

	#ifdef PREALLOC_OPTIMIZE // TODO: check optimization
	// Point
	POINT_PRECISION  d_sum_x;
	POINT_PRECISION  d_sum_y;

	int* d_points_per_centroid;
	#endif // PREALLOC_OPTIMIZE

};
typedef struct KMeansData_s KMeansData;

#define distance(ax, ay, bx, by) (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
//
//PRECISION distanceCPU(PRECISION ax, PRECISION ay, PRECISION bx, PRECISION by)
//{
//	return (ax - bx) * (ax - bx) + (ay - by) * (ay - by);
//}

__global__ void worker(KMeansData data, int datasetSize, int numClusters, int partitionSize, int numThreads)
{
	PRECISION dist = 0;
	int best_k;
	PRECISION min_d;
	PRECISION* sum_x;
	PRECISION* sum_y;
	int* points_per_centroid;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < numThreads) {

#ifdef PREALLOC_OPTIMIZE
		sum = &data.sum[(blockDim.x * blockIdx.x + threadIdx.x) * numClusters];
		pointsPerCentroid = &data.pointsPerCentroid[(blockDim.x * blockIdx.x + threadIdx.x) * numClusters];
#else
		sum_x = new PRECISION[numClusters];
		sum_y = new PRECISION[numClusters];
		points_per_centroid = new int[numClusters];
#endif // PREALLOC_OPTIMIZE

		for (int j = 0; j < numClusters; ++j)
		{
			sum_x[j] = 0.0;
			sum_y[j] = 0.0;
			points_per_centroid[j] = 0;
		}

		for (int elem = 0; elem < partitionSize; ++elem)
		{
			int partition_elem = elem * numThreads + index;
			if (partition_elem < datasetSize)
			{
				PRECISION p_x = data.d_points_p_x[partition_elem];
				PRECISION p_y = data.d_points_p_y[partition_elem];
				min_d = MAX_PRECISION;
				best_k = -1;
				for (int i = 0; i < numClusters; ++i)
				{
					dist = distance(p_x,
									p_y,
									data.d_centroids_p_x[i],
									data.d_centroids_p_y[i]);
					if (dist < min_d) {
						min_d = dist;
						best_k = i;
					}
					//best_k = i * (dist < min_d) + best_k * (dist >= min_d);
					//min_d = dist * (dist < min_d) + min_d * (dist >= min_d);
				}
				data.d_points_k[partition_elem] = best_k;
				sum_x[best_k] += p_x;
				sum_y[best_k] += p_y;
				points_per_centroid[best_k]++;
			}
		}

		for (int i = 0; i < numClusters; ++i)
		{
			data.d_centroids_sums_x[i * numThreads + index] = sum_x[i];
			data.d_centroids_sums_y[i * numThreads + index] = sum_y[i];
			data.d_centroids_partition_lengths[i * numThreads + index] = points_per_centroid[i];
		}

#ifndef PREALLOC_OPTIMIZE
		delete[] sum_x;
		delete[] sum_y;
		delete[] points_per_centroid;
#endif // !PREALLOC_OPTIMIZE
	}
}

void updateCenters(KMeansData &data)
{
	double max_err = numeric_limits<double>::min();
	double point_sum_x;
	double point_sum_y;
	int sum_of_lengths;
	for (int i = 0; i < NUM_CLUSTERS; ++i)
	{
		point_sum_x = 0.0;
		point_sum_y = 0.0;
		sum_of_lengths = 0;

		for (int j = 0; j < THREADS; j++)
		{
			point_sum_x += data.centroids_sum_x[i * THREADS + j];
			point_sum_y += data.centroids_sum_y[i * THREADS + j];
			sum_of_lengths += data.centroids_partition_lengths[i * THREADS + j];
		}

		point_sum_x /= sum_of_lengths;
		point_sum_y /= sum_of_lengths;

		PRECISION dist = distance(data.centroids_p_x[i], data.centroids_p_y[i], point_sum_x, point_sum_y);
		data.centroids_p_x[i] = point_sum_x;
		data.centroids_p_y[i] = point_sum_y;

		//PRECISION point_sum_square_norm = (point_sum_x * point_sum_x) + (point_sum_y * point_sum_y);
		PRECISION error = sqrt(dist); // point_sum_square_norm;
		if (error > max_err)
		{
			max_err = error;
		}

#ifdef PRINT_CENTERS
		printf("centroid %d (%f:%f) with %d elements (error: %f)\n", i, data.centroids_p_x[i], data.centroids_p_y[i], sum_of_lengths, error);
#endif
	}

#ifdef PRINT_CENTERS
	printf("==================================================\n");
#endif
	CONVERGED = (max_err < STOPPING_ERROR);
}

void performRounds(dim3 grid, dim3 block, KMeansData &data, int partitionSize)
{
	int round = 0;
	cudaError_t cerr;
	while(!CONVERGED)
	{
		cerr = cudaMemcpy(data.d_centroids_p_x, data.centroids_p_x, NUM_CLUSTERS * sizeof(PRECISION), cudaMemcpyHostToDevice);
		assert(cerr == cudaSuccess);
		cerr = cudaMemcpy(data.d_centroids_p_y, data.centroids_p_y, NUM_CLUSTERS * sizeof(PRECISION), cudaMemcpyHostToDevice);
		assert(cerr == cudaSuccess);

		worker<<<grid, block>>>(data, DATASET_SIZE, NUM_CLUSTERS, partitionSize, THREADS);
		cerr = cudaDeviceSynchronize();
		assert(cerr == cudaSuccess);

		cerr = cudaMemcpy(data.centroids_sum_x, data.d_centroids_sums_x, NUM_CLUSTERS * THREADS * sizeof(PRECISION), cudaMemcpyDeviceToHost);
		assert(cerr == cudaSuccess);
		cerr = cudaMemcpy(data.centroids_sum_y, data.d_centroids_sums_y, NUM_CLUSTERS * THREADS * sizeof(PRECISION), cudaMemcpyDeviceToHost);
		assert(cerr == cudaSuccess);

		cerr = cudaMemcpy(data.centroids_partition_lengths, data.d_centroids_partition_lengths, NUM_CLUSTERS * THREADS * sizeof(int), cudaMemcpyDeviceToHost);
		assert(cerr == cudaSuccess);

#ifdef PRINT_CENTERS
		printf("%f %f %f %f %f\n", data.centroids_sum_x[0], data.centroids_sum_x[THREADS * 1], data.centroids_sum_x[THREADS * 2], data.centroids_sum_x[THREADS * 3], data.centroids_sum_x[THREADS * 4]);
		printf("%f %f %f %f %f\n", data.centroids_sum_y[0], data.centroids_sum_y[THREADS * 1], data.centroids_sum_y[THREADS * 2], data.centroids_sum_y[THREADS * 3], data.centroids_sum_y[THREADS * 4]);
		printf("%d %d %d %d %d\n", data.centroids_partition_lengths[0], data.centroids_partition_lengths[THREADS * 1], data.centroids_partition_lengths[THREADS * 2], data.centroids_partition_lengths[THREADS * 3], data.centroids_partition_lengths[THREADS * 4]);
#endif

		updateCenters(data);
		round++;
	}

#ifdef PRINT_CENTERS
	printf("took %d rounds\n", round);
#endif
}

void setupRandomCentroids(KMeansData &data)
{
	srand(69420);
#ifdef PRINT_CENTERS
	printf("generated centroids:\n");
#endif // PRINT_CENTERS
	for (int i = 0; i < NUM_CLUSTERS; ++i)
	{
		int random_index = rand() % (DATASET_SIZE);
		data.centroids_p_x[i] = data.points_p_x[random_index];
		data.centroids_p_y[i] = data.points_p_y[random_index];
#ifdef PRINT_CENTERS
		printf("centroid %d (%f:%f)\n", i, data.centroids_p_x[i], data.centroids_p_y[i]);
	}
	printf("-----------------------\n");
#endif // PRINT_CENTERS
#ifndef PRINT_CENTERS
	}
#endif
}

void deserializePoints(char* inputFile, KMeansData &data)
{
	ifstream infile;
	int point_dimension;
	double tmp;
	infile.open(inputFile, ios::in | ios::binary);
	if (infile.fail())
	{
		cout << "can't find file " << inputFile << endl;
		exit(1);
	}

	infile.read((char*)(&DATASET_SIZE), sizeof(DATASET_SIZE));
	infile.read((char*)(&point_dimension), sizeof(int));
	assert(point_dimension == POINT_DIMENSION);
	data.points_p_x = new PRECISION[DATASET_SIZE];
	data.points_p_y = new PRECISION[DATASET_SIZE];
	data.points_k = new int[DATASET_SIZE];
	for (int i = 0; i < DATASET_SIZE; i++)
	{
		infile.read((char*)(&tmp), sizeof(double));
		data.points_p_x[i] = (PRECISION)tmp;
		infile.read((char*)(&tmp), sizeof(double));
		data.points_p_y[i] = (PRECISION)tmp;
	}
	infile.close();
}

int main(int argc, char** argv)
{
	if (argc < 5)
	{
		printf("[USAGE]: %s dataset.serialized num_clusters num_threads threads_per_block\n", argv[0]);
		exit(1);
	}

	NUM_CLUSTERS = stoi(argv[2]);
	THREADS = stoi(argv[3]);
	THREADS_PER_BLOCK = stoi(argv[4]);

	KMeansData data;
	deserializePoints(argv[1], data);

	data.centroids_p_x = new PRECISION[NUM_CLUSTERS];
	data.centroids_p_y = new PRECISION[NUM_CLUSTERS];
	data.centroids_sum_x = new PRECISION[THREADS * NUM_CLUSTERS];
	data.centroids_sum_y = new PRECISION[THREADS * NUM_CLUSTERS];
	data.centroids_partition_lengths = new int[THREADS * NUM_CLUSTERS];

#ifdef PREALLOC_OPTIMIZE
	data.sum = new double[NUM_CLUSTERS * THREADS * POINT_DIMENSION];
	data.pointsPerCentroid = new int[NUM_CLUSTERS * THREADS];
#endif // PREALLOC_OPTIMIZE

	int numBlocks = THREADS / THREADS_PER_BLOCK;
	if (THREADS % THREADS_PER_BLOCK)
		numBlocks++;

	//int partitionSize = DATASET_SIZE / THREADS;
	//if (DATASET_SIZE % THREADS)
	//	partitionSize++;
	int partitionSize = (DATASET_SIZE % THREADS == 0) ? (DATASET_SIZE / THREADS) : (DATASET_SIZE / (THREADS - 1));
	dim3 grid(numBlocks, 1, 1);
	dim3 block(THREADS_PER_BLOCK, 1, 1);

	cudaMalloc((void**)&data.d_points_p_x, DATASET_SIZE * sizeof(PRECISION));
	cudaMalloc((void**)&data.d_points_p_y, DATASET_SIZE * sizeof(PRECISION));
	cudaMalloc((void**)&data.d_points_k, DATASET_SIZE * sizeof(int));
	cudaMalloc((void**)&data.d_centroids_p_x, NUM_CLUSTERS * sizeof(PRECISION));
	cudaMalloc((void**)&data.d_centroids_p_y, NUM_CLUSTERS * sizeof(PRECISION));
	cudaMalloc((void**)&data.d_centroids_sums_x, THREADS * NUM_CLUSTERS * sizeof(PRECISION));
	cudaMalloc((void**)&data.d_centroids_sums_y, THREADS * NUM_CLUSTERS * sizeof(PRECISION));
	cudaMalloc((void**)&data.d_centroids_partition_lengths, THREADS * NUM_CLUSTERS * sizeof(int));

#ifdef PREALLOC_OPTIMIZE
	cudaMalloc((void**)&data.sum, NUM_CLUSTERS * THREADS * POINT_DIMENSION * sizeof(double));
	cudaMalloc((void**)&data.pointsPerCentroid, NUM_CLUSTERS * THREADS * sizeof(int));
#endif // PREALLOC_OPTIMIZE

	cudaError_t cerr;
	clock_t ds_tic = clock();
	cerr = cudaMemcpy(data.d_points_p_x, data.points_p_x, DATASET_SIZE * sizeof(PRECISION), cudaMemcpyHostToDevice);
	assert(cerr == cudaSuccess);
	cerr = cudaMemcpy(data.d_points_p_y, data.points_p_y, DATASET_SIZE * sizeof(PRECISION), cudaMemcpyHostToDevice);
	assert(cerr == cudaSuccess);
	clock_t ds_toc = clock();

	for (int rep = 0; rep < 2; rep++)
	{
		setupRandomCentroids(data);
		for (int i = 0; i < DATASET_SIZE; i++)
		{
			data.points_k[i] = -1;
		}

		CONVERGED = false;
		clock_t tic = clock();
		performRounds(grid, block, data, partitionSize);
		clock_t toc = clock();

#ifdef PRINT_CENTERS
		printf("execution time: %f (dataset load: %f)\n", (double)(toc - tic) / CLOCKS_PER_SEC, (double)(ds_toc - ds_tic) / CLOCKS_PER_SEC);
#else
		printf("%f\n", (double)(toc - tic) / CLOCKS_PER_SEC);
#endif
	}

	cudaFree(data.d_points_p_x);
	cudaFree(data.d_points_p_y);
	cudaFree(data.d_points_k);
	cudaFree(data.d_centroids_p_x);
	cudaFree(data.d_centroids_p_y);
	cudaFree(data.d_centroids_sums_x);
	cudaFree(data.d_centroids_sums_y);
	cudaFree(data.d_centroids_partition_lengths);

	delete[] data.centroids_p_x;
	delete[] data.centroids_p_y;
	delete[] data.centroids_sum_x;
	delete[] data.centroids_sum_y;
	delete[] data.centroids_partition_lengths;

	return 0;
}
