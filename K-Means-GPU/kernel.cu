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
#define STOPPING_ERROR 1e-2

using namespace std;
clock_t tic, toc;

int THREADS = 1;
bool CONVERGED = false;
int POINT_DIMENSION = 2;
int NUM_CLUSTERS = 2;
int DATASET_SIZE;
int THREADS_PER_BLOCK;

struct Point_s
{
	double coords[2];
};
typedef struct Point_s Point;

struct ClassedPoint_s
{
	Point p;
	int k;
};
typedef struct ClassedPoint_s ClassedPoint;

struct Centroid_s
{
	Point p;
	Point *sum;				// size(sum) == THREADS;
	int *partition_lengths; // size(partition_lengths) == THREADS;
};
typedef struct Centroid_s Centroid;

ClassedPoint *points;
Centroid *centroids;
ClassedPoint *d_points;
Point *d_centroids;
Point *d_centroids_sums;
int *d_centroids_plengths;
#ifdef PREALLOC_OPTIMIZE
Point* d_sum;
int* d_points_per_centroid;
#endif // PREALLOC_OPTIMIZE

// distance squared between 2 points
// root square is not necessarry for distance comparison
// and is removeed as optimization
__device__ double distance(Point &a, Point &b)
{
	double sum_of_squares = 0;
	double diff_coord;
	for (int i = 0; i < 2; ++i)
	{
		diff_coord = a.coords[i] - b.coords[i];
		sum_of_squares += (diff_coord * diff_coord);
	}
	return sum_of_squares;
}

double distanceCPU(Point &a, Point &b)
{
	double sum_of_squares = 0;
	double diff_coord;
	for (int i = 0; i < 2; ++i)
	{
		diff_coord = a.coords[i] - b.coords[i];
		sum_of_squares += (diff_coord * diff_coord);
	}
	return sum_of_squares;
}

#if __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double *address, double val)
{
	unsigned long long int *address_as_ull = (unsigned long long int *)address;
	unsigned long long int old = *address_as_ull, assumed;
	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);
	return __longlong_as_double(old);
}
#endif

__global__ void worker(ClassedPoint *d_point, Point *d_centr, Point* d_centroids_sums, int * d_centroids_plengths, int dataset_size, 
int num_clusters, int partition_size, int num_threads
#ifdef PREALLOC_OPTIMIZE
, Point* d_sum, int* d_points_per_centroid
#endif // PREALLOC_OPTIMIZE
)
{
	double dist = 0;
	int best_k;
	double min_d;
	Point *sum;
	int *points_per_centroid;

#ifdef PREALLOC_OPTIMIZE
	sum = &d_sum[(blockDim.x * blockIdx.x + threadIdx.x)*num_clusters];
	points_per_centroid = &d_points_per_centroid[(blockDim.x * blockIdx.x + threadIdx.x)*num_clusters];
#else
	sum = new Point[num_clusters];
	points_per_centroid = new int[num_clusters];
#endif // PREALLOC_OPTIMIZE

	for (int j = 0; j < num_clusters; ++j)
	{
		for (int k = 0; k < 2; ++k)
		{
			sum[j].coords[k] = 0;
		}
		points_per_centroid[j] = 0;
	}

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int elem = 0; elem < partition_size; ++elem)
	{
		int partition_elem = partition_size * index + elem;
		if (partition_elem < dataset_size)
		{
			min_d = 1.7976931348623157e+308; // +inf
			best_k = -1;
			for (int i = 0; i < num_clusters; ++i)
			{
				dist = distance(d_point[partition_elem].p, d_centr[i]);
				/*
				if (dist < min_d)
				{
					min_d = dist;
					best_k = i;
				}
				*/
				best_k = i * (dist < min_d) + best_k * (dist >= min_d);
				min_d = dist * (dist < min_d) + min_d * (dist >= min_d);
			}
			d_point[partition_elem].k = best_k;
			for (int i = 0; i < 2; ++i)
			{
				sum[best_k].coords[i] += d_point[partition_elem].p.coords[i];
			}
			points_per_centroid[best_k]++;
		}
	}
	//printf("%d) %f\n", index, sum[0].coords[0]);
	for (int i = 0; i < num_clusters; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			d_centroids_sums[i * num_threads + index].coords[j] = sum[i].coords[j];
		}
		d_centroids_plengths[i * num_threads + index] = points_per_centroid[i];
	}
	//printf("%d} %f\n", index, d_centroids_sums[0 + index].coords[0]);
	// ok????
#ifndef PREALLOC_OPTIMIZE
	delete[] sum;
	delete[] points_per_centroid;
#endif // !PREALLOC_OPTIMIZE

}

void updateCenters()
{
	double max_err = numeric_limits<double>::min();
	for (int i = 0; i < NUM_CLUSTERS; ++i)
	{
		Point point_sum = {};
		// point_sum.coords = new double[POINT_DIMENSION];
		for (int k = 0; k < POINT_DIMENSION; ++k)
		{
			point_sum.coords[k] = 0;
		}
		int sum_of_lengths = 0;

		for (int j = 0; j < THREADS; j++)
		{
			for (int k = 0; k < POINT_DIMENSION; ++k)
			{
				point_sum.coords[k] += centroids[i].sum[j].coords[k];
			}
			// point_sum.coords[k] = centroids[i].sum.coords[k];
			sum_of_lengths += centroids[i].partition_lengths[j];
		}

		//double point_sum_square_norm = 0;
		for (int j = 0; j < POINT_DIMENSION; ++j)
		{
			point_sum.coords[j] /= sum_of_lengths; // new centroid
			//point_sum_square_norm += (point_sum.coords[j] * point_sum.coords[j]);
		}
		double dist = distanceCPU(centroids[i].p, point_sum);
		for (int j = 0; j < POINT_DIMENSION; ++j)
		{
			centroids[i].p.coords[j] = point_sum.coords[j];
		}

		double error = sqrt(dist); // point_sum_square_norm;
		if (error > max_err)
		{
			max_err = error;
		}
#ifdef PRINT_CENTERS
		printf("centroid %d (%f:%f) with %d elements (error: %f)\n", i, centroids[i].p.coords[0], centroids[i].p.coords[1], sum_of_lengths, error);
#endif
	}
#ifdef PRINT_CENTERS
	printf("==================================================\n");
#endif
	CONVERGED = (max_err < STOPPING_ERROR);
	// CONVERGED = true;
}

void performRounds(dim3 grid, dim3 block, int partition_size)
{
	int round = 0;
	while (!CONVERGED)
	{
		// for (int thread_i = 0; thread_i < THREADS; ++thread_i)
		// {
		// 	threads[thread_i] = thread(worker, first_points[thread_i], partition_lengths[thread_i], thread_i);
		// }
		// cudaMemcpy(d_centroids, centroids, NUM_CLUSTERS * sizeof(Centroid), cudaMemcpyHostToDevice);
		cudaError_t cerr;
		for (int i = 0; i < NUM_CLUSTERS; ++i) {
			cerr = cudaMemcpy(&d_centroids[i], &centroids[i], sizeof(Point), cudaMemcpyHostToDevice);
		}
		assert(cerr == cudaSuccess);
		worker<<<grid, block>>>(d_points, d_centroids, d_centroids_sums, d_centroids_plengths, DATASET_SIZE, NUM_CLUSTERS, partition_size, THREADS
#ifdef PREALLOC_OPTIMIZE
		, d_sum, d_points_per_centroid
#endif // PREALLOC_OPTIMIZE
		);
		cudaDeviceSynchronize();
		
		for (int i = 0; i < NUM_CLUSTERS; ++i)
		{
			cerr = cudaMemcpy(centroids[i].sum, &d_centroids_sums[i*THREADS], THREADS * sizeof(Point), cudaMemcpyDeviceToHost);
			assert(cerr == cudaSuccess);
			cerr = cudaMemcpy(centroids[i].partition_lengths, &d_centroids_plengths[i*THREADS], THREADS * sizeof(int), cudaMemcpyDeviceToHost);
			assert(cerr == cudaSuccess);
		}
		/*
		int count = 0;
		for (int i = 0; i < NUM_CLUSTERS; ++i) {
			count += centroids[i].partition_lengths;
		}
		assert(count == DATASET_SIZE, "didn't count enough points\n");
		*/
		updateCenters();
		round++;
		// printf("%f\n", round, elapsed);
	}
#ifdef PRINT_CENTERS
	 printf("took %d rounds\n", round);
#endif
}

void setupRandomCentroids()
{

	srand(69420);
	for (int i = 0; i < NUM_CLUSTERS; ++i)
	{
		int random_index = rand() % (DATASET_SIZE);
		for (int coord = 0; coord < POINT_DIMENSION; coord++)
		{
			centroids[i].p.coords[coord] = points[random_index].p.coords[coord];
		}
		for (int j = 0; j < THREADS; j++)
		{
			for (int k = 0; k < POINT_DIMENSION; ++k)
			{
				centroids[i].sum[j].coords[k] = 0;
			}
			centroids[i].partition_lengths[j] = 0;
		}
	}
}

void generateRandomCentroids()
{
	centroids = new Centroid[NUM_CLUSTERS];
	for (int i = 0; i < NUM_CLUSTERS; ++i){
		//Centroid* c = new Centroid;
		//c->p = *new Point;
		//c->p.coords = new double[POINT_DIMENSION];
		centroids[i].sum = new Point[THREADS];
		centroids[i].partition_lengths = new int[THREADS];
		//c->sum[j].coords = new double[POINT_DIMENSION];
		for (int j = 0; j < THREADS; j++){
			for (int k = 0; k < POINT_DIMENSION; ++k){
				centroids[i].sum[j].coords[k] = 0;
			}
			centroids[i].partition_lengths[j] = 0;
		}
	}
}

void deserializePoints(char *intput_file)
{
	ifstream infile;
	infile.open(intput_file, ios::in | ios::binary);
	if (infile.fail())
	{
		cout << "can't find file " << intput_file << endl;
		exit(1);
	}
	infile.read((char *)(&DATASET_SIZE), sizeof(DATASET_SIZE));
	points = new ClassedPoint[DATASET_SIZE];
	infile.read((char *)(&POINT_DIMENSION), sizeof(POINT_DIMENSION));
	for (int i = 0; i < DATASET_SIZE; i++)
	{
		// points[i].p.coords = new double[POINT_DIMENSION];
		points[i].k = -1;
		for (int j = 0; j < POINT_DIMENSION; ++j)
		{
			infile.read((char *)(&points[i].p.coords[j]), sizeof(double));
		}
	}
	infile.close();
}

int main(int argc, char **argv)
{
	if (argc < 5)
	{
		printf("[USAGE]: %s dataset.serialized num_clusters num_threads threads_per_block\n", argv[0]);
		exit(1);
	}
	NUM_CLUSTERS = stoi(argv[2]);
	centroids = new Centroid[NUM_CLUSTERS];

	THREADS = stoi(argv[3]);
	THREADS_PER_BLOCK = stoi(argv[4]);

	deserializePoints(argv[1]);
	generateRandomCentroids();

	int num_blocks = THREADS / THREADS_PER_BLOCK;
	if (THREADS % THREADS_PER_BLOCK)
		num_blocks++;

	int partition_size;
	if (DATASET_SIZE % THREADS == 0)
	{
		partition_size = DATASET_SIZE / THREADS;
	}
	else
	{
		partition_size = DATASET_SIZE / (THREADS - 1);
	}

	dim3 grid(num_blocks, 1, 1);
	dim3 block(THREADS_PER_BLOCK, 1, 1);
	
	cudaMalloc((void **) &d_points, DATASET_SIZE * sizeof(ClassedPoint));
	cudaMalloc((void **) &d_centroids, NUM_CLUSTERS * sizeof(Point));
	cudaMalloc((void **) &d_centroids_sums, NUM_CLUSTERS * THREADS * sizeof(Point));
	cudaMalloc((void **) &d_centroids_plengths, NUM_CLUSTERS * THREADS * sizeof(int));

#ifdef PREALLOC_OPTIMIZE
	cudaMalloc((void **) &d_sum, NUM_CLUSTERS * THREADS * sizeof(Point));
	cudaMalloc((void **) &d_points_per_centroid, NUM_CLUSTERS * THREADS * sizeof(int));
#endif // PREALLOC_OPTIMIZE

	// must copy to device at each repetition
	// do it once for every repetition
	cudaError_t cerr;
	clock_t ds_tic = clock();
	cerr = cudaMemcpy(d_points, points, DATASET_SIZE * sizeof(ClassedPoint), cudaMemcpyHostToDevice);
	clock_t ds_toc = clock();
	assert(cerr == cudaSuccess);
	for (int rep = 0; rep < 30; rep++)
	{
		setupRandomCentroids();
		for (int i = 0; i < DATASET_SIZE; i++)
		{
			points[i].k = -1;
		}

		CONVERGED = false;
		// copy from host to device
		clock_t tic = clock();
		clock_t intermidiate_clock = clock();
		performRounds(grid, block, partition_size);
		clock_t toc = clock();
#ifdef PRINT_CENTERS
		printf("execution time: %f (dataset load %f)\n", (double)(toc - tic) / CLOCKS_PER_SEC, (double)(ds_toc - ds_tic) / CLOCKS_PER_SEC);
#else
		printf("Ok: %f\n", (double)(toc - intermidiate_clock) / CLOCKS_PER_SEC);
#endif
		/*
		for (int i = 0; i < NUM_CLUSTERS; ++i) {
			printf("(%f %f)\n", centroids[i].p.coords[0], centroids[i].p.coords[1]);
		}
		*/
		/*
		auto start = std::chrono::high_resolution_clock::now();
		performRounds(threads, first_points, partition_lengths);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> diff = end - start;
		printf("%f\n", diff.count());
		*/

#ifdef PRINT_CENTERS
		printf("/------------begin centroids-------------\\\n");
		for (int i = 0; i < NUM_CLUSTERS; i++)
		{
			printf("Centro %d : ", i);
			for (int j = 0; j < POINT_DIMENSION; j++)
			{
				printf("%f ", centroids[i].p.coords[j]);
			}
			printf("\n");
		}
		printf("\\------------end centroids---------------/\n");
#endif
	}
	cudaFree(d_points);
	cudaFree(d_centroids);
	cudaFree(d_centroids_sums);
	cudaFree(d_centroids_plengths);

	return 0;
}
