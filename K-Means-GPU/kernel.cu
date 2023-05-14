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

using namespace std;
clock_t tic, toc;

int THREADS = 1;
double STOPPING_VARIANCE = 0.05;
bool CONVERGED = false;
int POINT_DIMENSION = 2;
int NUM_CLUSTERS = 2;
int DATASET_SIZE;
int THREADS_PER_BLOCK = 1024;

struct Point_s
{
	double coords[2] ;
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
	Point sum;				// size(sum) == THREADS;
	int partition_lengths; // size(partition_lengths) == THREADS;
};
typedef struct Centroid_s Centroid;

ClassedPoint* points;
Centroid* centroids;
ClassedPoint* d_points;
Centroid* d_centroids;

// distance squared between 2 points
// root square is not necessarry for distance comparison
// and is removeed as optimization
__device__ double distance(Point& a, Point& b)
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

double distanceCPU(Point& a, Point& b)
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

/*
void aggregatePoints(ClassedPoint* first_point, int partition_length, int thread_num)
{
	for (int i = 0; i < NUM_CLUSTERS; ++i)
	{ // reset cluster sum and count for this thread
		for (int k = 0; k < POINT_DIMENSION; ++k)
		{
			centroids[i].sum[thread_num].coords[k] = 0;
		}

		centroids[i].partition_lengths[thread_num] = 0;
	}
	for (int i = 0; i < partition_length; ++i) {
		for (int j = 0; j < POINT_DIMENSION; ++j) {
			centroids[first_point[i].k].sum[thread_num].coords[j] += first_point[i].p.coords[j];
		}
		centroids[first_point[i].k].partition_lengths[thread_num]++;
	}
}
*/


#if __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

__global__ void worker(ClassedPoint* d_point, Centroid* d_centr, int dataset_size, int num_clusters)
{
	double min_d = 1.7976931348623157e+308; // +inf
	int best_k = -1;
	double dist = 0;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < dataset_size) {
		for (int i = 0; i < num_clusters; ++i) {
			dist = distance(d_point[index].p, d_centr[i].p);
			if (dist < min_d) {
				min_d = dist;
				best_k = i;
			}
		}
		d_point[index].k = best_k;
		for (int i = 0; i < 2; ++i) {
			double* ptr = &(d_centr[best_k].sum.coords[i]);
			double val = d_point[index].p.coords[i];
			atomicAddDouble(ptr, val);
		}
		atomicAdd(&(d_centr[best_k].partition_lengths), 1);
	}
}

__global__ void worker_test(ClassedPoint* d_pnt, Centroid* d_centr, int dataset_size, int num_clusters) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < dataset_size) {
		//printf("point %d (%f %f)\n", index, d_pnt[index].p.coords[0], d_pnt[index].p.coords[1]);
		if (index == 1){
			for (int i = 0; i < num_clusters; ++i) {
				printf("centroid %i (%f %f)\n", i, d_centr[i].p.coords[0], d_centr[i].p.coords[1]);
			}
		}
	}
}

void updateCenters()
{
	double max_var = numeric_limits<double>::min();
	for (int i = 0; i < NUM_CLUSTERS; ++i)
	{
		Point point_sum = {};
		//point_sum.coords = new double[POINT_DIMENSION];
		for (int k = 0; k < POINT_DIMENSION; ++k)
		{
			point_sum.coords[k] = 0;
		}
		int sum_of_lengths = 0;
		for (int k = 0; k < POINT_DIMENSION; ++k)
		{
			point_sum.coords[k] += centroids[i].sum.coords[k];
		}
		sum_of_lengths += centroids[i].partition_lengths;

		double point_sum_square_norm = 0;
		for (int j = 0; j < POINT_DIMENSION; ++j)
		{
			point_sum.coords[j] = point_sum.coords[j] / sum_of_lengths; // new centroid
			point_sum_square_norm += (point_sum.coords[j] * point_sum.coords[j]);
		}
		double dist = distanceCPU(centroids[i].p, point_sum);
		for (int j = 0; j < POINT_DIMENSION; ++j)
		{
			centroids[i].p.coords[j] = point_sum.coords[j];
		}

		double variance = dist / point_sum_square_norm;
		if (variance > max_var)
		{
			max_var = variance;
		}
		// printf("iter %d) centroid %d (%f:%f) with %d elements\n", iter, i, centroids[i].p.coords[0], centroids[i].p.coords[1], sum_of_lengths);
	}
	CONVERGED = (max_var < STOPPING_VARIANCE);
	//CONVERGED = true;
}

vector<double> distance_calls;

void performRounds(dim3 grid, dim3 block)
{
	int round = 0;
	while (!CONVERGED)
	{
		// for (int thread_i = 0; thread_i < THREADS; ++thread_i)
		// {
		// 	threads[thread_i] = thread(worker, first_points[thread_i], partition_lengths[thread_i], thread_i);
		// }
		cudaMemcpy(d_centroids, centroids, NUM_CLUSTERS * sizeof(Centroid), cudaMemcpyHostToDevice);
		//for (int i = 0; i < DATASET_SIZE; i+=CHUNK_SIZE)
		worker <<< grid, block>>> (d_points, d_centroids, DATASET_SIZE, NUM_CLUSTERS);
		cudaDeviceSynchronize();
		cudaMemcpy(centroids, d_centroids, NUM_CLUSTERS * sizeof(Centroid), cudaMemcpyDeviceToHost);
		/*
		int count = 0;
		for (int i = 0; i < NUM_CLUSTERS; ++i) {
			count += centroids[i].partition_lengths;
		}
		assert(count == DATASET_SIZE, "didn't count enough points\n");
		*/
		updateCenters();
		round++;
		//printf("%f\n", round, elapsed);
	}
	// printf("took %d rounds\n", round);
}

void generateRandomCentroids()
{
	srand(69420);
	for (int i = 0; i < NUM_CLUSTERS; ++i)
	{
		int random_index = rand() % (DATASET_SIZE);
		Centroid* c = new Centroid;
		//c->p = *new Point;
		//c->p.coords = new double[POINT_DIMENSION];
		for (int coord = 0; coord < POINT_DIMENSION; coord++)
		{
			c->p.coords[coord] = points[random_index].p.coords[coord];
		}
		//c->sum = *new Point;
		//c->partition_lengths = *new int;
			//c->sum[j].coords = new double[POINT_DIMENSION];
		for (int k = 0; k < POINT_DIMENSION; ++k)
		{
			c->sum.coords[k] = 0;
		}
		c->partition_lengths = 0;
		centroids[i] = *c;
	}
}

void deserializePoints(char* intput_file)
{
	ifstream infile;
	infile.open(intput_file, ios::in | ios::binary);
	if (infile.fail())
	{
		cout << "can't find file " << intput_file << endl;
		exit(1);
	}
	infile.read((char*)(&DATASET_SIZE), sizeof(DATASET_SIZE));
	points = new ClassedPoint[DATASET_SIZE];
	infile.read((char*)(&POINT_DIMENSION), sizeof(POINT_DIMENSION));
	for (int i = 0; i < DATASET_SIZE; i++)
	{
		//points[i].p.coords = new double[POINT_DIMENSION];
		points[i].k = -1;
		for (int j = 0; j < POINT_DIMENSION; ++j)
		{
			infile.read((char*)(&points[i].p.coords[j]), sizeof(double));
		}
	}
	infile.close();
}

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		printf("[USAGE]: %s dataset.serialized num_clusters\n", argv[0]);
		exit(1);
	}
	NUM_CLUSTERS = stoi(argv[2]);
	centroids = new Centroid[NUM_CLUSTERS];

	deserializePoints(argv[1]);
	dim3 grid((DATASET_SIZE + THREADS_PER_BLOCK - 1 / THREADS_PER_BLOCK), 1, 1);
	dim3 block(THREADS_PER_BLOCK, 1, 1);
	// device copy
	cudaMalloc((void**)&d_points, DATASET_SIZE*sizeof(ClassedPoint));
	cudaMalloc((void**)&d_centroids, NUM_CLUSTERS*sizeof(Centroid));
	for (int rep = 0; rep < 1; rep++)
	{
		generateRandomCentroids();
		for (int i = 0; i < DATASET_SIZE; i++)
		{
			points[i].k = -1;
		}

		CONVERGED = false;
		// copy from host to device
		clock_t tic = clock();
		cudaMemcpy(d_points, points, DATASET_SIZE * sizeof(ClassedPoint), cudaMemcpyHostToDevice);
		performRounds(grid, block);
		clock_t toc = clock();
		printf("%f\n", (double) (toc-tic)/CLOCKS_PER_SEC);
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

		/*
		for (int i = 0; i < NUM_CLUSTERS; i++) {
			printf("Centro %d : ", i);
			for (int j = 0; j < POINT_DIMENSION; j++) {
				printf("%f ", centroids[i].p.coords[j]);
			}
			printf("\n");
		}
		*/

	}
	cudaFree(d_points);
	cudaFree(d_centroids);
	/*
	double mean = 0;
	for (int i = 0; i < distance_calls.size(); ++i) {
		mean += distance_calls.at(i);
	}
	mean /= distance_calls.size();
	//printf("%f", mean);
	*/
	return 0;
}
