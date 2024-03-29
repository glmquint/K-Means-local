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

#define PRINT_CENTERS_OFF

using namespace std;

int THREADS = 1;
#define STOPPING_ERROR 1e-2
bool CONVERGED = false;
int POINT_DIMENSION = 0;
int NUM_CLUSTERS = 2;
int DATASET_SIZE;

struct Point_s
{
	double *coords;
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

// distance squared between 2 points
// root square is not necessarry for distance comparison
// and is removeed as optimization
double distance(Point &a, Point &b)
{
	double sum_of_squares = 0;
	double diff_coord;
	for (int i = 0; i < POINT_DIMENSION; ++i)
	{
		diff_coord = a.coords[i] - b.coords[i];
		sum_of_squares += (diff_coord * diff_coord);
	}
	return sum_of_squares;
}

/*
void aggregatePoints(ClassedPoint *first_point, int partition_length, int thread_num)
{
	for (int i = 0; i < NUM_CLUSTERS; ++i)
	{ // reset cluster sum and count for this thread
		for (int k = 0; k < POINT_DIMENSION; ++k)
		{
			centroids[i].sum[thread_num].coords[k] = 0;
		}

		centroids[i].partition_lengths[thread_num] = 0;
	}
	for (int i = 0; i < partition_length; ++i)
	{
		for (int j = 0; j < POINT_DIMENSION; ++j)
		{
			centroids[first_point[i].k].sum[thread_num].coords[j] += first_point[i].p.coords[j];
		}
		centroids[first_point[i].k].partition_lengths[thread_num]++;
	}
}
*/

// worker function for each spawned thread
// first_point -> begin of dataset partition
// partition_legth -> partition length
// thread_num -> this thread id
void worker(ClassedPoint *first_point, int partition_length, int thread_num)
{
	double min_d;
	int best_k;
	double dist;
	Point *sum; // local point sum for each centroid for this thread
	int* points_per_centroid; // local number of points for each centroid for this thread

	sum = new Point[NUM_CLUSTERS];
	points_per_centroid = new int[NUM_CLUSTERS];
	for (int j = 0; j < NUM_CLUSTERS; ++j)
	{
		sum[j].coords = new double[POINT_DIMENSION];
		for (int k = 0; k < POINT_DIMENSION; ++k)
		{
			sum[j].coords[k] = 0;
		}
		points_per_centroid[j] = 0;
	}
	for (int i = 0; i < partition_length; ++i) { // for each point in partition
		min_d = 1.7976931348623157e+308; // +inf
		best_k = -1; // invalid k
		for (int j = 0; j < NUM_CLUSTERS; ++j)
		{													   // for each centroid
			dist = distance(first_point[i].p, centroids[j].p); // distance between point_i and centroid_j
			if (dist < min_d)
			{
				best_k = j;
				min_d = dist;
			}
		}
		first_point[i].k = best_k;
		for (int j = 0; j < POINT_DIMENSION; ++j)
		{
			sum[best_k].coords[j] += first_point[i].p.coords[j];
		}
		points_per_centroid[best_k]++;
	}
	// we access this global structure only once at thread termination
	// this reduces the number of mutual cache invalidations
	for(int i=0; i<NUM_CLUSTERS; i++){
		for(int j=0; j<POINT_DIMENSION; j++){
			centroids[i].sum[thread_num].coords[j] = sum[i].coords[j];
		}
		centroids[i].partition_lengths[thread_num] = points_per_centroid[i];
	}
}

// equal load distribution for each thread
void buildPartitions(ClassedPoint **first_points, int *partition_lengths)
{
	int reminder = DATASET_SIZE % THREADS;
	int points_per_thread = DATASET_SIZE / THREADS;
	for (int i = 0; i < THREADS; ++i)
	{
		partition_lengths[i] = points_per_thread;
		if (i == THREADS - 1)
			partition_lengths[i] += reminder;
		first_points[i] = &points[points_per_thread * i];
	}
}

// main thread must merge all results coming from every thread
void updateCenters()
{
	double max_err = numeric_limits<double>::min();
	for (int i = 0; i < NUM_CLUSTERS; ++i)
	{
		Point point_sum = {};
		point_sum.coords = new double[POINT_DIMENSION];
		for (int k = 0; k < POINT_DIMENSION; ++k)
		{
			point_sum.coords[k] = 0;
		}
		int sum_of_lengths = 0;
		for (int j = 0; j < THREADS; ++j)
		{
			for (int k = 0; k < POINT_DIMENSION; ++k)
			{
				point_sum.coords[k] += centroids[i].sum[j].coords[k];
			}
			sum_of_lengths += centroids[i].partition_lengths[j];
		}

		for (int j = 0; j < POINT_DIMENSION; ++j)
		{
			point_sum.coords[j] = point_sum.coords[j] / sum_of_lengths; // new centroid
		}
		double dist = distance(centroids[i].p, point_sum);
		for (int j = 0; j < POINT_DIMENSION; ++j)
		{
			centroids[i].p.coords[j] = point_sum.coords[j];
		}

		double error = dist;
		if (error > max_err)
		{
			max_err = error;
		}
	}
	CONVERGED = (max_err < STOPPING_ERROR);
}

// delegates work between thrads after proper load distribution
void performRounds(thread *threads, ClassedPoint **first_points, int *partition_lengths)
{
	int round = 0;
	while (!CONVERGED) // set by updateCenters()
	{
		for (int j = 0; j < THREADS; ++j) {
			for (int i = 0; i < NUM_CLUSTERS; ++i)
			{ // reset cluster sum and count for this thread
				for (int k = 0; k < POINT_DIMENSION; ++k)
				{
					centroids[i].sum[j].coords[k] = 0;
				}
				centroids[i].partition_lengths[j] = 0;
			}
		}
		if (THREADS != 1) // multithreading
		{
			for (int thread_i = 0; thread_i < THREADS; ++thread_i)
			{
				threads[thread_i] = thread(worker, first_points[thread_i], partition_lengths[thread_i], thread_i);
			}
			for (int thread_i = 0; thread_i < THREADS; ++thread_i)
			{
				threads[thread_i].join();
			}
		}
		else // single-thread (we reuse the main thread)
		{
			worker(first_points[0], partition_lengths[0], 0);
		}
		updateCenters();
		round++;
	}
}

// initial centroids are picked at random from the dataset
void generateRandomCentroids(int seed)
{
	srand(seed);
	for (int i = 0; i < NUM_CLUSTERS; ++i)
	{
		int random_index = rand() % (DATASET_SIZE);
		Centroid *c = new Centroid;
		c->p = *new Point;
		c->p.coords = new double[POINT_DIMENSION];
		for (int coord = 0; coord < POINT_DIMENSION; coord++)
		{
			c->p.coords[coord] = points[random_index].p.coords[coord];
		}
		c->sum = new Point[THREADS];
		c->partition_lengths = new int[THREADS];
		for (int j = 0; j < THREADS; ++j)
		{
			c->sum[j].coords = new double[POINT_DIMENSION];
			for (int k = 0; k < POINT_DIMENSION; ++k)
			{
				c->sum[j].coords[k] = 0;
			}
			c->partition_lengths[j] = 0;
		}
		centroids[i] = *c;
	}
}

// read dataset from a preprocessed serialized file format
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
		points[i].p.coords = new double[POINT_DIMENSION];
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
	if (argc < 4)
	{
		exit(1);
	}
	NUM_CLUSTERS = stoi(argv[2]);
	THREADS = stoi(argv[3]);
	centroids = new Centroid[NUM_CLUSTERS];
	thread *threads = new thread[THREADS];
	ClassedPoint **first_points = new ClassedPoint *[THREADS];
	int *partition_lengths = new int[THREADS];

	deserializePoints(argv[1]);
	buildPartitions(first_points, partition_lengths);
	for (int rep = 0; rep < 10; rep++) // multiple repetitions hide the initial single-threaded file reading cost
	{
		generateRandomCentroids(rep);
		for (int i = 0; i < DATASET_SIZE; i++)
		{
			points[i].k = -1;
		}

		CONVERGED = false;
		clock_t tic = clock();
		performRounds(threads, first_points, partition_lengths);
		clock_t toc = clock();
		printf("%f\n", (double)(toc - tic)/CLOCKS_PER_SEC);
		
#ifdef PRINT_CENTERS
		for (int i = 0; i < NUM_CLUSTERS; i++) {
			printf("Centro %d : ", i);
			for (int j = 0; j < POINT_DIMENSION; j++) {
				printf("%f ", centroids[i].p.coords[j]);
			}
			printf("\n");
		}
#endif
		
	}
	return 0;
}