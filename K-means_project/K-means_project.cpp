// g++ main.cpp -o main
#include <stdio.h>
#include <thread>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <limits>

#define THREADS 4

using namespace std;

struct Point_s {
	vector<double> coords; // maybe optimize with 1 allocated array
};
typedef struct Point_s Point;

struct ClassedPoint_s {
	Point p;
	int k;
};
typedef struct ClassedPoint_s ClassedPoint;

struct Centroid_s {
	Point p;
	Point sum[THREADS];
	int partition_lengths[THREADS];
};
typedef struct Centroid_s Centroid;

int POINT_DIMENSION = 0;
int NUM_CLUSTERS = 10;

vector<ClassedPoint> points;
vector<Centroid> centroids;

// distance squared between 2 points
// root square is not necessarry for distance comparison
// and is removeed as optimization
double distance(Point a, Point b) {
	double sum_of_squares = 0;
	for (int i = 0; i < a.coords.size(); ++i) {
		double diff_coord = a.coords[i] - b.coords[i];
		sum_of_squares += (diff_coord * diff_coord);
	}
	return sum_of_squares;
}

void worker(ClassedPoint* first_point, int partition_length, int thread_num) {
	for (int i = 0; i < partition_length; ++i) { // for each point in partition
		double min_d = numeric_limits<double>::max();
		int best_k = -1;
		for (int j = 0; j < NUM_CLUSTERS; ++j) { // for each centroid
			double dist = distance(first_point[i].p, centroids[j].p); // distance between point_i and centroid_j
			// print works with only 2 dimensions
			// printf("\t\t(%f:%f) - (%f:%f) -> %f\n", first_point[i].p.coords[0], first_point[i].p.coords[1],
			//		 centroids[j].p.coords[0], centroids[j].p.coords[1], dist);
			 if (dist < min_d) {
				best_k = j;
				min_d = dist;
			}
		}
		first_point[i].k = best_k;
		// printf("\tbest_k = %d\n", first_point[i].k);
	}
	for (int i = 0; i < NUM_CLUSTERS; ++i) { // reset cluster sum and count for this thread
		Point p = {};
		centroids[i].sum[thread_num] = p;
		centroids[i].sum[thread_num].coords.resize(POINT_DIMENSION);
		centroids[i].partition_lengths[thread_num] = 0;
	}
	for (int i = 0; i < partition_length; ++i) {
		for (int j = 0; j < POINT_DIMENSION; ++j) {
			centroids[first_point[i].k].sum[thread_num].coords[j] += first_point[i].p.coords[j];
		}
		centroids[first_point[i].k].partition_lengths[thread_num]++;
	}
}

void build_partitions(ClassedPoint** first_points, int* partition_lengths) {
	int dataset_length = (int)points.size();
	int reminder = dataset_length % THREADS;
	int points_per_thread = dataset_length / THREADS;
	for (int i = 0; i < THREADS; ++i) {
		partition_lengths[i] = points_per_thread;
		if (i == THREADS - 1)
			partition_lengths[i] += reminder;
		first_points[i] = &points[points_per_thread * i];
	}
}

double STOPPING_VARIANCE = 0.1;
bool CONVERGED = false;

void update_centers() {
	double max_var = numeric_limits<double>::min();
	for (int i = 0; i < (int)centroids.size(); ++i) {
		Point point_sum = {};
		point_sum.coords.resize(POINT_DIMENSION);
		for (int k = 0; k < POINT_DIMENSION; ++k) {
			point_sum.coords[k] = 0;
		}
		int sum_of_lengths = 0;
		for (int j = 0; j < THREADS; ++j) {
			for (int k = 0; k < POINT_DIMENSION; ++k) {
				point_sum.coords[k] += centroids[i].sum[j].coords[k];
			}
			sum_of_lengths += centroids[i].partition_lengths[j];
		}
		
		double point_sum_square_norm = 0;
		for (int j = 0; j < POINT_DIMENSION; ++j) {
			point_sum.coords[j]  = point_sum.coords[j] / sum_of_lengths; // new centroid
			point_sum_square_norm += (point_sum.coords[j] * point_sum.coords[j]);
		}
		double dist = distance(centroids[i].p, point_sum);
		for(int j=0; j< POINT_DIMENSION; ++j){
			centroids[i].p.coords[j] = point_sum.coords[j];
		}
		
		double variance = dist / point_sum_square_norm;
		if (variance > max_var) {
			max_var = variance;
		}
		// printf("iter %d) centroid %d (%f:%f) with %d elements\n", iter, i, centroids[i].p.coords[0], centroids[i].p.coords[1], sum_of_lengths);
	}
	CONVERGED = (max_var < STOPPING_VARIANCE);

}

void performRounds(thread threads[], ClassedPoint** first_points, int* partition_lengths) {
	while (!CONVERGED){
		for (int thread_i = 0; thread_i < THREADS; ++thread_i) {
			threads[thread_i] = thread(worker, first_points[thread_i], partition_lengths[thread_i], thread_i);
		}
		for (int thread_i = 0; thread_i < THREADS; ++thread_i) {
			threads[thread_i].join();
		}
		update_centers();
	}
}

void readCSVFile(char* file_name, vector<ClassedPoint> *points) {
	fstream my_file;
	string line;
	string field;
	my_file.open(file_name, ios::in);
	while (getline(my_file, line)) { // for each line
		ClassedPoint classed_point = { {}, -1 };
		istringstream s(line);
		while (getline(s, field, ',')) { // for each comma-separated field
			double coord = 0;
			coord = stod(field);
			classed_point.p.coords.push_back(coord);
		}
		points->push_back(classed_point);
	}
}

void generateRandomCentroids() {
	srand(69420);
	for (int i = 0; i < NUM_CLUSTERS; ++i) {
		int random_index = rand() % (points.size());
		Centroid c = {};
		c.p = points[random_index].p;
		centroids.push_back(c);
	}
}

int main(int argc, char** argv) {
	if (argc < 2) {
		printf("Give me a dataset file\n");
		exit(1);
	}
	readCSVFile(argv[1], &points);
	POINT_DIMENSION = (int)points[0].p.coords.size();
	generateRandomCentroids();

	thread threads[THREADS];
	ClassedPoint* first_points[THREADS];
	int partition_lengths[THREADS];
	build_partitions(first_points, partition_lengths);
	performRounds(threads, first_points, partition_lengths);
	return 0;
}
