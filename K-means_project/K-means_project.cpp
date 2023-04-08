// g++ main.cpp -o main
#include <stdio.h>
#include <thread>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <limits>
#include <string>

using namespace std;

int THREADS = 1;
double STOPPING_VARIANCE = 0.1;
bool CONVERGED = false;
int POINT_DIMENSION = 0;
int NUM_CLUSTERS = 2;

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
	vector<Point> sum; // size(sum) == THREADS;
	vector<int> partition_lengths; // size(partition_lengths) == THREADS;
};
typedef struct Centroid_s Centroid;

vector<ClassedPoint> points;
vector<Centroid> centroids;

// distance squared between 2 points
// root square is not necessarry for distance comparison
// and is removeed as optimization
double distance(Point &a, Point &b) {
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

void buildPartitions(vector<ClassedPoint*> &first_points, vector<int> &partition_lengths) {
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

void updateCenters() {
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

void performRounds(vector<thread> &threads, vector<ClassedPoint*> first_points, vector<int> partition_lengths) {
	while (!CONVERGED){
		for (int thread_i = 0; thread_i < THREADS; ++thread_i) {
			threads[thread_i] = thread(worker, first_points[thread_i], partition_lengths[thread_i], thread_i);
		}
		for (int thread_i = 0; thread_i < THREADS; ++thread_i) {
			threads[thread_i].join();
		}
		updateCenters();
	}
}

void parseDataset(string line, vector<ClassedPoint> *points) {
	ClassedPoint classed_point = { {}, -1 };
	char* field;
	char* next_tok;
	field = strtok_s((char*)line.c_str(), ",", &next_tok);
	double coord = 0;
	while (field != NULL) {
		coord = stod(field);
		classed_point.p.coords.push_back(coord);
		field = strtok_s(NULL, ",", &next_tok);
	}
	points->push_back(classed_point);
}

void readCSVFile(char* file_name, vector<ClassedPoint> *points) {
	fstream my_file;
	string line;
	my_file.open(file_name, ios::in);
	while (getline(my_file, line)) { // for each line
		parseDataset(line, points);
	}
}

void generateRandomCentroids() {
	srand(69420);
	for (int i = 0; i < NUM_CLUSTERS; ++i) {
		int random_index = rand() % (points.size());
		Centroid c = {};
		c.p = points[random_index].p;
		c.sum.resize(THREADS);
		c.partition_lengths.resize(THREADS);
		centroids.push_back(c);
	}
}

int main(int argc, char** argv) {
	if (argc < 4) {
		printf("[USAGE]: %s dataset.csv num_clusters num_threads\n", argv[0]);
		exit(1);
	}
	NUM_CLUSTERS = stoi(argv[2]);
	THREADS = stoi(argv[3]);
	readCSVFile(argv[1], &points);
	POINT_DIMENSION = (int)points[0].p.coords.size();
	generateRandomCentroids();

	vector<thread> threads;
	threads.resize(THREADS);
	vector<ClassedPoint*> first_points;
	first_points.resize(THREADS);
	vector<int> partition_lengths;
	partition_lengths.resize(THREADS);
	buildPartitions(first_points, partition_lengths);
	performRounds(threads, first_points, partition_lengths);
	for(int i = 0; i<NUM_CLUSTERS; i++){
		printf("Centro %d : ", i);
		for(int j = 0; j<POINT_DIMENSION; j++){
			printf("%f ", centroids[i].p.coords[j]);
		}
		printf("\n");
	}
	return 0;
}
