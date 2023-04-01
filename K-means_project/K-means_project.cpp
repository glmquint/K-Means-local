// g++ main.cpp -o main
#include <stdio.h>
#include <thread>
using namespace std;
#define THREADS 2

// eventually a dynamic array with variable point dimensions
struct Point_s {
	float x;
	float y;
};
typedef struct Point_s Point;

struct ClassedPoint_s {
	Point p;
	int k;
};
typedef struct ClassedPoint_s ClassedPoint;

ClassedPoint points[] = {
	{{0, 1},-1},
	{{1, 0},-1},
	{{2, 3},-1},
	{{4, 5},-1},
	{{10,11},-1},
	{{11, 10},-1},
	{{11, 11},-1},
	{{10, 10},-1}
};

struct Centroid_s {
	Point p;
	Point sum[THREADS];
	int partition_lengths[THREADS];
};
typedef struct Centroid_s Centroid;

Centroid centroids[] = {
	{{1, 1}, {{0,0}}, {-1}},
	{{2, 2}, {{0,0}}, {-1}}
};

// distance squared between 2 points
// root square is not necessarry for distance comparison
// and is removeed as optimization
float distance(Point a, Point b) {
	float x = (b.x - a.x);
	float y = (b.y - a.y);
	return x * x + y * y;
}

void worker(ClassedPoint* first_point, int partition_length, Centroid* centroids, int num_clusters, int thread_num) {
	for (int i = 0; i < partition_length; ++i) { // for each point in partition
		float min_d = 999.99;
		int best_k = -1;
		for (int j = 0; j < num_clusters; ++j) { // for each centroid
			float dist = distance(first_point[i].p, centroids[j].p); // distance between point_i and centroid_j
			printf("\t\t(%f:%f) - (%f:%f) -> %f\n", first_point[i].p.x, first_point[i].p.y,
				centroids[j].p.x, centroids[j].p.y, dist);
			 if (dist < min_d) {
				best_k = j;
				min_d = dist;
			}
		}
		first_point[i].k = best_k;
		printf("\tbest_k = %d\n", first_point[i].k);
	}
	for (int i = 0; i < num_clusters; ++i) { // reset cluster sum and count for this thread
		centroids[i].sum[thread_num] = Point{ 0,0 };
		centroids[i].partition_lengths[thread_num] = 0;
	}
	for (int i = 0; i < partition_length; ++i) {
		centroids[first_point[i].k].sum[thread_num].x += first_point[i].p.x;
		centroids[first_point[i].k].sum[thread_num].y += first_point[i].p.y;
		centroids[first_point[i].k].partition_lengths[thread_num]++;
	}
}

void build_partitions(ClassedPoint** first_points, int* partition_lengths) {
	int dataset_length = (int)(sizeof(points) / sizeof(ClassedPoint));
	int reminder = dataset_length % THREADS;
	int points_per_thread = dataset_length / THREADS;
	for (int i = 0; i < THREADS; ++i) {
		partition_lengths[i] = points_per_thread;
		if (i == THREADS - 1)
			partition_lengths[i] += reminder;
		first_points[i] = &points[points_per_thread * i];
	}
}

void update_centers(int iter) {
	for (int i = 0; i < (int)(sizeof(centroids)/sizeof(Centroid)); ++i) {
		float sum_x = 0;
		float sum_y = 0;
		int sum_length = 0;
		for (int j = 0; j < THREADS; ++j) {
			sum_x += centroids[i].sum[j].x;
			sum_y += centroids[i].sum[j].y;
			sum_length += centroids[i].partition_lengths[j];
		}
		centroids[i].p.x = sum_x / sum_length;
		centroids[i].p.y = sum_y / sum_length;
		printf("iter %d) centroid %d (%f:%f) with %d elements\n", iter, i, centroids[i].p.x, centroids[i].p.y, sum_length);
	}
}

int main() {
	thread threads[THREADS];
	ClassedPoint* first_points[THREADS];
	int partition_lengths[THREADS];
	build_partitions(first_points, partition_lengths);
	for (int iter = 0; iter < 5; ++iter) { // stop condition
		for (int thread_i = 0; thread_i < THREADS; ++thread_i) {
			threads[thread_i] = thread(worker, first_points[thread_i], partition_lengths[thread_i], centroids, (int)(sizeof(centroids) / sizeof(Centroid)), thread_i);
		}
		for (int thread_i = 0; thread_i < THREADS; ++thread_i) {
			threads[thread_i].join();
		}
		update_centers(iter);
	}
	return 0;
}
