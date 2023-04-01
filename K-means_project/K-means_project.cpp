// g++ main.cpp -o main
#include <stdio.h>

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

Point centroids[] = {
	{1, 1},
	{2, 2}
};

// distance squared between 2 points
// root square is not necessarry for distance comparison
// and is removeed as optimization
float distance(Point a, Point b) {
	float x = (b.x - a.x);
	float y = (b.y - a.y);
	return x * x + y * y;
}

int main() {
	for (int iter = 0; iter < 5; ++iter) { // stop condition
		for (int i = 0; i < (int)(sizeof(points) / sizeof(ClassedPoint)); ++i) { // for each point in dataset
			float min_d = 99999.99;
			int best_k = -1;
			for (int j = 0; j < (int)(sizeof(centroids) / sizeof(Point)); ++j) { // for each centroid
				float dist = distance(points[i].p, centroids[j]); // distance between point_i and centroid_j
				printf("\t\t(%f:%f) - (%f:%f) -> %f\n", points[i].p.x, points[i].p.y,
					centroids[j].x, centroids[j].y, dist);
				if (dist < min_d) {
					best_k = j;
					min_d = dist;
				}
			}
			points[i].k = best_k;
			printf("\tbest_k = %d\n", points[i].k);
		}
		for (int i = 0; i < (int)(sizeof(centroids) / sizeof(Point)); ++i) { // for each centroid
			int cluster_length = 0;
			int new_x = 0;
			int new_y = 0;
			for (int j = 0; j < (int)(sizeof(points) / sizeof(ClassedPoint)); ++j) { // for each point in dataset
				if (i == points[j].k) { // accumulate positions of points associated with this centroid
					new_x += points[j].p.x;
					new_y += points[j].p.y;
					cluster_length++;
				}
			}
			centroids[i].x = new_x / cluster_length;
			centroids[i].y = new_y / cluster_length;
			printf("iter %d) centroid %d (%f:%f) with %d elements\n", iter, i, centroids[i].x, centroids[i].y, cluster_length);
		}
		printf("\n");
	}
	return 0;
}
