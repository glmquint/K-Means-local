// g++ main.cpp -o main
#include <stdio.h>
#include <thread>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <limits>
#include <string.h>

using namespace std;

int POINT_DIMENSION = 0;

struct Point_s {
	vector<double> coords; // maybe optimize with 1 allocated array
};
typedef struct Point_s Point;

struct Point_deserialized_s {
	double* coords;
};
typedef struct Point_deserialized_s Point_deserialized;

struct ClassedPoint_s {
	Point p;
	int k;
};
typedef struct ClassedPoint_s ClassedPoint;

vector<ClassedPoint> points;

struct ClassedPoint_deserialized_s {
	Point_deserialized p;
	int k;
};
typedef struct ClassedPoint_deserialized_s ClassedPoint_deserialized;

ClassedPoint_deserialized* points_deserialized;


void parseDataset(string line) {
	ClassedPoint classed_point = { {}, -1 };
	char* field;
	char* next_tok;
	field = strtok_r((char*)line.c_str(), ",", &next_tok); // change to strtok_s in windows
	double coord = 0;
	while (field != NULL) {
		coord = stod(field);
		classed_point.p.coords.push_back(coord);
		field = strtok_r(NULL, ",", &next_tok);
	}
	points.push_back(classed_point);
}

void readCSVFile(char* file_name) {
	fstream my_file;
	string line;
	my_file.open(file_name, ios::in);
	while (getline(my_file, line)) { // for each line
		parseDataset(line);
	}
}

void serializePoints(char* output_file) {
	ofstream outfile;
	outfile.open(output_file, ios::out | ios::binary);
	int point_size = points.size();
	cout << point_size << endl;
	outfile.write((char*)(&point_size), sizeof point_size);
	POINT_DIMENSION = points[0].p.coords.size();
	cout << POINT_DIMENSION << endl;
	outfile.write((char*)(&POINT_DIMENSION), sizeof POINT_DIMENSION);
	for (int i = 0; i < point_size; ++i) {
		for (int j = 0; j < POINT_DIMENSION; ++j) {
			outfile.write((char*)(&points[i].p.coords[j]), sizeof(points[i].p.coords[j]));
		}
	}
	outfile.close();
}

void deserializePoints(char* intput_file) {
	ifstream infile;
	infile.open(intput_file, ios::in | ios::binary);
	int point_size;
	infile.read((char *)(&point_size), sizeof(point_size));
	points_deserialized = new ClassedPoint_deserialized[point_size];
	int point_dimension;
	infile.read((char *)(&point_dimension), sizeof(point_dimension));
	for (int i = 0; i < point_size; i++) {
		points_deserialized[i].p.coords = new double[point_dimension];
		points_deserialized[i].k = 0;
		for (int j = 0; j < point_dimension; ++j) {
			infile.read((char *)(&points_deserialized[i].p.coords[j]), sizeof(points_deserialized[i].p.coords[j]));
		}
	}
	infile.close();
}

int main(int argc, char** argv) {
	if (argc < 3) {
		printf("Gimme the <csv file> <output file>\n");
		return 1;
	}
	readCSVFile(argv[1]);
	serializePoints(argv[2]);
	deserializePoints(argv[2]);
	return 0;
}
