#! /bin/bash

dsets=(
	10k
	100k
	1M
	10M
	100M
)
#for dataset in "${dsets[@]}";
for ((i=0; i < ${#dsets[@]}; ++i));
do
	dataset="${dsets[$i]}"
	for threads in $(seq -f "%02g" 1 20);
	#for threads in $(seq -f "%05g" 1000 1000 20000);
	do
		echo ${dataset} ${threads}
		K-means_project/x64/Release/K-means_project.exe dataset/${dataset}.serialized 5 ${threads} > results-old/${i}-${dataset}-${threads}.csv
	done
done
for i in $(ls results-old); do echo $i; cat results-old/$i | sed "s/\./,/g" > results/$i; done;
