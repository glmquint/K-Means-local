#! /bin/bash

dsets=(
	10k
	100k
	1M
	10M
	100M
)
threads=(
    32
    64
    128
    256
    512
    1024
    2048
    4096
    8192
    10240
)
#for dataset in "${dsets[@]}";
for ((i=0; i < ${#dsets[@]}; ++i));
do
	dataset="${dsets[$i]}"
	for ((j=0; j < ${#threads[@]}; ++j));
	do
        th="${threads[$j]}"
		echo ${dataset} ${th} 128
		K-Means-GPU/kernel dataset/${dataset}.serialized 5 ${th} 128 > results-old/${i}-${dataset}-${th}.csv
	done
done
for i in $(ls results-old); do echo $i; cat results-old/$i | sed "s/\./,/g" > results/$i; done;
