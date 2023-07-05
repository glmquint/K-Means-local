#! /bin/bash

th_p_blocks=(
	32
	64
	128
	256
	512
    1024
)
threads=(
    00032
    00064
    00128
    00256
    00512
    01024
    02048
    04096
    08192
    10240
)
for ((i=0; i < ${#th_p_blocks[@]}; ++i));
do
	th_p_block="${th_p_blocks[$i]}"
	for ((j=0; j < ${#threads[@]}; ++j));
	do
        th="${threads[$j]}"
		echo 10M ${th} ${th_p_block}
		K-Means-GPU/x64/Release/K-Means-GPU.exe dataset/10M.serialized 5 ${th} ${th_p_block} > results-old/${i}-10M-${th}-${th_p_block}.csv
	done
done
for i in $(ls results-old); do echo $i; cat results-old/$i | sed "s/\./,/g" > results/$i; done;
