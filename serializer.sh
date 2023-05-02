#! /bin/bash

dsets=(
	10k
	100k
	1M
	10M
	100M
)
for ((i=0; i < ${#dsets[@]}; ++i));
do
	dataset="${dsets[$i]}"
	echo ${dataset}
	CSV_Preprocessor/x64/Release/CSV_Preprocessor.exe dataset/${dataset}.csv dataset/${dataset}.serialized
done
