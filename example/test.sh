#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DATA_DIR=$DIR/../data
mkdir -p $DATA_DIR
cd $DATA_DIR

$DIR/../example/generate-dataset.py 20000 5000 666


PARAMS=(-b 18 -v info --l2 1e-4 -j 4 -I f*f -B 8 -T 500)

TRAIN=("${PARAMS[@]}")
TRAIN+=(-f model -p blr.train.out --passes 3 -- train.dat)

TEST=("${PARAMS[@]}")
TEST+=(-i model -p blr.test.out -t --predict_sample 10 --explain -- test.dat)

echo "Training..."

echo ./roll "${TRAIN[@]}"

"$DIR"/../build/roll "${TRAIN[@]}"

echo "Testing..."
echo ./roll "${TEST[@]}"

"$DIR"/../build/roll "${TEST[@]}"


