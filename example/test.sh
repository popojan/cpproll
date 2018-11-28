#!/bin/bash

dir=`pwd`

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $DIR/../data

$DIR/../example/generate-dataset.py 20000 5000 666


PARAMS=(-b 18 -v info --l2 0.001 -j 1 -I f*f -B 1 -T 500)

TRAIN=("${PARAMS[@]}")
TRAIN+=(-f model -p blr.train.out --passes 1 -- train.dat)

TEST=("${PARAMS[@]}")
TEST+=(-i model -p blr.test.out -t --predict_sample 0 test.dat)

echo "Training..."

echo ./roll "${TRAIN[@]}"

"$DIR"/../build/roll "${TRAIN[@]}"

echo "Testing..."
echo ./roll "${TEST[@]}"

"$DIR"/../build/roll "${TEST[@]}"


