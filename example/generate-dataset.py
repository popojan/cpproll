#!/usr/bin/env python3

from sklearn.datasets import make_classification
import sys
import random

N1 = int(sys.argv[1])
N2 = int(sys.argv[2])
seed = int(sys.argv[3])

data = make_classification(N1+N2, class_sep = 1.0, n_informative = 3, random_state=seed)

def save(fname, data, idx):
    with open(fname, "w") as fout:
        for x1, y1 in zip(data[0][idx], data[1][idx]):
            print("%f" % y1, file = fout, end=' ')
            print("|f %s" % (" ".join(["%d:%f" % (i, v) for i, v in enumerate(x1)])), file=fout, end=' ')
            print("", file=fout)

save("train.dat", data, range(0, N1))
save("test.dat",  data, range(N1, N1+N2))
