# cpproll

Simple machine learning tool optimizing logistic loss, coded according to Adroll [blog post](http://tech.adroll.com/blog/data-science/2017/03/06/thompson-sampling-bayesian-factorization-machines.html) and inspired by [vowpal wabbit](https://github.com/VowpalWabbit/vowpal_wabbit) data format and command line options. Useful for larger click prediction tasks. Usually beats vowpal on our data both in AUC and logloss, but training is much slower and throughput decays in time.

## Features
*   feature hashing with adjustable seed 
*   arbitrary feature interactions
*   logloss and roc auc reporting 
*   explain predictions
*   save model and continue training later
*   multi-threaded parsing
*   running mean and variance standardization
*   fast export of hashed features in svmlight format
*   save model and report metrics even after user termination

### Sample call
```bash
./roll -f model -j 4 -b 26 -v info --l2 0.01 --passes 1 -B 4 --log "o\^" -T 1500 --standardize -I "q*t,s*s,s*Q" train.vw
```

### Sample output
```text
...
[10:07:12.446] [info]   0.065375   0.063594     0.0049    0     231 feat    1795624 ex    10063 ex/s   12 it/ex
[10:07:13.948] [info]   0.065411   0.069606     0.0338    0     163 feat    1811024 ex    10260 ex/s   12 it/ex
[10:07:15.449] [info]   0.065418   0.066301     0.0038    0     237 feat    1826328 ex    10196 ex/s   13 it/ex
[10:07:16.950] [info]   0.065373   0.059995     0.0087    0     162 feat    1841888 ex    10366 ex/s   12 it/ex
[10:07:18.451] [info]   0.065341   0.061605     0.0428    0     201 feat    1857484 ex    10390 ex/s   12 it/ex
[10:07:19.952] [info]   0.065324   0.063255     0.0036    0     191 feat    1872516 ex    10015 ex/s   13 it/ex
[10:07:21.454] [info]   0.065331   0.066145     0.0311    0     197 feat    1887144 ex     9746 ex/s   12 it/ex
[10:07:22.955] [info]   0.065292   0.060226     0.0615    0     239 feat    1901724 ex     9714 ex/s   13 it/ex
[10:07:24.456] [info]   0.065316   0.068512     0.0236    0     188 feat    1916440 ex     9804 ex/s   13 it/ex
[10:07:25.957] [info]   0.065300   0.063132     0.0231    0     212 feat    1931104 ex     9769 ex/s   13 it/ex
[10:07:27.459] [info]   0.065300   0.065311     0.0225    0     334 feat    1945188 ex     9383 ex/s   13 it/ex
[10:07:28.960] [info]   0.065272   0.061542     0.0108    0     231 feat    1959476 ex     9519 ex/s   13 it/ex
[10:07:30.461] [info]   0.065251   0.062319     0.0116    0     153 feat    1973904 ex     9612 ex/s   13 it/ex
^
[10:07:31.104] [info] User termination in progress.
[10:07:31.107] [info] Average loss 0.065244, improvement +10.70 % over 0.073059, best constant [0.0139] baseline.
[10:07:31.343] [info] Global auROC 0.788851.
```
## Dependencies

Díky Véno!

```
protobuf-compiler
libeigen3-dev
liblbfgs-dev
libboost-serialization-dev
libboost-iostreams-dev
libboost-regex-dev
libtool
automake
cmake
```

## Copy & paste credits

*   [auROC calculation](https://github.com/vbalnt/cppROC)
*   [parse using boost.spirit.Qi](https://stackoverflow.com/questions/5678932/fastest-way-to-read-numerical-values-from-text-file-in-c-double-in-this-case)
*   [multithreaded parsing](https://codereview.stackexchange.com/questions/84109/a-multi-threaded-producer-consumer-with-c11)
*   [eigen matrix serialization](https://stackoverflow.com/questions/18382457/eigen-and-boostserialize)
*   [accurate running variance](https://www.johndcook.com/blog/standard_deviation/)
*   [argsort](https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes)
