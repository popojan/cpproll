#ifndef _H_OPTIONS
#define _H_OPTIONS


struct Options {
    int B = 18, passes = 1, period = 1000, batch = 1, seed = 0;
    std::string verbose, desc, svmlight, logfile, tblog;
    double lambda, intercept;
    std::vector<std::string> files;
    std::string interactions, relogit, relog, reignore, rekeep, rebaseline, fpred, fmodel, fmodelin;
    int jobs = 4, npredict = 0;
    bool standardize = false, testonly = false, explain = false;
    size_t N;
    double decay;
    Options(): verbose("info"), lambda(1.0), N(1<<B), decay(1.0) {}
};

#endif
