#ifndef _H_STAGE_3
#define _H_STAGE_3

#include <buffer.h>
#include <spdlog/spdlog.h>
#include <unordered_map>
#include <chrono>
#include <array>

#include "objective.h"
#include "stats.h"
#include "options.h"
#include "roc.h"

typedef std::chrono::milliseconds TimeT;

/**
*  TODO actual machine learning goes here
*/

template <class T>
class Stage3
{
    Buffer<T>&  buffer_;
    Eigen::VectorXd mj, sj;
    RunningStat rs;
    ROC roc;
    double logloss_sum;
    double logloss_asum;
    double logloss_aden;
    double logloss_den;
    double duration_sum;
    double duration_den;
    double iterations_sum;
    double label_sum;
    std::shared_ptr<spdlog::logger> console;
    const Options& opts;
public:

    Stage3(Buffer<T>& buffer, const Options& opts);

    void run();
    void save(const std::string& fmodel);
    void load(const std::string& fmodel);
    void eval();
};

#endif
