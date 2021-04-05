#ifndef _H_STAGE_2
#define _H_STAGE_2


#include <buffer.h>
#include "spdlog/spdlog.h"
#include <sstream>
#include <string>
#include <cmath>
#include <fstream>
#include <utility>
#include "eigen.h"
#include <regex>
#include "options.h"

/**
* parsing and hashing worker
*/

struct Batch {
    std::map<size_t, size_t> nzf;
    Eigen::MatrixXd x;
    Eigen::VectorXd labels;
    std::vector<std::string> desc;
    std::vector<std::string> names;
    std::vector<size_t> exids;
    size_t baselineId;
};


template <class T, class S>
class Stage2
{
    Buffer<T>&  ibuffer_;
    Buffer<S>&  obuffer_;
    std::vector<std::vector<std::string>> interact;
    const Options& opts;
    std::shared_ptr<spdlog::logger> console;
public:
    Stage2(Buffer<T>& ibuffer, Buffer<S>& obuffer, const Options& opts);
    void run();
protected:
    void compile(std::regex& regex, const std::string& sregex, const std::string& name);
};

#endif
