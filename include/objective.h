#ifndef _H_OBJECTIVE
#define _H_OBJECTIVE

#include <lbfgs.h>
#include <memory>
#include <spdlog/spdlog.h>
#include <utility>

#include "eigen.h"

typedef double Tfloat;

class objective_function {

protected:
    lbfgsfloatval_t *w;
    size_t kk;
    std::shared_ptr<spdlog::logger> console;
    Eigen::VectorXd label;
    Eigen::MatrixXd xx, mj, sj;
public:

    objective_function(const Eigen::MatrixXd& x, const Eigen::MatrixXd& label, const Eigen::MatrixXd& mj, const Eigen::MatrixXd& sj);

    virtual ~objective_function();

    std::pair<int, size_t> run(Eigen::MatrixXd& w0);

    Eigen::VectorXd predict(const Eigen::VectorXd& wj);

    Eigen::MatrixXd predict_sample(const size_t n);

protected:
    static lbfgsfloatval_t _evaluate(
        void *instance,
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
        )
    {
        return reinterpret_cast<objective_function*>(instance)->evaluate(x, g, n, step);
    }

    lbfgsfloatval_t evaluate(
        const lbfgsfloatval_t *x,
        lbfgsfloatval_t *g,
        const int n,
        const lbfgsfloatval_t step
     );

    static int _progress(
        void *instance,
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
        )
    {
        return reinterpret_cast<objective_function*>(instance)->progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
    }

    int progress(
        const lbfgsfloatval_t *x,
        const lbfgsfloatval_t *g,
        const lbfgsfloatval_t fx,
        const lbfgsfloatval_t xnorm,
        const lbfgsfloatval_t gnorm,
        const lbfgsfloatval_t step,
        int n,
        int k,
        int ls
    );
};

#endif
