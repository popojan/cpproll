#include "objective.h"

#include <unordered_map>
#include <limits>
#include <random>
#include "stage2.h"

using namespace Eigen;

std::vector<std::string> lbfgserr = {{
    "LBFGSERR_UNKNOWNERROR",
    "Logic error. LBFGSERR_LOGICERROR",
    "Insufficient memory. LBFGSERR_OUTOFMEMORY",
    "The minimization process has been canceled. LBFGSERR_CANCELED",
    "Invalid number of variables specified. LBFGSERR_INVALID_N",
    "Invalid number of variables (for SSE) specified. LBFGSERR_INVALID_N_SSE",
    "The array x must be aligned to 16 (for SSE). LBFGSERR_INVALID_X_SSE",
    "Invalid parameter lbfgs_parameter_t::epsilon specified. LBFGSERR_INVALID_EPSILON",
    "Invalid parameter lbfgs_parameter_t::past specified. LBFGSERR_INVALID_TESTPERIOD",
    "Invalid parameter lbfgs_parameter_t::delta specified. LBFGSERR_INVALID_DELTA",
    "Invalid parameter lbfgs_parameter_t::linesearch specified. LBFGSERR_INVALID_LINESEARCH",
    "Invalid parameter lbfgs_parameter_t::max_step specified. LBFGSERR_INVALID_MINSTEP",
    "Invalid parameter lbfgs_parameter_t::max_step specified. LBFGSERR_INVALID_MAXSTEP",
    "Invalid parameter lbfgs_parameter_t::ftol specified. LBFGSERR_INVALID_FTOL",
    "Invalid parameter lbfgs_parameter_t::wolfe specified. LBFGSERR_INVALID_WOLFE",
    "Invalid parameter lbfgs_parameter_t::gtol specified. LBFGSERR_INVALID_GTOL",
    "Invalid parameter lbfgs_parameter_t::xtol specified. LBFGSERR_INVALID_XTOL",
    "Invalid parameter lbfgs_parameter_t::max_linesearch specified. LBFGSERR_INVALID_MAXLINESEARCH",
    "Invalid parameter lbfgs_parameter_t::orthantwise_c specified. LBFGSERR_INVALID_ORTHANTWISE",
    "Invalid parameter lbfgs_parameter_t::orthantwise_start specified. LBFGSERR_INVALID_ORTHANTWISE_START",
    "Invalid parameter lbfgs_parameter_t::orthantwise_end specified. LBFGSERR_INVALID_ORTHANTWISE_END",
    "The line-search step went out of the interval of uncertainty. LBFGSERR_OUTOFINTERVAL",
    "A logic error occurred; alternatively, the interval of uncertainty became too small. LBFGSERR_INCORRECT_TMINMAX",
    "A rounding error occurred; alternatively, no line-search step satisfies the sufficient decrease and curvature conditions. LBFGSERR_ROUNDING_ERROR",
    "The line-search step became smaller than lbfgs_parameter_t::min_step. LBFGSERR_MINIMUMSTEP",
    "The line-search step became larger than lbfgs_parameter_t::max_step. LBFGSERR_MAXIMUMSTEP",
    "The line-search routine reaches the maximum number of evaluations. LBFGSERR_MAXIMUMLINESEARCH",
    "The algorithm routine reaches the maximum number of iterations. LBFGSERR_MAXIMUMITERATION",
    "Relative width of the interval of uncertainty is at most lbfgs_parameter_t::xtol. LBFGSERR_WIDTHTOOSMALL",
    "A logic error (negative line-search step) occurred. LBFGSERR_INVALIDPARAMETERS",
    "The current search direction increases the objective function value. LBFGSERR_INCREASEGRADIENT" }};

objective_function::objective_function(
    const Eigen::MatrixXd& x, const Eigen::VectorXd& label,
    const Eigen::VectorXd& mj, const Eigen::VectorXd& sj
):  w(0), console(spdlog::get("console")), label(label), xx(x), mj(mj), sj(sj)
{ }

objective_function::~objective_function()
{
    if (w != NULL) {
        lbfgs_free(w);
        w = NULL;
    }
}

std::pair<int, size_t> objective_function::run(VectorXd& w0)
{
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *w = lbfgs_malloc(w0.size());

    if (w == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return std::pair<int, size_t>(1, 0);
    }

    /* Initialize the variables. */
    lbfgsfloatval_t *pw = w;
    for(int i = 0; i < w0.size(); ++i) {
        *pw = w0(i);
        ++pw;
    }

    /*
        Start the L-BFGS optimization; this will invoke the callback functions
        evaluate() and progress() when necessary.
     */
    lbfgs_parameter_t params;
    lbfgs_parameter_init(&params);

    params.epsilon = 1e-2;
    params.linesearch = LBFGS_LINESEARCH_DEFAULT;

    int ret = lbfgs(w0.size(), w, &fx, _evaluate, _progress, this, &params);

    /* Report the result. */
    if(ret >= 0) {
        console->debug("L-BFGS optimization terminated with status code = {0}, fx = {1}, iterations = {2}", ret, fx, kk);
    }
    else {
        console->warn("L-BFGS error {0} {3}, fx = {1}, iterations = {2}", ret, fx, kk, lbfgserr[ret + 1024]);
    }
    for(int i = 0; i < w0.size(); ++i) {
        w0(i) = w[i];
    }
    if (w != NULL) {
        lbfgs_free(w);
        w = NULL;
    }
    return std::pair<int, size_t>(ret, kk);
}

VectorXd objective_function::predict(const VectorXd& wj) {
    return (1.0 + (-(xx*wj)).array().exp()).inverse().matrix();
}

MatrixXd objective_function::predict_sample(const size_t n) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> d;

    MatrixXd ret(MatrixXd::Zero(xx.rows(), n));
    for(size_t i = 0; i < n; ++i) {
        VectorXd wj;
        wj.resize(mj.size());
        for(long int j = 0; j < wj.size(); ++j) {
            auto params(std::normal_distribution<double>::param_type(mj(j), 1.0/sj(j)));
            wj(j) = d(gen, params);
        }
        auto pred = (1.0 + (-(xx*wj)).array().exp()).inverse().matrix();
        ret.col(i) = pred.transpose();
    }
    return ret;
}

lbfgsfloatval_t objective_function::evaluate(
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{
    Map<const VectorXd> wj(x, xx.cols());

    Map<VectorXd> grad(g, wj.size());

    grad = (wj - mj);

    ArrayXd ui((((-(xx*wj)).array().exp() + 1.0).inverse()));

    const double eps = 1e-20;
    double ret = 0.5 * (sj.array()*grad.array().square()).sum() - (label.array() == 0.0).select(1.0 - ui, ui).max(eps).min(1.0-eps).log().sum();
    grad = (sj.array()*grad.array()).matrix() + xx.transpose()*(ui-label.array()).matrix(); //grad
    return ret;

}

int objective_function::progress(
    const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,
    int n, int k, int ls)
{
    kk = k;
    return 0;
}
