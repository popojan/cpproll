#include "objective.h"

#include <unordered_map>
#include <limits>
#include <random>
#include "stage2.h"
#include <optim.hpp>
#include <limits>

inline arma::mat sigm(const arma::mat& X) {
    return 1.0 / (1.0 + arma::exp(-X));
}

double blr(const arma::vec& vals_inp, void* opt_data) {
    return blr_fn_whess(vals_inp, 0, 0, opt_data);

}
double blr_fn(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data) {
    return blr_fn_whess(vals_inp, grad_out, 0, opt_data);
}

double blr_fn_whess(const arma::vec& vals_inp, arma::vec* grad_out, arma::mat* hess_out, void* opt_data)
{
    objective_function* objfn_data = reinterpret_cast<objective_function*>(opt_data);
    objfn_data->kk += 1;

    auto console = objfn_data->console;

    const arma::vec& Y = objfn_data->Y;
    const arma::mat& X = objfn_data->X;
    const arma::vec& sj = objfn_data->sj;
    const arma::vec& mj = objfn_data->mj;

    arma::vec mu = sigm(X*vals_inp);//, 1e-8, 1.0-1e-8);

    const double regul_term = 0.5 * arma::accu(sj % arma::square(vals_inp - mj));

    const double obj_val = regul_term - arma::accu(Y%arma::log(mu) + (1.0-Y)%arma::log(1.0-mu) );

    arma::vec dreg = sj % (vals_inp - mj);
    if(grad_out) {

        *grad_out = dreg + X.t() * (mu - Y);

    }
    if (hess_out) {
        arma::mat dd = arma::join_rows(dreg, arma::ones(dreg.size()));
        arma::mat ddr = arma::join_rows(arma::ones(dreg.size()),dreg);
        arma::mat hr = dd*ddr.t();
        hr.diag() = sj;
        //hr -= arma::diagmat(arma::diagvec(hr) - sj);
        arma::mat S = arma::diagmat(mu%(1.0-mu));
        *hess_out = X.t() * S * X + hr;
    }

    return obj_val;
}

objective_function::objective_function(
    const arma::mat& x, const arma::vec& y,
    const arma::vec& mj, const arma::vec& sj
):  kk(0), X(x), Y(y), mj(mj), sj(sj), console(spdlog::get("console"))
{
}

objective_function::~objective_function()
{
}

std::pair<int, size_t> objective_function::run(const arma::vec& w0)
{

    w = w0; //arma::randu(w0.n_rows)/w0.n_rows;

    bool success = optim::newton(w, blr_fn_whess, this);
    //console->warn("{0}",std::sqrt(arma::accu(arma::square(w-w1))));
    if (success) {
        console->debug("CG optimization successfully terminated.");
    }
    else {
        console->warn("CG error.");
    }

    return std::pair<int, size_t>(1-static_cast<int>(success), kk);

}


arma::vec objective_function::predict(const arma::vec& wj) {
    return sigm(X * wj);
}

arma::mat objective_function::predict_sample(const size_t n) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<double> d;

    arma::mat ret = arma::mat(X.n_rows, n);
    for(size_t i = 0; i < n; ++i) {
        arma::vec wj = arma::vec(mj.n_rows);
        for(arma::uword j = 0; j < wj.n_rows; ++j) {
            auto params(std::normal_distribution<double>::param_type(mj(j), 1.0/sj(j)));
            wj(j) = d(gen, params);
        }
        arma::vec pred = sigm(X * wj);
        ret.col(i) = pred.t();
    }
    return ret;
}

