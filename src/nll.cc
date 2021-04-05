#include "nll.h"
#include <cmath>
#include <boost/math/special_functions/erf.hpp>
#include <iostream>

double NLL::logistic(double w) {
    return 1.0/(1.0 + std::exp(-w));
}

double NLL::logitnormal_moment(double m, double s, int n, int K) {
    double ret = 0.0;
    for(int i = 1; i < K; ++i) {
        ret += std::pow(logistic(boost::math::erf_inv(i/K)*s + m),n);
    }
    return ret / (K - 1);
}

double NLL::beta_prior_metric(long int Ni, long int Nc, double a, double b) {
    double denom = std::lgamma(a) + std::lgamma(b) /*+ std::lgamma(Nc + 1) + std::lgamma(Ni - Nc + 1)*/ + std::lgamma(Ni + a + b);
    double nom = std::lgamma(a+b) + /*std::lgamma(Ni + 1) +*/ std::lgamma(Nc + a) + std::lgamma(Ni - Nc + b);
    return (denom - nom);
}

std::pair<double, double> NLL::get_beta_params(double p, double s, double prior_aplusb) {
    double aplusb = std::max(prior_aplusb, (p - p*p - s)/s);
    if(s < 1e-15) aplusb = prior_aplusb;
    return std::make_pair(p*aplusb, (1-p)*aplusb);
}

double NLL::logitnorm_to_beta_nll(const example& ex, double prior_aplusb) {
    double p = logistic(ex.m);
    double s = logitnormal_moment(ex.m, ex.s, 2);
    auto params = get_beta_params(p, s, prior_aplusb);
    return beta_prior_metric(ex.Ni, ex.Nc, params.first, params.second);
}

void NLL::push(size_t key, long int Ni, long int Nc, double m, double s) {
    if(nlldata.find(key) == nlldata.end()) {
        example ex = {0, 0, m, s};
        nlldata[key] = ex;
    }
    auto& el = nlldata[key];
    el.Ni += Ni;
    el.Nc += Nc;
    if(el.m != m || el.s != s) {
        std::cerr << "Unequal mean or variance in binomial group. Hashing collision?" << el.m << " " << m << " " << el.s << " " << s <<std::endl;
    }
}

double NLL::eval() {
    double nll = 0.0;
    double baseline_nll = 0.0;
    double den = 0.0;
    double baseline_sumu = 0.0;
    double baseline_sumv = 0.0;
    double baseline_sumw = 0.0;

    for(auto it: nlldata) {
        const example& ex = it.second;
        double k = ex.Nc;
        double n = ex.Ni;
        baseline_sumu += n*(k/n);
        baseline_sumw += n;
    }
    double mean = baseline_sumu/baseline_sumw;
    for(auto it: nlldata) {
        const example& ex = it.second;
        double k = ex.Nc;
        double n = ex.Ni;
        baseline_sumv += n*(mean - k/n)*(mean - k/n);
    }
    double var = baseline_sumv / baseline_sumw;
    baseline_params = get_beta_params(mean, var);
    for(auto it: nlldata) {
        const example& ex = it.second;
        nll += logitnorm_to_beta_nll(ex);
        baseline_nll += beta_prior_metric(ex.Ni, ex.Nc, baseline_params.first, baseline_params.second);
        den += ex.Ni;
    }
    baseline = baseline_nll / den;
    return nll/den;
}
