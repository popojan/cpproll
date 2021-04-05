#pragma once
#include <unordered_map>
#include <utility>

class NLL {


public:
    NLL() {}
    ~NLL() {}
    void push(size_t key, long int Ni, long int Nc, double m, double s);
    double eval();
    double base() { return baseline; }
    std::pair<double, double> baseline_params;
private:
    struct example {
        long int Ni;
        long int Nc;
        double m;
        double s;
    };

    double baseline;

    std::unordered_map<size_t, example> nlldata;

    static double logistic(double w);

    static double logitnormal_moment(double m, double s, int n, int K = 100);

    static double beta_prior_metric(long int Ni, long int Nc, double a, double b);

    static double logitnorm_to_beta_nll(const example& ex, double prior_aplusb = 2.0);

    static std::pair<double, double> get_beta_params(double u, double v, double prior_aplusb = 2.0);

};


