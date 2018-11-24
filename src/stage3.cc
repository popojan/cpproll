#include "stage2.h"
#include "stage3.h"
#include <fstream>
#include <numeric>

using Eigen::VectorXd;
using Eigen::VectorXf;
using Eigen::VectorXi;
using Eigen::MatrixXd;
using Eigen::MatrixXf;

template <class T>
Stage3<T>::Stage3(
    Buffer<T>& buffer,
    const Options& opts
)
 :
    buffer_(buffer),
    mj(),
    sj(),
    rs(opts.N),
    console(spdlog::get("console")),
    opts(opts)
{
    duration_sum = 0;
    duration_den = 0;
    mj.resize(opts.N);
    mj = VectorXd::Zero(opts.N, 1);
    sj.resize(opts.N);
    sj = VectorXd::Zero(opts.N, 1);
    sj = ((sj.array() + 1.0)/opts.lambda).matrix();
    logloss_sum = 0.0;
    logloss_asum = 0.0;
    logloss_aden = 0.0;
    logloss_den = 0.0;
    label_sum = 0.0;
    duration_sum = 0.0;
    duration_den = 0.0;
    iterations_sum = 0.0;

}

template <typename T>
std::vector<size_t> sort_indexes(const T &v) {

  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v(i1) < v(i2);});

  return idx;
}


template <class T>
void Stage3<T>::run() {
    size_t lineno = 0ul;
    auto start = std::chrono::steady_clock::now();
    size_t lineno0 = 0ul;
    std::ofstream fout;
    if(!opts.fpred.empty()) {
        fout.open(opts.fpred);
        std::setprecision(8);
    }
    std::ofstream svmout;
    if(!opts.svmlight.empty()) {
        svmout.open(opts.svmlight);
    }
    while (true) {
        T batch = buffer_.remove();
        if(batch.nzf.size() <= 0)
            break;
        lineno += batch.x.rows();
        const size_t K = batch.x.cols(), k = batch.x.rows();

        VectorXd w, mjt, sjt;
        w.resize(K);
        mjt.resize(K);
        sjt.resize(K);
        if(!opts.svmlight.empty()) {
            for(size_t j = 0; j < k; ++j) {
                svmout << batch.labels(j);
                for(auto nit = batch.nzf.begin(); nit != batch.nzf.end(); ++nit) {
                    svmout << " " << nit->first << ":" << batch.x(j, nit->second);
                }
                svmout << std::endl;
            }
        }

        for(auto bit = batch.nzf.begin(); bit != batch.nzf.end(); ++bit) {
            size_t idx = bit->second;
            size_t ii = bit->first;
            double wii = mj(ii);
            w(idx) = wii;
            mjt(idx) = wii;
            sjt(idx) = sj(ii);
            if(opts.standardize) {
                for(int j = 0; j < batch.x.rows(); ++j) {
                    auto val = batch.x(j, idx);
                    if(val != 0) {
                        if(!opts.testonly) {
                            rs.push(ii, val);
                        }
                        auto mean = rs.mean(ii);
                        auto dev = rs.dev(ii);
                        if(dev > 0)
                            batch.x(j, idx) = (val - mean)/dev + 1.0;
                    }
                }
            }
        }

        objective_function f(batch.x, batch.labels, mjt, sjt);

        VectorXd pred = f.predict(mjt);
        MatrixXd mpred;
        if(opts.npredict > 0) {
            mpred = f.predict_sample(opts.npredict);
        }
        if(!opts.fpred.empty()) {
            auto dit = batch.desc.begin();
            MatrixXd zpred;
            if(opts.explain) {
                zpred.resize(batch.x.rows(), batch.x.cols());
                for(long int j = 0; j < batch.x.cols(); ++j) {
                    VectorXd wz(mjt);
                    wz(j) = 0.0;
                    zpred.col(j) = f.predict(wz);
                 }
            }
            for(int i = 0; i < batch.labels.size(); ++i, ++dit) {
                fout << batch.labels(i) << "\t" << pred(i);
                if(opts.npredict > 0) {
                    for(long int j = 0; j < mpred.cols(); ++j) {
                        fout << "\t" << mpred(i, j);
                    }
                }
                if(!opts.desc.empty()) {
                    fout << "\t" << (*dit);
                }
                fout << std::endl;
                if(opts.explain) {
                    for(auto j: sort_indexes(zpred.row(i))) {
                        if(batch.x(i, j) != 0)
                            fout << "\t" << std::fixed << (pred(i)-zpred(i, j)) << "\t" << batch.names[j] << std::endl;
                    }
                    fout << std::endl;
                }
            }
        }

        for(int i = 0; i < batch.labels.size(); ++i) {
            roc.push(static_cast<int>(batch.labels(i)), static_cast<float>(pred(i)));
        }

        console->debug("Prediction before {0}", pred(0));
        if(lineno - lineno0 > 0) {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<TimeT>(now - start);
            if(duration.count() > opts.period) {
                double fps = 1000*(lineno-lineno0)/static_cast<double>(duration.count());
                start = now;
                lineno0 = lineno;
                console->info("{0:>10.6f} {1:>10.6f} {2:>10.4f} {3:>4} {4:>7} feat {5:>10} ex {6:>8.0f} ex/s {7:>4} it/ex",
                    logloss_sum/logloss_den, logloss_asum/logloss_aden, pred(0), batch.labels(0), batch.x.cols(),
                    lineno, fps, std::round(iterations_sum/duration_den));
                logloss_asum = 0.0;
                logloss_aden = 0.0;
                duration_den = 0;
                duration_sum = 0;
                iterations_sum = 0.0;
            }
        }

        const double eps = 1e-15;

        double ll = (batch.labels.array() == 0.0).select(1.0-pred.array(), pred.array()).min(1.0 - eps).max(eps).log().sum();

        logloss_aden += k;
        logloss_den += k;
        logloss_sum -= ll;
        logloss_asum -= ll;
        label_sum += batch.labels.sum();

        if(!opts.testonly) {
            auto start = std::chrono::steady_clock::now();

            std::pair<int, size_t>  ret = f.run(w);

            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<TimeT>(now - start);
            duration_sum += duration.count();
            duration_den += k;
            iterations_sum += ret.second;

            if(ret.first == 0) {
                auto ui = f.predict(w);
                auto uio = (ui.array()*(1.0 - ui.array()));
                VectorXd s(sjt + batch.x.array().square().matrix().transpose()*uio.matrix());
                for(auto it = batch.nzf.begin(); it != batch.nzf.end(); ++it) {
                    mj(it->first) = w(it->second);
                    sj(it->first) = s(it->second);
                }
            }
        }
    }
    if(!opts.svmlight.empty()) {
        svmout.close();
    }
}

template <class T>
void Stage3<T>::save(const std::string& fmodel) {
    serialize(mj, fmodel + ".mj.bin");
    serialize(sj, fmodel + ".sj.bin");
    rs.save(fmodel);
}

template <class T>
void Stage3<T>::load(const std::string& fmodel) {
    deSerialize(mj, fmodel + ".mj.bin");
    deSerialize(sj, fmodel + ".sj.bin");
    rs.load(fmodel);
}
template <class T>
void Stage3<T>::eval() {
    double avg = label_sum/logloss_den;
    double baseline = -((label_sum)*std::log(avg)
             + (logloss_den - label_sum)*std::log(1.0 - avg))
            / logloss_den;
    double best = logloss_sum/logloss_den;
    console->info("Average loss {0:8.6f}, improvement {1:+4.2f} % over {2:8.6f}, best constant [{3:6.4f}] baseline.", best, 100*(1.0 - best/baseline), baseline, avg);
    roc.run();
    console->info("Global auROC {0:8.6f}.", roc.getAreaUnderCurve());
}
template class Stage3<Batch>;
