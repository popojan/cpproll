#include "stage2.h"
#include "stage3.h"
#include <fstream>
#include <numeric>
#include <iostream>

template <class T>
Stage3<T>::Stage3(
    Buffer<T>& buffer,
    const Options& opts
)
 :
    buffer_(buffer),
    rs(opts.N),
    console(spdlog::get("console")),
    opts(opts)
{
    duration_sum = 0;
    duration_den = 0;
    //mj = (0.5-arma::randu(opts.N))/200.0;
    mj = arma::zeros(opts.N);//(0.5-arma::randu(opts.N))/200.0;
    mj(0) = opts.intercept;
    sj = arma::ones(opts.N, 1)/opts.lambda;
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

  std::vector<size_t> idx(v.n_cols);
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
    std::ofstream ffout;
    std::streambuf * buf = 0;
    if(!opts.fpred.empty()) {
        if(opts.fpred != "-") {
            ffout.open(opts.fpred);
            buf = ffout.rdbuf();
        }
        else {
            buf = std::cout.rdbuf();
        }
        std::setprecision(8);
    }
    std::ostream fout(buf);

    std::ofstream svmout;
    if(!opts.svmlight.empty()) {
        svmout.open(opts.svmlight);
    }
    while (true) {
        T batch = buffer_.remove();
        if(batch.nzf.size() <= 0)
            break;
        lineno += batch.x.n_rows;
        const size_t K = batch.x.n_cols, k = batch.x.n_rows;

        arma::vec w = arma::zeros(K,1);
        arma::vec mjt = arma::zeros(K,1);
        arma::vec sjt = arma::ones(K, 1)/opts.lambda;
        if(!opts.svmlight.empty()) {
            for(size_t j = 0; j < k; ++j) {
                svmout << batch.labels(j, 0);
                for(auto nit = batch.nzf.begin(); nit != batch.nzf.end(); ++nit) {
                    svmout << " " << nit->first << ":" << batch.x(j, nit->second);
                }
                svmout << std::endl;
            }
        }

        for(auto bit = batch.nzf.begin(); bit != batch.nzf.end(); ++bit) {
            size_t idx = bit->second;
            size_t ii = bit->first;
            w(idx) = mj(ii);// + 1e-6;
            mjt(idx) = mj(ii);
            sjt(idx) = sj(ii);
            if(opts.standardize) {
                for(arma::uword j = 0; j < batch.x.n_rows; ++j) {
                    double val = batch.x(j, idx);
                    if(val != 0) {
                        if(!opts.testonly) {
                            rs.push(ii, val);
                        }
                        double mean = rs.mean(ii);
                        double dev = rs.dev(ii);
                        if(dev > 0)
                            batch.x.col(idx) = (batch.x.col(idx) - mean)/dev + 1.0;
                    }
                }
            }
        }
        objective_function f(batch.x, batch.labels, mjt, sjt);

        arma::vec pred = f.predict(mjt);
        arma::mat mpred;
        if(opts.npredict > 0) {
            mpred = f.predict_sample(opts.npredict);
        }
        if(!opts.fpred.empty()) {
            auto dit = batch.desc.begin();
            arma::mat zpred;
            if(opts.explain) {
                zpred.resize(batch.x.n_rows, batch.x.n_cols);
                for(arma::uword j = 0; j < batch.x.n_cols; ++j) {
                    arma::vec wz(mjt);
                    wz(j) = 0.0;
                    zpred.col(j) = f.predict(wz);
                 }
            }
            for(arma::uword i = 0; i < batch.labels.size(); ++i, ++dit) {
                fout << batch.labels(i) << "\t" << pred(i);
                if(opts.npredict > 0) {
                    for(arma::uword j = 0; j < mpred.n_cols; ++j) {
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

        for(arma::uword i = 0; i < batch.labels.size(); ++i) {
            roc.push(static_cast<int>(batch.labels(i)), static_cast<float>(pred(i)));
            //iroc.push(static_cast<int>(batch.labels(i)), static_cast<float>(pred(i)));
        }

        console->debug("Prediction before {0}", pred(0));
        if(lineno - lineno0 > 0) {
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<TimeT>(now - start);
            if(duration.count() > opts.period) {
                double fps = 1000*(lineno-lineno0)/static_cast<double>(duration.count());
                start = now;
                lineno0 = lineno;
                //iroc.run();
                //float auc = iroc.getAreaUnderCurve();
                double hashload = static_cast<double>(arma::accu(mj != 0.0))/opts.N;
                unsigned nonzero_cnt = static_cast<double>(arma::accu(mjt != 0.0));
                console->info("{0:>10.6f} {1:>10.6f} {2:>8.3f} load  {3:>10.4f} {4:>4} {5:>7} feat {9:>7} nz {6:>10} ex {7:>8.0f} ex/s {8:>4} ev/ex",
                    logloss_sum/logloss_den, logloss_asum/logloss_aden, hashload, pred(0), batch.labels(0), batch.x.n_cols,
                    lineno, fps, std::round(iterations_sum/duration_den),
                    nonzero_cnt);
                logloss_asum = 0.0;
                logloss_aden = 0.0;
                duration_den = 0;
                duration_sum = 0;
                iterations_sum = 0.0;
            }
        }

        const double eps = 1e-15;
        pred = arma::clamp(pred, eps, 1.0-eps);
        double ll = arma::accu(batch.labels % arma::log(pred) + (1.0-batch.labels)%arma::log(1.0 - pred));
        logloss_aden += k;
        logloss_den += k;
        logloss_sum -= ll;
        logloss_asum -= ll;
        label_sum += arma::accu(batch.labels);

        if(!opts.testonly) {
            auto start = std::chrono::steady_clock::now();

            std::pair<int, size_t>  ret = f.run(w);
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<TimeT>(now - start);
            duration_sum += duration.count();
            duration_den += k;
            iterations_sum += ret.second;

            if(ret.first == 0) {
                w = f.w;
                arma::vec ui = f.predict(w);
                arma::vec uio = ui % (1.0 - ui);
                for(auto it = batch.nzf.begin(); it != batch.nzf.end(); ++it) {
                    if(it->first > 0) {
                        mj(it->first) = w(it->second);
                        sj(it->first) = sjt(it->second) + arma::accu(arma::square(batch.x.col(it->second)%uio));
                    }
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
    mj.save(fmodel + ".mj.bin");
    sj.save(fmodel + ".sj.bin");
    rs.save(fmodel);
}

template <class T>
void Stage3<T>::load(const std::string& fmodel) {
    mj.load(fmodel + ".mj.bin");
    sj.load(fmodel + ".sj.bin");
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
    //iroc.run();
    console->info("Global auROC {0:8.6f}.", roc.getAreaUnderCurve());
    //console->info("Increm auROC {0:8.6f}.", iroc.getAreaUnderCurve());
}
template class Stage3<Batch>;
