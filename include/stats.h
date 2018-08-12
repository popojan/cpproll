#ifndef _H_STATS
#define _H_STATS

#include "eigen.h"

class RunningStat {

public:
    RunningStat(const size_t N) : m_n(Eigen::VectorXi::Zero(N)) {
        m_newM.resize(N);
        m_newS.resize(N);
    }

    void clear(size_t idx) {
        m_n(idx) = 0;
    }

    void push(size_t idx, double x) {
        m_n(idx)++;

        // See Knuth TAOCP vol 2, 3rd edition, page 232
        if (m_n(idx) == 1) {
            m_newM(idx) = x;
            m_newS(idx) = 0.0;
        }
        else {
            double m_oldM = m_newM(idx);
            double m_oldS = m_newS(idx);

            m_newM(idx) = m_oldM + (x - m_oldM)/m_n(idx);
            m_newS(idx) = m_oldS + (x - m_oldM)*(x - m_newM(idx));
        }
    }

    int count(size_t idx) const {
        return m_n(idx);
    }

    double mean(size_t idx) const {
        return (m_n(idx) > 0) ? m_newM(idx) : 0.0;
    }

    double var(size_t idx) const {
        return ( (m_n(idx) > 1) ? m_newS(idx)/(m_n(idx) - 1) : 0.0 );
    }

    double dev(size_t idx) const {
        return std::sqrt(var(idx));
    }

    void save(const std::string& fmodel) const {
        serialize(m_n, fmodel + ".n.bin");
        serialize(m_newM, fmodel + ".m.bin");
        serialize(m_newS, fmodel + ".s.bin");
    }
    void load(const std::string& fmodel) {
        deSerialize(m_n, fmodel + ".n.bin");
        deSerialize(m_newM, fmodel + ".m.bin");
        deSerialize(m_newS, fmodel + ".s.bin");
    }
private:
    Eigen::VectorXi m_n;
    Eigen::VectorXd m_newM, m_newS;
};

#endif
