#pragma once
#include "data.h"

class SNPFilter {
private:
    LDScoreParams params_;
public:
    SNPFilter() = default;
    explicit SNPFilter(LDScoreParams &params) {
        this->params_ = params;
    }

    inline bool is_used(int idx) const {
        if (0 <= idx and idx < params_.num_of_snp) {
            return params_.positions[idx] >= 0;
        } else {
            return false;
        }
    }

    inline bool filter_maf(double maf) const {
        return maf > params_.maf;
    }

    inline bool filter(int fst, int snd) const {
        if (is_used(fst) and is_used(snd)) {
            auto dist = std::abs(params_.positions[snd] - params_.positions[fst]);
            return dist <= params_.ld_wind;
        } else {
            return false;
        }
    }
};


namespace Regression {
    static inline arma::fvec residuals(const arma::fvec &x, const arma::fvec &y)
    {
        double x_mean = arma::mean(x);
        double y_mean = arma::mean(y);
        auto n = static_cast<double>(x.n_elem);
        double coef = (arma::dot(x, y) / n - x_mean*y_mean)
                 / (arma::dot(x, x) / n - x_mean*x_mean);
        //double intercept = fit_intercept_ ? y_mean - coeff_ * x_mean : 0;

        return y - coef * x;
    }
}




