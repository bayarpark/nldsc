#ifndef LDSCORE_TOOLS_H
#define LDSCORE_TOOLS_H

#include "data.h"

class SNPFilter {
private:
    LDScoreParams params_;
public:
    SNPFilter() = default;
    explicit SNPFilter(LDScoreParams &params) {
        this->params_ = params;
    }

    inline bool
    is_used(int idx) const
    {
        if (0 <= idx and idx < params_.num_of_snp) {
            return params_.positions[idx] >= 0;
        } else {
            return false;
        }
    }

    inline bool
    filter_maf(double maf) const
    {
        return maf > params_.maf;
    }

    inline bool
    filter(int fst, int snd) const
    {
        if (is_used(fst) and is_used(snd)) {
            auto dist = std::abs(params_.positions[snd] - params_.positions[fst]);
            return dist <= params_.ld_wind;
        } else {
            return false;
        }
    }
};


namespace Math {
    inline arma::fvec
    regression_residuals(const arma::fvec &x, const arma::fvec &y)
    {
        /**
         * Fitting 1d-linear regression and calculates regression residuals
         * @param x: Regressor
         * @param y: Regressand
         *
         * @return Regression residuals
         */
        double x_mean = arma::mean(x);
        double y_mean = arma::mean(y);
        auto n = static_cast<double>(x.n_elem);
        double slope = (arma::dot(x, y) / n - x_mean*y_mean)
                       / (arma::dot(x, x) / n - x_mean*x_mean);
        //double intercept = fit_intercept_ ? y_mean - coeff_ * x_mean : 0;

        return y - slope * x;
    }


    inline double
    r2_adjusted(const arma::fvec& fst, const arma::fvec& snd, int n) {
        double corr = arma::as_scalar(arma::cor(fst, snd));
        double r2 = corr*corr;
        return (1. - (1. - r2) * (n - 1) / (n - 2));
    }

//    inline arma::fvec
//    r2_adjusted(const arma::fvec& y, const arma::fmat& X, int n) {
//        arma::fmat corr = arma::cor(y, X);
//        double r2 = corr*corr;
//        return (1. - (1. - r2) * (n - 1) / (n - 2));
//    }
}



#endif //LDSCORE_TOOLS_H
