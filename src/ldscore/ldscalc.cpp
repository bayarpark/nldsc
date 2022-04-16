#include "data.h"
#include "stream.cpp"


inline double r2_adjusted(const arma::fvec& fst, const arma::fvec& snd, int n) {
    double corr = arma::as_scalar(arma::cor(fst, snd));
    double r2 = corr*corr;
    return (1. - (1. - r2) * (n - 1) / (n - 2));
}


class LDSCalculator {
public:
    explicit LDSCalculator(LDScoreParams& params) {
        this->params_ = params;
        this->cache_ = ChunkCache(params);
        this->filter_ = SNPFilter(params);
        this->lds_add_ = std::vector<double>(params.num_of_snp, std::nan(""));
        this->lds_nadd_ = std::vector<double>(params.num_of_snp, std::nan(""));
    }

    void calculate() {
        for (int idx = 0; idx < params_.num_of_snp; ++idx) {
            if (filter_.is_used(idx)) {
                auto chunk = cache_.next_chunk();
                eval_for_chunk(chunk.first, chunk.second, idx);
            } else {
                cache_.pass_chunk();
            }
        }
    }

    std::vector<double> get_add() {
        return lds_add_;
    }

    std::vector<double> get_nadd() {
        return lds_nadd_;
    }

private:
    LDScoreParams params_;
    SNPFilter filter_;
    ChunkCache cache_;
    std::vector<double> lds_add_;
    std::vector<double> lds_nadd_;


    void eval_for_chunk(arma::fvec* y, const std::vector<PairSNP>& X, int idx) {
        if (y == nullptr) {
            return;
        }

        double add = 1;
        double nadd = 0;

        for (auto x : X) {
            add += r2_adjusted(*y, *x.add_, params_.num_of_org);
            nadd += r2_adjusted(*y, Regression::residuals(*x.add_, *x.nadd_), params_.num_of_org);
        }
        lds_add_[idx] = add;
        lds_nadd_[idx] = nadd;
    }
};


std::pair<std::vector<double>, std::vector<double>> calculate(LDScoreParams& params) {
    auto calculator = LDSCalculator(params);
    calculator.calculate();
    return {calculator.get_add(), calculator.get_nadd()};
}




