#ifndef LDSCORE_LDSCALC_H
#define LDSCORE_LDSCALC_H

#include "data.h"
#include "stream.h"



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
                bool is_succeed = cache_.init_next_chunk();
                if (is_succeed) {
                    eval_for_chunk(
                            cache_.get_add(),
                            idx);
                }
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


    void eval_for_chunk(const arma::fvec& y, int idx) {
        double add = 1;
        double nadd = 0;

        while (auto x = cache_.get()) {
            add += Math::r2_adjusted(y, x.add(), params_.num_of_org);
            nadd += Math::r2_adjusted(y, x.residuals(), params_.num_of_org);
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



#endif //LDSCORE_LDSCALC_H
