#ifndef LDSCORE_LDSCALC_H
#define LDSCORE_LDSCALC_H

#include "data.h"
#include "stream.h"


class LDSCalculator {
private:
    LDScoreParams params_;
    SNPFilter filter_;
    ChunkCache cache_;
    std::vector<double> lds_add_;
    std::vector<double> lds_nadd_;
    std::vector<int> add_winsizes_;
    std::vector<int> nadd_winsizes_;


public:
    explicit
    LDSCalculator(LDScoreParams& params) {
        this->params_ = params;
        this->cache_ = ChunkCache(params);
        this->filter_ = SNPFilter(params);
        this->lds_add_ = std::vector<double>(params.num_of_snp, std::nan(""));
        this->lds_nadd_ = std::vector<double>(params.num_of_snp, std::nan(""));
        this->add_winsizes_ = std::vector<int>(params.num_of_snp, -1);
        this->nadd_winsizes_ = std::vector<int>(params.num_of_snp, -1);
    }

    void
    calculate() {
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

    std::vector<double>
    get_add() {
        return lds_add_;
    }

    std::vector<double>
    get_nadd() {
        return lds_nadd_;
    }

    std::vector<double>
    get_mafs() {
        std::vector<double> mafs(params_.num_of_snp);
        for (int i = 0; i < params_.num_of_snp; ++i) {
            mafs[i] = cache_[i].maf();
        }
        return mafs;
    }

    std::vector<double>
    get_residuals_std() {
        std::vector<double> stds(params_.num_of_snp);
        for (int i = 0; i < params_.num_of_snp; ++i) {
            stds[i] = cache_[i].residuals_std();
        }
        return stds;
    }

    std::vector<int>
    get_additive_win_sizes() {
        return add_winsizes_;
    }

    std::vector<int>
    get_non_additive_win_sizes() {
        return nadd_winsizes_;
    }

private:

    void
    eval_for_chunk(const arma::fvec& y, int idx) {
        double add = 1;
        double nadd = 0;

        auto indices = cache_.get_chunk();
        int passed = 0;

        #pragma omp parallel for schedule(static) \
            shared(cache_, y, indices) reduction(+:nadd, add, passed) default(none)
        for (auto index : indices) {
            const SNPInMemory& snp = cache_[index];
            add += Math::r2_adjusted(y, snp.add());
            if (filter_.filter_residuals_std(snp.residuals_std())) {
                nadd += Math::r2_adjusted(y, snp.residuals());
            } else {
                passed += 1;
            }
        }

        add_winsizes_[idx] = indices.size();
        nadd_winsizes_[idx] = indices.size() - passed;
        lds_add_[idx] = add;
        lds_nadd_[idx] = nadd;
    }
};


LDScoreResult
calculate(LDScoreParams& params) {
    auto calculator = LDSCalculator(params);
    calculator.calculate();
    return {
        calculator.get_add(),
        calculator.get_nadd(),
        calculator.get_mafs(),
        calculator.get_residuals_std(),
        calculator.get_additive_win_sizes(),
        calculator.get_non_additive_win_sizes()
    };
}



#endif //LDSCORE_LDSCALC_H
