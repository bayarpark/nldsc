#pragma once

#include "data.h"
#include "stream.h"


LDScoreResult calculate(LDScoreParams& params) {
    ChunkwiseReader cache(params);
    SNPFilter filter(params);

    std::vector<double> l2(params.n_snp, std::nan(""));
    std::vector<double> l2d(params.n_snp, std::nan(""));

    std::vector<int> l2_ws(params.n_snp, -1);
    std::vector<int> l2d_ws(params.n_snp, -1);
    std::vector<int> l2d_wse(params.n_snp, -1);

    for (int idx = 0; idx < params.n_snp; ++idx) {
        if (filter.is_used(idx)) {
            if (cache.initialize_next_chunk()) { // evaluation for next SNP
                const arma::fvec& y = cache.y();
                double add = 1.0;
                double dom = 0.0;

                const auto& indices = cache.chunk_indices();
                int passed = 0;
                int effective = 0;

                #pragma omp parallel for schedule(static) \
                    shared(cache, y, indices, filter) reduction(+:dom, add, passed, effective) default(none)
                for (auto index : indices) {
                    const SNPInMemory& snp = cache[index];
                    add += Math::r2_adjusted(y, snp.add());

                    if (filter.residuals_std(snp.residuals_std())) {
                        double rsq = Math::r2_adjusted(y, snp.residuals());
                        dom += rsq;
                        effective += filter.r2_dom(rsq);
                    } else {
                        passed += 1;
                    }
                }

                l2[idx] = add;
                l2d[idx] = dom;

                l2_ws[idx] = static_cast<int>(indices.size());
                l2d_ws[idx] = static_cast<int>(indices.size()) - passed;
                l2d_wse[idx] = effective;
            }
        } else {
            cache.pass_chunk();
        }
    }
    return {l2, l2d, cache.mafs(), cache.residual_stds(), l2_ws, l2d_ws, l2d_wse};
}
