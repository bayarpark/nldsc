import numpy as np
import pandas as pd

from core.common import elapsed_time, NLDSCParameterError
from core.logger import log
from .utils import merge_ld_sumstats, cols, prettify_summary, attempt_save
from .common import GWASSumStatsReader, LDScoreReader
from .regressions import HSQEstimator


def _estimate_h2(
    sumstats: pd.DataFrame,
    ld: pd.DataFrame,
    M: int,
    MD: int,
    n_blocks: int,
    intercept_h2: float,
    chisq_max: float,
    two_step: int,
    strategy: str,
):
    overall = merge_ld_sumstats(sumstats, ld)
    n_overall = len(overall)

    chisq = cols(overall['Z'] ** 2, n_overall)

    if chisq_max is None:
        chisq_max = max(0.001 * overall['N'].max(), 80)


    indices = np.ravel(chisq < chisq_max)
    overall = overall.iloc[indices]
    n_overall, n_old = len(overall), n_overall
    log.info(f'Removed {n_old - n_overall} SNPs with chi^2 > {chisq_max} ({n_overall} SNPs remain)')

    chisq = cols(chisq[indices], n_overall)

    ref_ld_add = cols(overall['L2'], n_overall)
    ref_ld_dom = cols(overall['L2D'], n_overall)
    M = np.array([[M]])
    MD = np.array([[MD]])

    if strategy == 'one-stg':
        raise NotImplementedError("one-staged estimator has not been implemented")
    elif strategy == 'two-stg':
        hsq = HSQEstimator(
            chisq=chisq,
            x_add=ref_ld_add,
            w_add=ref_ld_add,
            x_dom=ref_ld_dom,
            w_dom=ref_ld_dom,
            N=cols(overall['N'], n_overall),
            M_add=M,
            M_dom=MD,
            n_blocks=n_blocks,
            intercept_add=intercept_h2,
            slow=False,
            two_step=two_step
        )
    else:
        raise NLDSCParameterError("Unknown estimation strategy. Only `one-stg` and `two-stg` are allowed")

    return hsq


@elapsed_time
def estimate_h2(
    sumstats: str,
    ldscore: str,
    n_blocks: int = 200,
    intercept_h2: float = None,
    chisq_max: float = None,
    use_m: bool = False,
    two_step: int = None,
    strategy: str = 'two-staged',
    save_to_json: str = None
):
    log.info("Reading GWAS summary statistics...")
    sumstats = GWASSumStatsReader(sumstats, alleles=False, dropna=True)

    log.info("Reading LD Scores...")
    ld = LDScoreReader(ldscore, use_m=use_m)

    if chisq_max is None:
        chisq_max = max(sumstats.data['N'].max() * 1e-3, 80)

    if two_step is None and intercept_h2 is None:
        two_step = 30

    log.info("Estimating heritability...")
    hsq_est = _estimate_h2(
        sumstats=sumstats.data,
        ld=ld.data,
        M=ld.M,
        MD=ld.MD,
        n_blocks=n_blocks,
        intercept_h2=intercept_h2,
        chisq_max=chisq_max,
        two_step=two_step,
        strategy=strategy,
    )

    summary = hsq_est.summary()
    print(prettify_summary(summary))

    if save_to_json:
        attempt_save(save_to_json, summary)

