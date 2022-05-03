from .common import *
from .logger import log

import pandas as pd

try:
    import ldscore
    log.warn('Using precompiled ldscore module.')
except ImportError:
    log.warn('Using local ldscore module.')
    try:
        from .. import ldscore
    except ImportError as ex:
        log.error('Unable to find ldscore module. Please compile or add it in some other way.')
        raise ex


def _make_params(params: ArgParams) -> ldscore.LDScoreParams:
    ld_params = ldscore.LDScoreParams()

    ld_params.ld_wind = params.ld_wind.data
    ld_params.bedfile = params.bedfile
    ld_params.num_of_snp = params.num_of_snp
    ld_params.num_of_org = params.num_of_org
    ld_params.maf = params.maf
    ld_params.positions = params.positions

    return ld_params


def _join_ld_out(params: ArgParams, ld_a: List[float], ld_d: List[float]) -> pd.DataFrame:
    COLUMNS = (
        'CHR',
        'SNP',
        params.ld_wind.metric.upper(),
        'ADD',
        'DOM'
    )

    return pd.DataFrame(list(zip(
        params.bim.chr.tolist(),
        params.bim.snp.tolist(),
        params.positions,
        ld_a,
        ld_d
    )), columns=COLUMNS)


@elapsed_time
def ld_calc(params: ArgParams):
    log.info(f"Data size: ({params.num_of_snp}, {params.num_of_org})")

    log.info("Preparing the data..")
    ld_params = _make_params(params)

    log.info("Running the calculation algorithm. It may take a long time.")
    ld_a, ld_d = ldscore.calculate(ld_params)

    log.info("Writing data to disk..")
    out_df = _join_ld_out(params, ld_a, ld_d)
    out_df.to_csv(params.out, sep='\t', index=False)
    log.info(f"Done. File: {params.out}")


@elapsed_time
def h2_regression(params: ArgParams):
    raise NotImplementedError()
