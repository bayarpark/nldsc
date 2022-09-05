from .common import *
from .logger import log

import pandas as pd

try:
    from ldscore import *
    log.warn('Using precompiled ldscore module.')
except ImportError:
    log.warn('Using local ldscore module.')
    try:
        from ..ldscore import *
    except ImportError as ex:
        log.error('Unable to find ldscore module. Please compile or add it in some other way.')
        raise ex

pd.set_option("display.precision", 4)


def _lds_make_params(params: ArgParams) -> LDScoreParams:
    ld_params = LDScoreParams()

    ld_params.ld_wind = params.ld_wind.data
    ld_params.bedfile = params.bedfile
    ld_params.num_of_snp = params.num_of_snp
    ld_params.num_of_org = params.num_of_org
    ld_params.maf = params.maf
    ld_params.std_threshold = params.std_threshold
    ld_params.positions = params.positions

    return ld_params


def _join_ld_out(params: ArgParams,
                 result: LDScoreResult,
                 extra: bool=True
) -> pd.DataFrame:
    COLUMNS = (
        'CHR',
        'SNP',
        params.ld_wind.metric.upper(),
        'ADD',
        'DOM',
    )

    EXT_COLUMNS = (
        'MAF',
        'WSA',
        'WSD',
        'RSTD',
    )

    if extra:
        data = pd.DataFrame(list(zip(
            params.bim.chr.tolist(),
            params.bim.snp.tolist(),
            params.positions,
            result.l2_add,
            result.l2_nadd,
            result.mafs,
            result.additive_winsizes,
            result.non_additive_winsizes,
            result.residuals_std
        )), columns=COLUMNS + EXT_COLUMNS)
    else:
        data = pd.DataFrame(list(zip(
            params.bim.chr.tolist(),
            params.bim.snp.tolist(),
            params.positions,
            result.l2_add,
            result.l2_nadd
        )), columns=COLUMNS)

    return data


def show_statistics(params: ArgParams, out_df: pd.DataFrame):
    if params.verbose >= 2:
        corr = out_df[['ADD', 'DOM', 'MAF']].corr()
        log.info("ADD/DOM/MAF Correlation matrix\n" + str(corr))
        description = out_df[['ADD', 'DOM', 'MAF']].describe().drop('count')

        log.info(f"Short summary\n" 
                 f"Number of additive non-null LD: {out_df['ADD'].count()}\n"
                 f"Number of non-additive non-null LD: {out_df['DOM'].count()}\n"

                 + str(description))


@elapsed_time
def ld_calc(params: ArgParams):
    log.info(f"Input: {params.bedfile}, size: ({params.num_of_snp}, {params.num_of_org})")

    log.info("Preparing the data..")
    ld_params = _lds_make_params(params)

    log.info("Running the calculation algorithm. It may take a long time.")
    result = calculate(ld_params)

    out_df = _join_ld_out(params, result)

    show_statistics(params, out_df)

    log.info("Writing data to disk..")
    out_df.to_csv(params.out, sep='\t', index=False)
    log.info(f"Done. File: {params.out}")


@elapsed_time
def h2_regression(params: ArgParams):
    raise NotImplementedError()
