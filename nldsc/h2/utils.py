import pandas as pd
import numpy as np

from core.logger import log, log_exit


def cols(x: pd.Series | np.ndarray, n: int) -> np.ndarray:
    """
    Makes an array a column vector

    Parameter
    ---------
    x : pd.Series | np.ndarray
        (n, ) or (n, 1) or (1, n) dim data

    Returns
    -------
    np.ndarray with shape (n, 1)
    """
    return np.array(x).reshape((n, 1))


def merge_ld_sumstats(
        sumstats: pd.DataFrame,
        ld: pd.DataFrame
) -> pd.DataFrame:

    out = pd.merge(sumstats, ld, how='inner', on='SNP')
    msg = f"After merging with [reference panel LD/regression SNP LD], {len(out)} SNPs remain"
    if len(sumstats) == 0:
        raise log_exit(msg)
    log.info(msg)

    return out
