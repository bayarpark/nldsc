import pandas as pd
import numpy as np


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


def merge_sumstats(
        sumstats: pd.DataFrame,
        ldscore: pd.DataFrame
) -> pd.DataFrame:
    """Check if SNP columns are equal. If so, save time by using concat instead of merge."""
    if (
            len(sumstats) == len(ldscore)
            and (sumstats.index == ldscore.index).all()
            and (sumstats["SNP"] == ldscore["SNP"]).all()
    ):
        sumstats = sumstats.reset_index(drop=True)
        ldscore = ldscore.reset_index(drop=True).drop('SNP', 1)
        out = pd.concat([sumstats, ldscore], axis=1)
    else:
        out = pd.merge(sumstats, ldscore, how='inner', on='SNP')

    msg = f"After merging with [reference panel LD/regression SNP LD], {len(out)} SNPs remain."

    if len(sumstats) == 0:
        raise ValueError(msg)
    else:
        log(msg)

    return out
