from __future__ import annotations 

from typing import Dict
from pathlib import Path
import json

import pandas as pd
import numpy as np

from core.logger import log


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
        raise RuntimeError(msg)
    log.info(msg)

    return out


def prettify_summary(summary: Dict):
    text = "\n========================= h2 summary =========================\n"
    text += f"Additive h2: {summary['additive']['hsq']:.4f} ± std: {summary['additive']['hsq.std']:.4f}\n"
    text += f"lambda GC: {summary['additive']['lambda_gc']:.4f}, chi2 mean: {summary['additive']['chisq.mean']:.4f}\n"
    text += f"Dominant h2: {summary['dominant']['hsq']:.4e} ± std: {summary['dominant']['hsq.std']:.4e}\n"
    text += f"residuals mean: {summary['dominant']['residuals.mean']:.4e}\n"
    return text


def attempt_save(filename: str, summary: Dict):
    path = Path(filename)
    if path.is_file():
        raise FileExistsError("File already exists")

    with open(filename, 'w') as f:
        json.dump(summary, f)
