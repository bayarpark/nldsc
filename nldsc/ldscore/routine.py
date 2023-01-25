from typing import List

import click
import pandas as pd

from .common import *
from core.logger import log

from . import _ldscore as lds


__all__ = ["estimate_lds"]


def show_summary(ld: lds.LDScoreResult):
    pd.set_option("display.precision", 3)
    data = pd.DataFrame(list(zip(ld.l2, ld.l2d, ld.maf)), columns=['L2', 'L2D', 'MAF'])

    corr = data[['L2', 'L2D', 'MAF']].corr()
    click.echo("=" * 62)
    click.echo("L2/L2D/MAF Correlation matrix\n" + str(corr))
    description = data[['L2', 'L2D', 'MAF']].describe().drop('count')

    click.echo(f"\nShort summary:\n"
               f"- Number of additive non-null LD: {data['L2'].count()}\n"
               f"- Number of non-additive non-null LD: {data['L2D'].count()}\n"
               + str(description))

    click.echo("=" * 62)


def make_output(
        bim: BIMFile,
        ld: lds.LDScoreResult,
        *,
        extra=False,
) -> pd.DataFrame:
    columns = ['CHR', 'SNP', 'BP', 'L2', 'L2D']
    data = [bim.chr, bim.snp, bim.bp, pd.Series(ld.l2), pd.Series(ld.l2d)]

    if extra:
        columns += ['MAF', 'WSA', 'WSD', 'WSDE', 'RSTD']
        data += [pd.Series(c) for c in [ld.maf, ld.l2_ws, ld.l2d_ws, ld.l2d_wse, ld.residuals_std]]

    out_df = pd.concat(data, axis=1)
    out_df.columns = columns

    return out_df


@elapsed_time
def estimate_lds(
        bfile: str,
        ld_wind: float,
        wind_metric: str,
        maf_thr: float = 1e-5,
        std_thr: float = 1e-5,
        rsq_thr: float = None,
        *,
        out: str = None,
        extra=False,
        summary=False,
        verbose=0
) -> pd.DataFrame | None:
    bed_, bim_, fam_ = PLINKFile.parse(bfile)
    ld_wind_ = LDWindow(ld_wind, metric=wind_metric)
    maf_thr_ = MAF(maf_thr)
    std_thr_ = ResidualsSTDThreshold(std_thr)

    if rsq_thr is None:
        rsq_thr = 1. / bim_.n_snp

    rsq_thr_ = RSQThreshold(rsq_thr)

    log.info(f"Input: {bed_.data}, size: (M={bim_.n_snp}, N={fam_.n_org})")

    params = lds.LDScoreParams(
        bfile=bed_.data,
        n_snp=bim_.n_snp,
        n_org=fam_.n_org,
        ld_wind=ld_wind_.data,
        maf=maf_thr_.data,
        std_thr=std_thr_.data,
        rsq_thr=rsq_thr_.data,
        positions=getattr(bim_, wind_metric)
    )

    log.info("Running the estimator. It may take a long time.")
    ld = lds.calculate(params)
    log.info("Estimation completed")

    if summary:
        show_summary(ld)

    out_df = make_output(bim_, ld, extra=extra)

    if out:
        log.info("Writing data to disk...")
        out_df.to_csv(out, sep="\t", index=False, float_format='%.5f')
        log.info(f"Completed. File: {out}")
    else:
        return out_df

