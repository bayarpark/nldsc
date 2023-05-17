import click

import routines
from core.logger import log

__version__ = "0.1.4+alpha"

__header__ = (f"\n==============================================================\n"
              f"* Non-additive LD Score Regression (NLDSC)\tv{__version__}\n"
              f"* (C) 2021-2023 Bayar Park\twww.github.com/bayarpark/nldsc\n"
              f"* Based on LDSC \t\twww.github.com/bulik/ldsc\n"
              f"* (C) 2014-2019 Brendan Bulik-Sullivan and Hilary Finucane\n"
              f"* GNU General Public License v3\n"
              f"==============================================================\n")


def handle_exception(func):
    def handler(*args, **kwargs):
        display = kwargs.pop('display', None)
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            log.critical(f"The program crashed with {ex.__class__.__name__}, what: {str(ex)}\n"
                         f"Use `--display` flag for traceback", exc_info=display)
            raise SystemExit()
    return handler


@click.group()
@click.version_option(version=__version__)
def main():
    click.echo(__header__)


@main.command("ld"
              , help="Estimate additive and non-additive LD Scores")
@click.option('--bfile'
              , help="Path prefix for PLINK .bed/.bim/.fam file or path to one of them"
              , metavar="FILE"
              , required=True)
@click.option('-o'
              , '--out'
              , help="Path prefix for output. If not specified it will be <bfile>.L2/<bfile>.M"
              , metavar="FILE")
@click.option('-kb'
              , '--ld-wind-kb'
              , help="Specify the window size to be used for estimating LD Scores in kilo-base pairs (kb)"
              , metavar='W')
@click.option('-cm'
              , '--ld-wind-cm'
              , help="Specify the window size to be used for estimating LD Scores in centi-morgans (cM)"
              , metavar='W')
@click.option('-maf'
              , '--maf-thr'
              , help="Minor allele frequency threshold (lower bound)"
              , metavar="F")
@click.option('-std'
              , '--std-thr'
              , help="Standard deviation threshold for regression residuals"
              , metavar="F"
              , default=1e-4)
@click.option("-rsq"
              , "--rsq-thr"
              , help="R-squared threshold for regression residual. "
                     "It affects only dominant window sizes and, therefore, non-additive sample size (MD)"
              , metavar="F")
@click.option("--extra"
              , help="Include additional information to the .L2 file"
              , metavar="B"
              , is_flag=True
              , default=False)
@click.option("--display"
              , help="Display traceback"
              , is_flag=True
              , default=False)
@handle_exception
def est_ld(bfile, out, ld_wind_kb, ld_wind_cm, maf_thr, std_thr, rsq_thr, extra):
    if sum(map(bool, [ld_wind_kb, ld_wind_cm])) != 1:
        raise RuntimeError("Please, specify exactly one --ld-wind option")
    elif ld_wind_kb:
        wind_metric = 'kbp'
        ld_wind = ld_wind_kb
    else:
        wind_metric = 'cm'
        ld_wind = ld_wind_cm

    routines.estimate_lds(
        bfile,
        ld_wind=ld_wind,
        wind_metric=wind_metric,
        maf_thr=maf_thr,
        std_thr=std_thr,
        rsq_thr=rsq_thr,
        out=out,
        extra=extra,
        summary=True
    )


@main.command("h2"
              , help="Estimate additive and non-additive heritability")
@click.option('--sumstats'
              , help="Path to the GWAS sumstats file"
              , metavar="FILE"
              , required=True)
@click.option('--ref-ld'
              , help="Which file/path with LD Scores to use as the predictors in the LD Score regression"
              , metavar="PTH"
              , required=True)
@click.option('--w-ld'
              , help="Which file/path with LD Scores with sum r^2 taken over SNPs included to use for the "
                     "regression weights. ATTENTION: THIS FLAG ONLY FOR INTERFACE COMPATIBILITY WITH LDSC"
              , metavar="PTH"
              , required=True)
@click.option('--strategy'
              , help="Which method use for heritability estimation"
              , type=click.Choice(["one-stg", "two-stg"])
              , default="two-stg")
@click.option('--chisq-max'
              , help="Maximum value of the chi-square statistic. "
                     "All values greater than `chisq-max` are replaced by `chisq-max`"
              , metavar='F'
              , default=None)
@click.option('--n-blocks'
              , help="Number of jackknife blocks"
              , metavar='N'
              , default=200)
@click.option('--use-M'
              , help="Use .M file instead of .M_5_50"
              , is_flag=True
              , default=False)
@click.option('-s'
              , '--save-to-json'
              , help="Path to file where to write results"
              , metavar='W'
              , default=None)
@click.option("--display"
              , help="Display traceback"
              , is_flag=True
              , default=False)
@handle_exception
def est_h2(sumstats, ref_ld, w_ld, strategy, chisq_max, n_blocks, use_m, save_to_json):
    if ref_ld != w_ld:
        raise NotImplementedError("Method for different annotations is not yet implemented. "
                                  "Please, make sure that `ref_ld` and `w_ld` are equal.")

    routines.estimate_h2(
        sumstats=sumstats,
        ldscore=ref_ld,
        n_blocks=n_blocks,
        intercept_h2=None,
        chisq_max=chisq_max,
        two_step=30,
        strategy=strategy,
        save_to_json=save_to_json
    )


if __name__ == "__main__":
    main()
