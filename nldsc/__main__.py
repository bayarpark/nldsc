import click

from core import routines
from core.logger import log_exit

__version__ = "0.1.3+alpha"

__header__ = (f"==============================================================\n"
              f"* Non-additive LD Score Regression (NLDSC)\tv{__version__}\n"
              f"* (C) 2022-2023 Bayar Park\twww.github.com/bayarpark/nldsc\n"
              f"* Based on LDSC \t\twww.github.com/bulik/ldsc\n"
              f"* (C) 2014-2019 Brendan Bulik-Sullivan and Hilary Finucane\n"
              f"* GNU General Public License v3\n"
              f"==============================================================\n"
)


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
def est_ld(bfile, out, ld_wind_kb, ld_wind_cm, maf_thr, std_thr, rsq_thr, extra):
    if sum(map(bool, [ld_wind_kb, ld_wind_cm])) != 1:
        log_exit("Please, specify exactly one --ld-wind option")
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
@click.option('--strategy'
              , help="Which method use for heritability estimation"
              , type=click.Choice(["one-stg", "two-stg"])
              , default="two-stg"
              )
@click.option('--sumstats'
              , help="Path to the GWAS sumtats file"
              , metavar="FILE"
              , required=True)
@click.option('--ref-ld'
              , help="Which file with LD Scores to use as the predictors in the LD Score regression"
              , metavar="FILE")
@click.option('--ref-ld-chr'
              , help="The same as --ref-ld, but files must be split across chromosomes"
              , metavar="FOLDER")
@click.option('--w-ld'
              , help="Which file with LD Scores with sum r^2 taken over SNPs included to use for the "
                     "regression weights. ATTENTION: THIS FLAG ONLY FOR INTERFACE COMPATIBILITY WITH LDSC"
              , metavar="FILE")
@click.option('--w-ld-chr'
              , help="The same as --w-ld, but files must be split across chromosomes"
              , metavar="FOLDER")
@click.option('--chisq-max'
              , help="Maximum value of the chi-square statistic. "
                     "All values greater than `chisq-max` are replaced by `chisq-max`"
              , metavar='F'
              , default=80)
@click.option('--n-blocks'
              , help="Number of jackknife blocks"
              , metavar='N'
              , default=200)
def est_h2(sumstats, ref_ld_chr, w_ld_chr):
    pass


if __name__ == "__main__":
    main()
