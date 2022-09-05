import argparse as argp

parser = argp.ArgumentParser()

parser.add_argument(
    "-o",
    "--out",
    default='result',
    required=False,
)

parser.add_argument(
    "-ld",
    action='store_true'
)

parser.add_argument(
    "-h2",
    action='store_true'
)

parser.add_argument(
    "-p",
    "--bfile",
    required=False,
    metavar="PATH"
)

parser.add_argument(
    "-kb",
    "--ld-wind-kb",
    help="Specify the window size to be used for estimating LD Scores in kilo-base pairs (kb).",
    default=None,
    metavar="W",
)

parser.add_argument(
    "-cm",
    "--ld-wind-cm",
    help="Specify the window size to be used for estimating LD Scores in centi-morgans (cM).",
    default=None,
    metavar="W",
)

parser.add_argument(
    "-maf",
    "--maf",
    help="Minor allele frequency threshold",
)

parser.add_argument(
    "-std",
    "--std-thr",
    help="Standard deviation threshold for regression residuals",
)

parser.add_argument(
    "-v",
    "--verbose",
    default=2
)
