from .common import *
from .logger import log
from .core import ld_calc, h2_regression


def dispatch(*args, **kwargs) -> None:
    params = ArgParams(*args, **kwargs)

    if params.ld:
        log.info("Mode selected: LD Score calculation")
        ld_calc(params)
    elif params.h2:
        log.info("Mode selected: LD Score Regression")
        h2_regression(params)
    else:
        log.error("Select one of two modes of using the -ld or -h2 flags")


