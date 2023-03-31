from .logger import log

from h2 import estimate_h2

try:
    from ldscore import estimate_lds
except (ImportError, AttributeError) as ex:
    msg = "Can't load `ldscore` sub-package. Please build it correctly."
    log.error(msg)

    def estimate_lds(*args, **kwargs):
        raise RuntimeError(msg)
