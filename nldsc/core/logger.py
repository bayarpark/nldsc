import logging


class ColorLogFormatter(logging.Formatter):
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD_WHITE = BOLD + WHITE
    BOLD_BLUE = BOLD + BLUE
    BOLD_GREEN = BOLD + GREEN
    BOLD_YELLOW = BOLD + YELLOW
    BOLD_RED = BOLD + RED
    END = "\033[0m"

    FORMAT = "%(prefix)s%(msg)s%(suffix)s"

    LOG_LEVEL_COLOR = {
        "DEBUG": {'prefix': '', 'suffix': ''},
        "INFO": {'prefix': BOLD_GREEN + " ❱ " + GREEN, 'suffix': END},
        "WARNING": {'prefix': BOLD_YELLOW + " ⚠ " +BOLD_YELLOW, 'suffix': END},
        "ERROR": {'prefix': BOLD_RED + " ⚠ " + BOLD_RED, 'suffix': END},
        "CRITICAL": {'prefix': BOLD_RED + " ❌ " + BOLD_RED, 'suffix': END},
    }

    def format(self, record):
        if not hasattr(record, 'prefix'):
            record.prefix = self.LOG_LEVEL_COLOR.get(record.levelname.upper()).get('prefix')

        if not hasattr(record, 'suffix'):
            record.suffix = self.LOG_LEVEL_COLOR.get(record.levelname.upper()).get('suffix')

        formatter = logging.Formatter(self.FORMAT)
        return formatter.format(record)


log = logging.getLogger('nldsc')

log.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("nldsc.log")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s  %(name)s  %(levelname)s: %(message)s")
file_handler.setFormatter(file_formatter)
log.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(ColorLogFormatter())
log.addHandler(stream_handler)


def log_exit(msg, exception=None):
    log.critical(msg)
    if exception is not None:
        raise exception
    else:
        raise RuntimeError("See description above")
