from abc import ABC, abstractmethod
from functools import wraps
from typing import Any

import time
from datetime import timedelta

from .logger import log


def elapsed_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = str(timedelta(seconds=end - start))
        log.info(f'Elapsed time: {elapsed}')
        return result
    return wrapper


class NLDSCParameterError(Exception):
    pass


class Data(ABC):
    _data = None

    @property
    def data(self) -> Any:
        return self._data

    def __str__(self):
        return self.__repr__()

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def _validate(self):
        pass
