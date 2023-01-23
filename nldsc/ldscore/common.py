from pathlib import Path
from abc import ABC

import os
import pandas as pd

from core.common import *


class LDWindow(Data):
    def __init__(self, ld_wind: float, metric: str = 'bp'):
        self._data = float(ld_wind)
        self._metric = metric

        if self._metric == 'kbp':
            self._data *= 1000
            self._metric = 'bp'

        self._validate()

    @property
    def metric(self) -> str:
        return self._metric

    def __repr__(self) -> str:
        return f"LDWindow(ld_wind={self._data}, metric='{self._metric}')"

    def _validate(self):
        if self._metric not in ['bp', 'cm']:
            raise NLDSCParameterError('Invalid metric')
        if self._data <= 0:
            raise NLDSCParameterError(f'The ld-window must be greater than 0')
        elif self._metric == 'bp' and self._data > 5 * 10 ** 6:
            raise NLDSCParameterError('The ld-window cannot be larger than 5 Mbp')
        elif self._metric == 'cm' and self._data > 100:
            raise NLDSCParameterError('The ld-window cannot be larger than 100 cm')


class PLINKFile(Data, ABC):
    def __init__(self, path: str):
        self._path = str(path)
        self._validate_path()

    def _validate_path(self):
        if not os.path.exists(self._path):
            raise FileNotFoundError(f'No such file: "{self._path}"')

    @staticmethod
    def parse(bfile: str):
        path = Path(bfile).resolve()
        exts = ['*.bed', '*.bim', '*.fam']
        if any(path.match(ext) for ext in exts):
            path = path.with_suffix('')
        elif path.is_dir():
            raise NotImplementedError("")

        bed_path = path.as_posix() + '.bed'
        bim_path = path.as_posix() + '.bim'
        fam_path = path.as_posix() + '.fam'

        return BEDFile(bed_path), BIMFile(bim_path), FAMFile(fam_path)


class BEDFile(PLINKFile):
    def __init__(self, path: str):
        super().__init__(path)
        self._data = path

    def __repr__(self):
        return f"BEDFile(path='{self._data}')"

    def _validate(self):
        pass


class BIMFile(PLINKFile):
    COLUMNS = (
        'CHR',  # Chromosome code (either an integer, or 'X'/'Y'/'XY'/'MT'; '0' indicates unknown) or name
        'SNP',  # Variant identifier
        'CM',   # Position in morgans or centimorgans
        'BP',   # Base-pair coordinate (1-based; limited to 2^31-2)
        'A1',   # Allele 1 (corresponding to clear bits in .bed; usually minor)
        'A2'    # Allele 2 (corresponding to set bits in .bed; usually major)
    )

    def __init__(self, path: str, **kwargs):
        super().__init__(path)
        self._data = pd.read_csv(path, sep="\t", names=self.COLUMNS)
        self._validate()

    def __repr__(self):
        return f"BIMFile(n_snp={self.n_snp})"

    @property
    def chr(self) -> pd.Series:
        return self._data['CHR']

    @property
    def snp(self) -> pd.Series:
        return self._data['SNP']

    @property
    def cm(self) -> pd.Series:
        return self._data['CM']

    @property
    def bp(self) -> pd.Series:
        return self._data['BP']

    @property
    def n_snp(self) -> int:
        return len(self._data)

    def _validate(self):
        if len(self._data['CHR'].unique()) != 1:
            raise NLDSCParameterError('The current version of the program '
                                      'can only work with one chromosome in one file.')


class FAMFile(PLINKFile):
    COLUMNS = (
        'FID',     # Family ID ('FID')
        'IID',     # Within-family ID ('IID'; cannot be '0')
        'FATHER',  # Within-family ID of father ('0' if father isn't in dataset)
        'MOTHER',  # Within-family ID of mother ('0' if mother isn't in dataset)
        'SEX',     # Sex code ('1' = male, '2' = female, '0' = unknown)
        'TRAIT'    # Phenotype value ('1' = control, '2' = case, '-9'/'0'/non-numeric = missing data if case/control)
    )

    def __init__(self, path: str):
        super().__init__(path)
        self._data = pd.read_csv(path, sep='\t', names=self.COLUMNS)
        self._validate()

    def __repr__(self):
        return f"FAMFile(n_org={self.n_org})"

    @property
    def n_org(self) -> int:
        return len(self._data)

    def _validate(self):
        pass


class MAF(Data):
    def __init__(self, maf: float):
        self._data = float(maf)
        self._validate()

    def __repr__(self) -> str:
        return f"MAF(maf={self._data})"

    def _validate(self):
        if not (0 <= self._data < 1):
            raise NLDSCParameterError('Minor allele frequency must be between 0 and 1!')


class ResidualsSTDThreshold(Data):
    def __init__(self, std_thr: float):
        self._data = float(std_thr)
        self._validate()

    def __repr__(self) -> str:
        return f"ResidualsSTDThreshold(std_thr={self._data})"

    def _validate(self):
        if not (0 <= self._data < 1):
            raise NLDSCParameterError('standard deviation threshold must be between 0 and 1!')


class RSQThreshold(Data):
    def __init__(self, rsq_thr: float):
        self._data = float(rsq_thr)
        self._validate()

    def __repr__(self) -> str:
        return f"RSQThreshold(std_thr={self._data})"

    def _validate(self):
        if not (0 <= self._data < 0.1):
            raise NLDSCParameterError('r-squared threshold must be between 0 and 0.1!')
