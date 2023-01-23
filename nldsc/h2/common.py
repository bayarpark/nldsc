from pathlib import Path
from typing import List, Set
import glob

from ..core.common import Data
from ..core.logger import log, log_exit

import pandas as pd


def get_compression(filename):
    extensions = [
        '.gz', '.bz2', '.zip', '.xz',
        '.zst', '.tar', '.tar.gz',
        '.tar.xz',  '.tar.bz2'
    ]

    suffixes = Path(filename).suffixes[-2:]
    long = ''.join(suffixes)
    short = suffixes[-1]

    if long in extensions:
        return long
    elif short in extensions:
        return short
    else:
        return None


class GWASSumStatsReader(Data):
    coltypes = {'SNP': str, 'Z': float, 'N': float, 'A1': str, 'A2': str}

    def __init__(self, file: str, alleles=False, dropna=False):

        self.filename = file
        self.alleles = alleles
        self.dropna = dropna

        columns = ['SNP', 'Z', 'N'] + (['A1', 'A2'] if alleles else [])
        self._data = pd.read_csv(
            file,
            delim_whitespace=True,
            na_values='.',
            usecols=columns,
            dtype=self.coltypes,
            compression=get_compression(file),
        )

        if dropna:
            self._data.dropna(how='any', inplace=True)

        n_snp = len(self._data)
        self._data.drop_duplicates(subset='SNP', inplace=True)

        if n_snp > (n_snp_new := len(self._data)):
            log.info(f'Dropped {n_snp - n_snp_new} SNPs with duplicated rs numbers.')

    @property
    def n_snp(self):
        return len(self._data)

    def __repr__(self):
        return f'GWASSumStatsReader(file={self.filename}, alleles={self.alleles}, dropna={self.dropna})'

    def _validate(self):
        pass


class LDScoreReader(Data):
    columns = ['CHR', 'SNP', 'ADD', 'DOM']
    whole_genome = set(range(1, 23))

    def __init__(self, path: str, columns=None):
        if columns is not None:
            self.columns = columns

        self.path = path
        self._data = self.read()

    def read(self):
        try:
            path = Path(self.path)
            if path.is_dir():
                files = glob.glob("*.ldscore")
                return self._read_ldscores_from_files_by_chr(files, self.whole_genome)
            else:
                return self._read_ldscores_from_files_by_chr(
                    self._str_to_filepaths(self.path), self.whole_genome
                )

        except ValueError as ex:
            log_exit('Error parsing reference panel LD Score.', ex)

    @staticmethod
    def _str_to_filepaths(files_args: str):
        files = files_args.replace(' ', '').split(',')
        files = [str(Path(file).absolute()) for file in files]
        return files

    def read_ldscore(self, filename: str):
        """Parse .ldscore files, split across chromosomes"""
        ld = pd.read_csv(filename, sep='\t')
        ld.sort_values(by=['CHR', 'BP'], inplace=True)  # SEs will be wrong unless sorted
        ld = ld[self.columns].drop_duplicates(subset='SNP')
        return ld

    def _read_ldscores_from_files_by_chr(self, files: List[str], selected_chrs: Set[int]) -> pd.DataFrame:
        ldscores = pd.concat([self.read_ldscore(file) for file in files], axis=0)
        ldscores = ldscores[ldscores['CHR'].apply(lambda x: x in selected_chrs)]
        ldscores = ldscores.drop('CHR', axis=1).drop_duplicates(subset='SNP')
        return ldscores

    def __repr__(self):
        return f'LDScoreReader(path={self.path})'

    def _validate(self):
        pass

