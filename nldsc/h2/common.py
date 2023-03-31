from typing import Tuple
from pathlib import Path

from core.common import Data
from core.logger import log

import pandas as pd


def get_compression(filename):
    extensions = [
        '.gz', '.bz2', '.zip', '.xz',
        '.zst', '.tar', '.tar.gz',
        '.tar.xz', '.tar.bz2'
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
        """

        Parameters
        ----------
        file : str
            Path to the .sumstats file
        alleles : bool, default=False
            Remain A1 and A2 columns
        dropna : bool, default=False
            Drop snp with missing values
        """

        self.filename = file
        self.alleles = alleles
        self.dropna = dropna

        columns = ['SNP', 'Z', 'N'] + (['A1', 'A2'] if self.alleles else [])
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
    columns = ['CHR', 'SNP', 'BP', 'L2', 'L2D']

    def __init__(self, path: str):
        self.path = Path(path)
        self._data, self.M, self.MD = self.read()

    def read(self):
        try:
            if self.path.is_dir():
                return self._read_folder()
            else:
                return self._read_file()

        except ValueError as ex:
            raise RuntimeError('Error parsing reference panel LD Score', ex)

    def _read_folder(self):
        scores_path = self.path.glob("*.L2")

        scores = []
        overall_m = 0
        overall_md = 0

        for path in scores_path:
            score, m, md = self._read_file(path)
            overall_m += m
            overall_md += md
            scores.append(score[self.columns])

        overall_scores = pd.concat(scores, axis=0).reset_index(drop=True)
        overall_scores.sort_values(by=['CHR', 'BP'], inplace=True)
        return overall_scores, overall_m, overall_md

    def _read_file(self, path: Path = None):
        if path is None:
            path = self.path

        precalculated = path.with_suffix('.M').exists()
        score = self._read_l2(str(path))
        if precalculated:
            m, md = self._read_m(str(path.with_suffix('.M')))
        else:
            m = len(score['L2'])
            md = m * (score['WSDE'] / score['WSA']).mean()

        return score, m, int(md)

    @staticmethod
    def _read_l2(path: str) -> pd.DataFrame:
        score = pd.read_csv(path, sep='\t')
        score.sort_values(by=['CHR', 'BP'], inplace=True)  # SEs will be wrong unless sorted
        score = score.dropna().drop_duplicates(subset='SNP')
        return score

    @staticmethod
    def _read_m(path: str) -> Tuple[int, int]:
        m = pd.read_csv(path, sep='\t')
        return int(m['M']), int(m['MD'])

    def __repr__(self):
        return f'LDScoreReader(path={self.path})'

    def _validate(self):
        pass
