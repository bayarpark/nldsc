"""
(c) 2014 Brendan Bulik-Sullivan and Hilary Finucane

Rewritten version of https://github.com/bulik/ldsc/blob/master/ldscore/regression.py
(c) 2022 Bayar Park

Estimators of heritability and genetic correlation.

Shape convention is (n_snp, n_annot) for all classes.
Last column = intercept.

"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple
from abc import abstractmethod, ABC

import numpy as np
from scipy import stats as ss

from core.logger import log

from . import jackknife as jk
from . import irwls
from .utils import cols


def h2_obs_to_liability(h2_obs, P, K):
    """
    Converts heritability on the observed scale in an ascertained sample to heritability
    on the liability scale in the population.

    Parameters
    ----------
    h2_obs : float
        Heritability on the observed scale in an ascertained sample.
    P : float in (0, 1)
        Prevalence of the phenotype in the sample.
    K : float in (0, 1)
        Prevalence of the phenotype in the population.

    Returns
    -------
    h2_liab : float
        Heritability of liability in the population.
    """
    if np.isnan(P) and np.isnan(K):
        return h2_obs
    if not 0 < K < 1:
        raise ValueError('K must be in the range (0, 1)')
    if not 0 < P < 1:
        raise ValueError('P must be in the range (0,1 )')

    thresh = ss.norm.isf(K)
    conversion_factor = (K ** 2 * (1 - K) ** 2 / (P * (1 - P) * ss.norm.pdf(thresh) ** 2))
    return h2_obs * conversion_factor


def update_stdparators(separators, idx_mask: np.ndarray):
    """separators with ii masked. Returns unmasked separators."""
    maplist, _ = np.where(idx_mask)
    t = np.apply_along_axis(
        lambda i: maplist[i], 0, separators[1:-1]
    )
    t = np.hstack([0, t, len(idx_mask)])
    return t


def append_intercept(*xs):
    """
    Appends an intercept term to the design matrix for a linear regression.

    Parameters
    ----------
    xs : a number of the np.matrix with shape (n_row, n_col)
        Design matrix. Columns are predictors; rows are observations.

    Returns
    -------
    x_new : np.matrix with shape (n_row, n_col + 1)
        Design matrix with intercept term appended.

    """
    for x in xs:
        n_row = x.shape[0]
        intercept = np.ones((n_row, 1))
        x_new = np.concatenate((x, intercept), axis=1)
        yield x_new


def remove_intercept(x: np.ndarray) -> np.ndarray:
    """Removes the last column."""
    n_col = x.shape[1]
    return x[:, 0:n_col - 1].copy()


def _eval_z_score(est, std):
    """Convert estimate and std to Z-score"""
    try:
        z_score = est / std
    except (FloatingPointError, ZeroDivisionError):
        z_score = float('inf')
    return z_score


def _eval_p_value(z_score):
    """Estimates p-value from z_score"""
    p_value = ss.chi2.sf(z_score ** 2, 1, loc=0, scale=1)  # 0 if Z=inf
    return p_value


@dataclass
class Coefficient:
    value: np.ndarray | float
    cov: np.ndarray | float = None
    std: np.ndarray | float = None
    name: str = None

    def __str__(self) -> str:
        string = "Estimated: "
        if self.name:
            string += f"{self.name} "
        string += f"{self.value} "
        if self.std:
            string += f"± {self.std} std."

        return string


class LDScoreRegression(ABC):
    null_intercept = None

    def __init__(
            self,
            y, x, w, N, M,
            n_blocks,
            intercept=None,
            slow=False,
            step1_idx_mask=None,
            old_weights=False,
            additive_weights=None
    ):

        self._validate(y, x, w, M, N)
        n_snp, self.n_annot = x.shape

        self._check_input_shapes(y, w, N, M, n_snp, self.n_annot)

        M_tot = np.sum(M)
        x_tot = cols(np.sum(x, axis=1), n_snp)  # vector with shape (n_snp, 1)
        self.constrain_intercept = (intercept is not None)
        self.intercept = intercept
        self.n_blocks = n_blocks

        self.M = M
        tot_agg = self.aggregate(y, x_tot, N, M_tot, intercept)

        if additive_weights is None:
            initial_w = self._update_weights(
                x_tot, w, N, M_tot, tot_agg, intercept)
        else:
            initial_w = np.square(additive_weights)

        N_mean = float(np.mean(N))  # keep condition number low
        x = np.multiply(N, x) / N_mean

        if not self.constrain_intercept:
            x, x_tot = append_intercept(x, x_tot)
            yp = y.copy()
        else:
            yp = y - intercept
            self.intercept_std = np.nan  # replaced "NaN"

        if step1_idx_mask is not None and self.constrain_intercept:
            raise ValueError("two-step is not compatible with constrain_intercept.")
        elif step1_idx_mask is not None and self.n_annot > 1:
            raise ValueError("two-step not compatible with partitioned LD Score yet.")
        elif step1_idx_mask is not None:  # two-step routine
            n1 = int(np.sum(step1_idx_mask))  # number of object that less than two-step constant (30)
            x1 = x[step1_idx_mask.flatten(), :]
            yp1, w1, N1, initial_w1 = (a[step1_idx_mask].reshape((n1, 1)) for a in [yp, w, N, initial_w])

            irwls_ufunc_1 = lambda a: self._update_func(  # aka update_func_1
                a, x1, w1, N1, M_tot, N_mean, ii=step1_idx_mask
            )
            step1_jknife = irwls.irwls(
                x1, yp1, irwls_ufunc_1, n_blocks, slow=slow, w=initial_w1
            )

            step1_intercept = self._extract_intercept(step1_jknife)
            yp -= step1_intercept.value
            x = remove_intercept(x)
            x_tot = remove_intercept(x_tot)

            irwls_ufunc_2 = lambda a: self._update_func(
                a, x_tot, w, N, M_tot, N_mean, step1_intercept.value)

            separators = update_stdparators(step1_jknife.separators, step1_idx_mask)

            step2_jknife = irwls.irwls(
                x, yp, irwls_ufunc_2, n_blocks, slow=slow, w=initial_w, separators=separators
            )

            c = (np.sum(np.multiply(initial_w, x)) / np.sum(np.multiply(initial_w, np.square(x))))

            jknife = self._combine_twostep_jknives(
                step1_jknife, step2_jknife, c
            )

        elif old_weights:
            initial_w = np.sqrt(initial_w)
            x = irwls.reweigh(x, initial_w)
            y = irwls.reweigh(yp, initial_w)
            jknife = jk.LSTSQJackknifeFast(x, y, n_blocks)
        else:
            irwls_ufunc = lambda a: self._update_func(
                a, x_tot, w, N, M_tot, N_mean, intercept
            )
            jknife = irwls.irwls(
                x, yp, irwls_ufunc, n_blocks, slow=slow, w=initial_w, is_dominant=(additive_weights is not None)
            )

        self.jknife = jknife

        self.coefficients = self._extract_coefs(jknife, N_mean)
        self.category = self._eval_catwise(M, self.coefficients.value, self.coefficients.cov)
        self.total = self._eval_total(self.category.value, self.category.cov)

        self.proportion = self._eval_proportion(jknife, M, N_mean, self.category.value, self.total.value)

        self.enrichment, self.M_prop = self._eval_enrichment(
            M, M_tot, self.category.value, self.total.value)

        if not self.constrain_intercept:
            self.intercept = self._extract_intercept(jknife)

        self.tot_delete_values = self._delete_vals_tot(jknife, N_mean, M)
        self.part_delete_values = self._delete_vals_part(jknife, N_mean, M)

        if not self.constrain_intercept:
            self.intercept_delete_values = jknife.delete_values[:, self.n_annot]

    @abstractmethod
    def _update_func(self, *args, **kwargs):
        pass

    @abstractmethod
    def _update_weights(self, *args, **kwargs):
        pass

    @classmethod
    def aggregate(cls, y, x, N, M, intercept=None):
        if intercept is None:
            intercept = cls.null_intercept

        num = M * (np.mean(y) - intercept)
        denom = np.mean(np.multiply(x, N))
        return num / denom

    def _delete_vals_tot(self, jknife: jk.JackknifeEstimation, N_mean, M):
        """Get delete values for total h2 or gencov."""
        n_annot = self.n_annot
        tot_delete_vals = jknife.delete_values[:, 0:n_annot]  # shape (n_blocks, n_annot)
        tot_delete_vals = np.dot(tot_delete_vals, M.T) / N_mean  # shape (n_blocks, 1)
        return tot_delete_vals

    def _delete_vals_part(self, jknife: jk.JackknifeEstimation, N_mean, M):
        """Get delete values for partitioned h2 or gencov."""
        n_annot = self.n_annot
        return jknife.delete_values[:, 0:n_annot] / N_mean

    def _extract_coefs(self, jknife: jk.JackknifeEstimation, N_mean: float) -> Coefficient:
        """Get coefficient estimates + cov from the jackknife."""
        n_annot = self.n_annot
        coef = jknife.est[0, 0:n_annot] / N_mean
        coef_cov = jknife.jk_cov[0:n_annot, 0:n_annot] / N_mean ** 2
        coef_std = np.sqrt(np.diag(coef_cov))
        return Coefficient(coef, coef_cov, coef_std)

    @staticmethod
    def _eval_catwise(M, coef, coef_cov) -> Coefficient:
        """Convert coefficients to per-category h2 or gencov."""
        cat = np.multiply(M, coef)
        cat_cov = np.multiply(np.dot(M.T, M), coef_cov)
        cat_std = np.sqrt(np.diag(cat_cov))
        return Coefficient(cat, cat_cov, cat_std)

    @staticmethod
    def _eval_total(cat, cat_cov) -> Coefficient:
        """Convert per-category h2 to total h2 or gencov."""
        tot = float(np.sum(cat))
        tot_cov = float(np.sum(cat_cov))
        tot_std = float(np.sqrt(tot_cov))
        return Coefficient(tot, tot_cov, tot_std)

    def _eval_proportion(self, jknife: jk.JackknifeEstimation, M, N_mean, cat, tot) -> Coefficient:
        """Convert total h2 and per-category h2 to per-category proportion h2 or gencov."""
        n_annot = self.n_annot
        n_blocks = jknife.delete_values.shape[0]
        numer_delete_vals = np.multiply(
            M, jknife.delete_values[:, 0:n_annot]) / N_mean  # (n_blocks, n_annot)
        denom_delete_vals = np.sum(
            numer_delete_vals, axis=1).reshape((n_blocks, 1))
        denom_delete_vals = np.dot(denom_delete_vals, np.ones((1, n_annot)))
        prop = jk.RatioJackknife(
            cat / tot, numer_delete_vals, denom_delete_vals)
        return Coefficient(prop.est, prop.jk_cov, prop.jk_std)

    def _eval_enrichment(self, M, M_tot, cat, tot):
        """Compute proportion of SNPs per-category enrichment for h2 or gencov."""
        M_prop = M / M_tot
        enrichment = np.divide(cat, M) / (tot / M_tot)
        return enrichment, M_prop

    def _extract_intercept(self, jknife: jk.JackknifeEstimation) -> Coefficient:
        """Extract intercept and intercept standard error from block jackknife."""
        n_annot = self.n_annot
        intercept = jknife.est[0, n_annot]
        intercept_std = jknife.jk_std[0, n_annot]
        return Coefficient(intercept, std=intercept_std)

    def _combine_twostep_jknives(self, step1_jknife: jk.JackknifeEstimation, step2_jknife: jk.JackknifeEstimation, c):
        """Combine free intercept and constrained intercept jackknives for --two-step."""
        n_blocks, n_annot = step1_jknife.delete_values.shape
        n_annot -= 1
        if n_annot > 2:
            raise NotImplementedError("twostep not yet implemented for partitioned LD Score")

        step1_int = self._extract_intercept(step1_jknife)
        est = np.hstack([step2_jknife.est, np.array(step1_int.value).reshape((1, 1))])
        delete_values = np.zeros((n_blocks, n_annot + 1))
        delete_values[:, n_annot] = step1_jknife.delete_values[:, n_annot]
        delete_values[:, 0:n_annot] = (
                step2_jknife.delete_values -
                c * (
                        step1_jknife.delete_values[:, n_annot] - step1_int.value
                ).reshape((n_blocks, n_annot))
        )
        pseudovalues = jk.Jackknife.delete_values_to_pseudovalues(delete_values, est)
        jk_est, jk_var, jk_std, jk_cov = jk.Jackknife.jknife(pseudovalues)
        jknife = namedtuple(
            'jk',
            ['est', 'jk_std', 'jk_est', 'jk_var', 'jk_cov', 'delete_values']
        )
        return jknife(est, jk_std, jk_est, jk_var, jk_cov, delete_values)

    @staticmethod
    def _validate(*arrays):
        for array in arrays:
            if isinstance(array, np.ndarray):
                if len(array.shape) != 2:
                    raise ValueError(f"Array shape is {array.shape}. Only 2D arrays is allowed.")
            else:
                raise TypeError(f"Array has type {type(array)} but numpy.ndarray is expected")

    @staticmethod
    def _check_input_shapes(y, w, N, M, n_snp: int, n_annot: int):
        if any(i.shape != (n_snp, 1) for i in [y, w, N]):
            raise ValueError('N, weights and response (z1z2 or chisq) must have shape (n_snp, 1).')
        if M.shape != (1, n_annot):
            raise ValueError(f'M must have shape (1, n_annot), but {M.shape} were given.')


class HSQAdditive(LDScoreRegression):
    """
    Additive heritability
    """
    null_intercept = 1.

    def __init__(
            self,
            y, x, w, N, M,
            n_blocks=200,
            intercept=None,
            slow=False,
            two_step=None,
            old_weights=False,
            additive_weights=None
    ):
        """

        Parameters
        ----------
        y : np.ndarray (n_snp, 1)
            target vector
        x : np.ndarray (n_snp, n_annot)
            regressors
        w : np.ndarray (n_snp, 1)
            weights
        N : np.ndarray (n_snp, 1)
            GWAS sample size
        M : np.ndarray (1, n_annot)
            SNP sample size
        n_blocks : int
            number of blocks
        intercept : float | None
            regression intercept
        slow : bool, default=False
            use slow jackknife or not
        two_step : float | None
            cutoff for two-step estimator
        old_weights : bool
            whether use old weights
        """

        step1_idx_mask = two_step if two_step is None else (y < two_step)
        super().__init__(
            y, x, w, N, M, n_blocks,
            intercept=intercept, slow=slow,
            step1_idx_mask=step1_idx_mask, old_weights=old_weights,
            additive_weights=additive_weights
        )

        self.mean_chisq, self.lambda_gc = self._summarize_chisq(y)
        if not self.constrain_intercept:
            self.ratio = self._ratio(
                self.intercept.value, self.intercept.std, self.mean_chisq
            )

    def _update_func(
        self,
        x: Tuple[np.ndarray, ...],
        ref_ld_tot,
        w_ld,
        N,
        M,
        N_mean,
        intercept=None,
        ii=None
    ):
        """
        Update function for IRWLS

        x is the output of np.linalg.lstsq
        x[0] is the regression coefficients
        x[0].shape is (# of dimensions, 1)
        value x[0, -1] is the intercept.

        intercept is None --> free intercept
        intercept is not None --> constrained intercept
        """
        hsq = M * x[0][0] / N_mean
        if intercept is None:
            intercept = max(x[0][1])  # divide by zero error if intercept < 0
        else:
            if ref_ld_tot.shape[1] > 1:
                raise ValueError('Design matrix has intercept column for constrained intercept regression!')

        ld = ref_ld_tot[:, 0].reshape(w_ld.shape)  # remove intercept
        w = self.weights(ld, w_ld, N, M, hsq, intercept)
        self._weights_checkpoint = w.copy()
        return w

    def _update_weights(self, ld, w_ld, N, M, hsq, intercept, ii=None):
        if intercept is None:
            intercept = self.null_intercept
        w = self.weights(ld, w_ld, N, M, hsq, intercept)
        self._weights_checkpoint = w.copy()

        return self._weights_checkpoint

    @staticmethod
    def weights(ld, w_ld, N, M, hsq, intercept=None):
        """
        Regression weights

        Parameters
        ----------
        ld : np.matrix with shape (n_snp, 1)
            LD Scores (non-partitioned).
        w_ld : np.matrix with shape (n_snp, 1)
            LD Scores (non-partitioned) computed with sum r^2 taken over only those SNPs included
            in the regression.
        N :  np.ndarray of ints > 0 with shape (n_snp, 1)
            Number of individuals sampled for each SNP.
        M : float > 0
            Number of SNPs used for estimating LD Score (need not equal number of SNPs included in
            the regression).
        hsq : float in [0,1]
            Heritability estimate
        intercept : float
            Intercept

        Returns
        -------
        w : np.matrix with shape (n_snp, 1)
            Regression weights. Approx equal to reciprocal of conditional variance function.
        """
        M = float(M)
        if intercept is None:
            intercept = 1.

        hsq = np.clip(hsq, .0, 1.)
        ld = np.fmax(ld, 1.)
        w_ld = np.fmax(w_ld, 1.)
        c = hsq * N / M
        het_w = 1. / (2. * np.square(intercept + np.multiply(c, ld)))
        oc_w = 1. / w_ld
        w = np.multiply(het_w, oc_w)
        return w

    @staticmethod
    def _summarize_chisq(chisq):
        """Compute mean chi^2 and lambda_GC."""
        mean_chisq = np.mean(chisq)
        lambda_gc = np.median(chisq.flatten()) / 0.4549
        return mean_chisq, lambda_gc

    @staticmethod
    def _ratio(intercept, intercept_std, mean_chisq):
        """Compute ratio (intercept - 1) / (mean chi^2 - 1)"""
        if mean_chisq > 1.:
            ratio_std = intercept_std / (mean_chisq - 1.)
            ratio = (intercept - 1.) / (mean_chisq - 1.)
        else:
            ratio, ratio_std = np.nan, np.nan

        return Coefficient(ratio, std=ratio_std)


class HSQDominant(HSQAdditive):
    null_intercept = .0

    def __init__(
        self,
        chisq: np.ndarray,
        x: np.ndarray,
        w: np.ndarray,
        w_add: np.ndarray,
        N: np.ndarray,
        M: np.ndarray,
        n_blocks: int,
        slow: bool,
        hsq_add: HSQAdditive
    ):
        beta, *_ = np.mean(N).flatten()*hsq_add.coefficients.value.flatten()
        intercept, *_ = hsq_add.intercept.value.flatten()

        weights = hsq_add._weights_checkpoint

        residuals = irwls.reweigh(chisq - w_add * beta - np.ones_like(w_add) * intercept, weights)

        super().__init__(
            residuals, x=x, w=w, N=N, M=M,
            n_blocks=n_blocks,
            intercept=None,
            slow=slow,
            two_step=None,
            # old_weights=True,  # warn!
            # additive_weights=weights
        )


class HSQEstimator:
    def __init__(
        self,
        chisq: np.ndarray,
        x_add,
        w_add,
        x_dom,
        w_dom,
        N,
        M_add,
        M_dom,
        n_blocks=200,
        intercept_add=None,
        slow=False,
        two_step=None
    ):
        log.info("Estimating additive heritability...")
        self.additive = HSQAdditive(chisq, x_add, w_add, N, M_add, n_blocks, intercept_add, slow, two_step)
        log.info("Estimating non-additive heritability...")
        self.dominant = HSQDominant(chisq, x_dom, w_dom, w_add, N, M_dom, n_blocks, slow, hsq_add=self.additive)

    def summary(self):
        summary_dict = {
            "additive": {
                "hsq": self.additive.total.value,
                "hsq.std": self.additive.total.std,
                "lambda_gc": self.additive.lambda_gc,
                "chisq.mean": self.additive.mean_chisq,

                "intercept": self.additive.intercept.value,
                "intercept.std": self.additive.intercept.std,
                "intercept.constrained": self.additive.constrain_intercept
            },
            "dominant": {
                "hsq": self.dominant.total.value,
                "hsq.std": self.dominant.total.std,
                "residuals.mean": self.dominant.mean_chisq,

                "intercept": self.dominant.intercept.value,
                "intercept.std": self.additive.intercept.std,
            }
        }

        return summary_dict
