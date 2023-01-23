"""
(c) 2015 Brendan Bulik-Sullivan and Hilary Finucane

Iteratively re-weighted least squares.
"""
import numpy as np

import jackknife as jk


def reweigh(x, w):
    """
    Weight x by w.

    Parameters
    ----------
    x : np.ndarray with shape (n, p)
        Rows are observations.
    w : np.ndarray with shape (n, 1)
        Regression weights (1 / sqrt(CVF) scale).

    Returns
    -------
    x_new : np.ndarray with shape (n, p)
        x_new[i,j] = x[i,j] * w'[i], where w' is w normalized to have sum 1.

    Raises
    ------
    ValueError :
        If any element of w is <= 0 (negative weights are not meaningful in WLS).
    """
    if np.any(w <= 0):
        raise ValueError('Weights must be > 0')
    n, p = x.shape
    if w.shape != (n, 1):
        raise ValueError(f'w has shape {w.shape}. w must have shape (n, 1).')

    w = w / float(np.sum(w))
    x_new = np.multiply(x, w)
    return x_new


def wls(x, y, w):
    """
    Weighted least squares.

    Parameters
    ----------
    x : np.matrix with shape (n, p)
        Independent variable.
    y : np.matrix with shape (n, 1)
        Dependent variable.
    w : np.matrix with shape (n, 1)
        Regression weights (1/CVF scale).

    Returns
    -------
    coef : list with four elements (coefficients, residuals, rank, singular values)
        Output of np.linalg.lstsq

    """
    n, p = x.shape
    if y.shape != (n, 1):
        raise ValueError(f'y has shape {y.shape}. y must have shape ({n}, 1).')
    if w.shape != (n, 1):
        raise ValueError(f'w has shape {w.shape}. w must have shape ({n}, 1).')

    x = reweigh(x, w)
    y = reweigh(y, w)
    coef = np.linalg.lstsq(x, y, rcond=-1)
    return coef


def irwls(x, y, update_func, n_blocks, w, slow=False, separators=None):
    """
    Iteratively re-weighted least squares (IRWLS).

    Parameters
    ----------
    x : np.ndarray with shape (n, p)
        Independent variable.
    y : np.ndarray with shape (n, 1)
        Dependent variable.
    update_func : function
        Transforms output of np.linalg.lstsq to new weights.
    n_blocks : int
        Number of jackknife blocks (for estimating SE via block jackknife).
    w : np.matrix with shape (n, 1)
        Initial regression weights.
    slow : bool
        Use slow block jackknife? (Mostly for testing)
    separators : np.ndarray or None
        Block jackknife block boundaries (optional).

    Returns
    -------
    jknife : jk.LSTSQJackknifeFast
        Block jackknife regression with the final IRWLS weights.

    """

    n, p = x.shape
    w = np.ones_like(y) if w is None else w

    if y.shape != (n, 1):
        raise ValueError(f'y has shape {y.shape}. y must have shape ({n}, 1).')
    if w.shape != (n, 1):
        raise ValueError(f'w has shape {y.shape}. w must have shape ({n}, 1).')

    w = np.sqrt(w)
    for i in range(2):  # update this later
        new_w = np.sqrt(update_func(wls(x, y, w)))
        if new_w.shape != w.shape:
            print('IRWLS update:', new_w.shape, w.shape)
            raise ValueError('New weights must have same shape.')
        else:
            w = new_w

    x = reweigh(x, w)
    y = reweigh(y, w)
    if slow:
        jknife = jk.LSTSQJackknifeSlow(
            x, y, n_blocks, separators=separators)
    else:
        jknife = jk.LSTSQJackknifeFast(
            x, y, n_blocks, separators=separators)

    return jknife
