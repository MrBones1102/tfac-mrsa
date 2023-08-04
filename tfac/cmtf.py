"""
Coupled Matrix Tensor Factorization
"""

import numpy as np
from statsmodels.multivariate.pca import PCA
import tensorly as tl
from tensorly.tenalg import khatri_rao, svd_interface
from tqdm import tqdm
from tensorpack.cmtf import (
    cp_normalize,
    reorient_factors,
    sort_factors,
    mlstsq,
    calcR2X,
)


tl.set_backend("numpy")


def perform_CMTF(tOrig, mOrig, r=8, tol=1e-6, maxiter=1000, progress=True):
    """Perform CMTF decomposition."""
    assert tOrig.dtype == float
    assert mOrig.dtype == float
    factors = [np.ones((tOrig.shape[i], r)) for i in range(tOrig.ndim)]

    # SVD init mode 0
    unfold = np.hstack((tl.unfold(tOrig, 0), mOrig))
    factors[0] = svd_interface(
        np.nan_to_num(unfold),
        mask=np.isnan(unfold),
        method="randomized_svd",
        n_eigenvecs=r,
        random_state=0,
        n_iter_mask_imputation=20,
    )[0]
    tFac = tl.cp_tensor.CPTensor((None, factors))

    # Pre-unfold
    unfolded = np.hstack((tl.unfold(tOrig, 0), mOrig))
    missingM = np.all(np.isfinite(mOrig), axis=1)
    assert np.sum(missingM) >= 1, "mOrig must contain at least one complete row"
    R2X = -np.inf

    # Precalculate the missingness patterns
    uniqueInfo = np.unique(np.isfinite(unfolded.T), axis=1, return_inverse=True)

    tq = tqdm(range(maxiter), disable=(not progress))
    for _ in tq:
        for m in [1, 2]:
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = mlstsq(kr, tl.unfold(tOrig, m).T).T

        # Solve for the mRNA factors
        tFac.mFactor = np.linalg.lstsq(
            tFac.factors[0][missingM, :], mOrig[missingM, :], rcond=None
        )[0].T
        tFac.mFactor = tl.qr(tFac.mFactor)[0]

        # Solve for subjects factors
        kr = khatri_rao(tFac.factors, skip_matrix=0)
        kr = np.vstack((kr, tFac.mFactor))
        tFac.factors[0] = mlstsq(kr, unfolded.T, uniqueInfo).T

        R2X_last = R2X
        R2X = calcR2X(tFac, tOrig, mOrig)
        tq.set_postfix(R2X=R2X, delta=R2X - R2X_last, refresh=False)
        assert R2X > 0.0

        if R2X - R2X_last < tol:
            break

    assert not np.all(tFac.mFactor == 0.0)
    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)
    tFac = sort_factors(tFac)
    tFac.R2X = R2X

    return tFac


def perform_pca(tensor, rna, r=8):
    """
    Returns PCA factors.

    Parameters:
        tensor (tl.CPTensor): cytokine data, in tensor form
        rna (np.array): RNA-seq matrix
        r (int, default:8): PCA components to use

    Returns:
        components (numpy.array): PCA components
        var_explained (float): variance explained by PCA
    """
    unfolded_tensor = tl.unfold(tensor, 0)
    stacked = np.hstack((unfolded_tensor, rna))

    pca = PCA(stacked, ncomp=r, missing='fill-em', method='nipals')
    components = pca.factors / abs(pca.factors).max(axis=0)
    var_explained = pca.rsquare[-1]

    return components, var_explained
