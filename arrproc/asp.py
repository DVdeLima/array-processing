# %% Docstring

"""
Array Signal Processing.

------------------------

Includes:
    1. Basic operations
        Column normalization (colnorm)
        Least-Squares Khatri-Rao factorization (lskrf)
        Root mean square error (rmse)
    2. Preprocessing
        Low-rank approximation (lra)
        Selection matrix (selM)
        Forward-backward averaging (fba)
        Spatial smoothing (sps)
        De-sps (desps)
        Multiple denoising (MuDe)
    3. Array processing
        Uniform linear array (ula)
        1-D ESPRIT (esprit1D)
        Sort spatial frequencies (sort_sf)
    4. Model order estimation
        Akaike Information Criterion (aic)
        Exponential Fitting Test (eft)
"""

# %% __all__

__all__ = [
    "colnorm",
    "lskrf",
    "noisy",
    "rmse",
    "lra",
    "selM",
    "fba",
    "sps",
    "desps",
    "MuDe",
    "ula",
    "esprit1D",
    "sort_sf",
    "aic",
    "eft",
]

# %% Load dependencies

import itertools as it

import numpy as np
import scipy.linalg as la
import typing as tp

from scipy.sparse.linalg import svds
from scipy.stats import gmean

# %% Debug modules
# import pdb as ipdb
# ipdb.set_trace()


# %% Basic operations


def colnorm(X: np.ndarray) -> np.ndarray:
    """
    Column Normalization.

    Parameters
    ----------
    X : NumPy array
        Input matrix.

    Returns
    -------
    NumPy array
        Input matrix with normalized columns.

    """
    if X.ndim == 1:
        return X / la.norm(X)
    return X @ np.diag(1 / la.norm(X, axis=0))


def lskrf(K: np.ndarray, M: tp.Union[int, list, np.ndarray]) -> list:
    """
    Least squares Khatri-Rao factorization.

    Parameters
    ----------
    K : NumPy array
        Khatri-Rao product.
    M : int, list, or NumPy array
        Size of first dimension or list
        of sizes (of first dimension)

    Returns
    -------
    list (of NumPy arrays)
        Factor matrices.
    """
    I, R = K.shape
    if isinstance(M, int):
        M = [M, I // M]
    baT = [np.reshape(K.T[r], M).T for r in range(R)]
    u, s, vH = zip(*[svds(baTr, 1, solver="lobpcg") for baTr in baT])
    N = np.diagflat(np.sqrt(s))
    A = [np.squeeze(vH).conj().T @ N, np.squeeze(u).T @ N]
    return A


def noisy(M: np.ndarray, SNR: float = 0.0) -> np.ndarray:
    """
    Add noise to a matrix.

    Parameters
    ----------
    M : NumPy array
        Input matrix.
    SNR : float, optional
        Signal-to-Noise Ratio. The default is 0.0.

    Returns
    -------
    NumPy Array
        Noisy matrix.

    """
    size_mat = M.shape
    N = np.random.randn(*size_mat)
    if np.iscomplex(M).any():
        N = N + 1j * np.random.randn(*size_mat)
    scale = la.norm(M, "fro") * (10 ** (-SNR / 20)) / la.norm(N, "fro")
    N *= scale
    M = M + N
    return (M, N)


def rmse(E: np.ndarray, D: tp.Union[np.ndarray, None] = None) -> np.ndarray:
    """
    Root Mean Square Error.

    Parameters
    ----------
    E : NumPy array
        Error or Estimates.
    D : NumPy array, optional
        Desideratum. The default is None.

    Returns
    -------
    None.

    """
    if D is None:
        E2 = E**2
    elif E.shape == D.shape:
        E2 = (E - D) ** 2
    else:
        E2 = (E - np.tile(D, (E.shape[1], 1)).T) ** 2
    rmse = np.mean(E2, axis=0) ** (1 / 2)
    return rmse


# %% Preprocessing


def lra(X: np.ndarray, R: int) -> np.ndarray:
    """
    Low-Rank Approximation.

    Parameters
    ----------
    X : np.ndarray
        Input matrix.
    R : int
        Rank.

    Returns
    -------
    NumPy array
        Low-rank approximation of input matrix.

    """
    U, s, VH = svds(X, k=R, return_singular_vectors=True, solver="lobpcg")
    return U @ np.diag(s) @ VH


def selM(M: int, Msub: int, shift: int = 0) -> np.ndarray:
    """
    Maximum overlap selection matrix.

    Parameters
    ----------
    M : int
        Number of array elements.
    Msub : int
        Number of subarrays.
    shift : int, optional
        Selection array shift. The default is 0.
        Maximum is M - Msub + 1 (the number of subarrays).

    Raises
    ------
    SystemExit
        Shift exceeds number of subarrays.

    Returns
    -------
    None.

    """
    L = M - Msub + 1
    if shift >= L:
        raise SystemExit("Shift cannot exceed no. of subarrays.")
    return np.hstack(
        (np.zeros((Msub, shift)), np.eye(Msub), np.zeros((Msub, M - (Msub + shift))))
    ).astype(int)


def fba(X: np.ndarray) -> np.ndarray:
    """
    Forward-backward averaging.

    Parameters
    ----------
    X : NumPy array
        Input matrix.

    Returns
    -------
    NumPy array
        Forward-backward averaged input matrix.

    """
    return np.hstack((X, np.rot90(X.conj(), 2)))


def sps(X: np.ndarray, L: int = 2) -> np.ndarray:
    """
    Spatial smoothing.

    Increases no. of samples using subarrays of the original data

    Parameters
    ----------
    X : NumPy array
        Input matrix.
    L : int, optional
        Number of subarrays. The default is 2.

    Raises
    ------
    SystemExit
        If number of subarrays >= the number of elements.

    Returns
    -------
    NumPy array
        Spatially smoothed input matrix.

    """
    M = X.shape[0]
    if L >= M:
        raise SystemExit("No. of subarrays cannot equal no. of elements")
    J = [selM(M, M - L + 1, m) for m in range(L)]
    return np.hstack([j @ X for j in J])


def desps(X: np.ndarray, L: int = 2) -> np.ndarray:
    """
    De-SPS: undo spatial smoothing.

    Does not check for consistency.

    Parameters
    ----------
    X : NumPy array
        Spatially smoothed input matrix.
    L : int, optional
        Number of subarrays. Defaults to 2.

    Raises
    ------
    SystemExit
        If no. of samples / no. of subarrays is not a whole no.

    Returns
    -------
    NumPy array
        Original input data.

    """
    Msub, LN = X.shape
    if np.mod(LN, L) == 0:
        N = int(LN / L)
        X_top = X[:, :N]
        X_row = X[Msub - 1, N:]
        return np.vstack((X_top, X_row.reshape((L - 1, N))))
    else:
        raise SystemExit("No. of samples must be a multiple of the no. of subarrays")


def MuDe(X: np.ndarray, D: int, L: int = 2) -> np.ndarray:
    """
    Multiple Denoising.

    Parameters
    ----------
    X : NumPy array
        Input matrix.
    D : int
        Model order.
    L : int, optional
        Number of subarrays. The default is 2.

    Raises
    ------
    SystemExit
        If number of subarrays >= number of elements.

    Returns
    -------
    NumPy array
        Processed matrix.

    """
    M = X.shape[0]
    if L >= M:
        raise SystemExit("No. of subarrays cannot equal no. of elements")
    W = lra(X, D)
    Y = sps(W, L)
    Z = lra(Y, D)
    return desps(Z, L)


# %% Array Processing


def ula(M: int, D: tp.Union[int, list, np.ndarray]) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Uniform Linear Array generator.

    Parameters
    ----------
    M : int
        Number of array elements.
    D : int or list
        Number of signals or directions of arrival in radians.

    Returns
    -------
    A : NumPy array
        Array steering matrix.
    az : NumPy array
        Directions of arrival (azimuths).

    """
    if isinstance(D, int):
        az = (np.random.rand(D) - 0.5) * np.pi
    else:
        az = D
    x = np.arange((1 - M), M, 2) / 2
    mu_x = np.pi * np.sin(az)
    A = np.exp(1j * np.outer(x, mu_x))
    return A, az


def esprit1D(
    X: np.ndarray, D: tp.Union[int, None] = None, tol: float = 1e-3
) -> np.ndarray:
    """
    One-dimensional ESPRIT.

    Parameters
    ----------
    X : NumPy array
        Input matrix.
    D : int, optional
        Model order. Defaults to no. of columns.
    tol : float, optional
        Tolerance. Defaults to 1e-6

    Returns
    -------
    mu : NumPy array
        Spatial frequencies vector

    """
    M = len(X)
    if D is None:
        if X.ndim == 1:
            if X.conj().T @ X - M == 0:
                return np.angle(X[1] / X[0])
            else:
                Us = X
        else:
            if np.trace(X.conj().T @ X - M * X.shape[1] == 0):
                return np.angle(X[1] / X[0])
            else:
                Us = X
    else:
        Us = svds(X, k=D, return_singular_vectors="u", solver="lobpcg")[0]
    Psi = la.pinv(Us[:-1]) @ Us[1:]
    Phi = la.eigvals(Psi)
    return np.angle(Phi)


def sort_sf(mu: np.ndarray, mu_hat: np.ndarray) -> tp.Tuple[np.ndarray, float]:
    """
    Sort spatial frequences.

    Parameters
    ----------
    mu : NumPy array
        True spatial frequencies.
    mu_hat : NumPy array
        Estimated spatial frequencies.

    Returns
    -------
    NumPy array
        Sorted spatial frequencies.
    Float
        Error.
    """
    D = len(mu)
    perms = list(it.permutations(range(D)))
    e = np.array([la.norm(mu - mu_hat[p,]) ** 0.5 for p in perms])
    idx = e.argmin()
    return (mu_hat[perms[idx],], e[idx])


# %% Model order estimation


def aic(X: np.ndarray) -> tp.Union[int, tp.Tuple[int, np.ndarray, np.ndarray]]:
    """
    Akaike Information Criterion.

    Usage:
        d, L, K = AIC(X)

    Parameters
    ----------
    X : np.ndarray
        Observation matrix.

    Returns
    -------
    int
        0 if model order estimation failed.
    int, NumPy array, NumPy array
        model order, L, and K.

    """
    M, N = X.shape
    R = np.cov(X)

    evs = np.flip(la.eig(R)[0])

    k = np.arange(M)
    K = 2 * k * (2 * M - k)

    if evs.sum() < 0:
        return 0
    else:
        L = np.array(
            [
                -2 * N * (M - m) * np.log((gmean(evs[m:M]) / (evs[m:M].mean())))
                for m in range(M)
            ]
        ).real
        aic = np.flip(L + K)
        return (aic.argmin() + 2, np.flip(L), np.flip(K))


def eft(X: np.ndarray, tol: float = 1e-2, q: tp.Union[None, float] = None) -> int:
    """
    Exponential Fitting Test.

    Usage:
        d = EFT(X)
        d = EFT(X, tol)
        d = EFT(X, tol, q)

    Parameters
    ----------
    X : np.ndarray
        Observation matrix.
    tol : float, optional
        Tolerance. The default is 1e-2.
    q : float, optional
        Adjustment function value. Defaults to classical EFT.

    Returns
    -------
    int
        DESCRIPTION.

    """
    M, N = X.shape
    R = np.cov(X)

    evs = la.eig(R)[0]
    if q is None:
        q = np.exp(
            np.sqrt(
                30 / (M**2 + 2)
                - np.sqrt(
                    900 / ((M**2 + 2) ** 2) - 720 * M / (N * (M**4 + M**2 - 2))
                )
            )
        )
    eft = evs[-1] * (q ** np.arange(M - 1, -1, -1))
    d = 0
    while abs(evs[d] - eft[d]) > tol:
        d = d + 1
    return d
