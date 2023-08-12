# %% Docstring

"""
Array Signal Processing.

------------------------

Includes:
    1. Basic operations
        Column normalization (colnorm)
        Add noise to RX matrix (noisy)
        Root mean square error (rmse)
        Least-Squares Khatri-Rao factorization (lskrf)
    2. Preprocessing
        Low-rank approximation (lra)
        Selection matrix (selM)
        Forward-backward averaging (fba)
        Spatial smoothing (sps)
        De-sps (desps)
        Multiple denoising (MuDe)
    3. Beamforming
        Bartlett's beamformer (bartlett)
        Capon's beamformer (capon)
        Linear predictor beamformer (linear_pred)
        Multiple signal classification (music)
        MUSIC polynomial roots (music_roots)
        MUSIC root selector (root_music)
    4. Array processing
        Uniform linear array (ula)
        1-D ESPRIT (esprit1D)
        Sort spatial frequencies (sort_sf)
    5. Model order estimation
        Minimum Descriptor Length (mdl)
        Akaike Information Criterion (aic)
        Exponential Fitting Test (eft)
"""

# %% __all__

__all__ = [
    "colnorm",
    "noisy",
    "rmse",
    "lskrf",
    "lra",
    "selM",
    "fba",
    "sps",
    "desps",
    "MuDe",
    "beamformer",
    "bartlett",
    "capon",
    "linear_pred",
    "music",
    "music_roots",
    "pair_roots",
    "root_music",
    "una",
    "ula",
    "esprit1d",
    "sort_sf",
    "mdl",
    "aic",
    "eft",
]

# %% Load dependencies

import itertools as it

import numpy as np
import numpy.linalg as la
import typing as tp

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
    scale = la.norm(M, 'fro') * (10 ** (-SNR / 20)) / la.norm(N, 'fro')
    N *= scale
    M = M + N
    return (M, N)


def rmse(E: np.ndarray,
         D: tp.Union[np.ndarray, None] = None) -> np.ndarray:
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
    NumPy array
        Root mean squared error.
    """
    if D is None:
        E2 = E**2
    elif E.shape == D.shape:
        E2 = (E - D) ** 2
    else:
        E2 = (E - np.tile(D, (E.shape[1], 1)).T) ** 2
    rmse = np.mean(E2, axis=0) ** (1 / 2)
    return rmse


def lskrf(K: np.ndarray,
          M: tp.Union[int, list]) -> list:
    """
    Least squares Khatri-Rao factorization.

    Parameters
    ----------
    K : NumPy array
        Khatri-Rao product.
    M : int, list
        Size of first dimension or list
        of sizes (of first dimension)

    Returns
    -------
    list (of NumPy arrays)
        Factor matrices.
    """
    R, C = K.shape
    if isinstance(M, int):
        M = [M, R // M]
    N = len(M)
    F = []
    for n in range(N - 1):
        FnK = [np.reshape(K[:, c], [M[n], np.prod(M[(n + 1):])])
               for c in range(C)]
        U, s, V = zip(*[la.svd(m, full_matrices=False) for m in FnK])
        S = np.diagflat([ess[0] ** 0.5 for ess in s])
        U = np.stack([u[:, 0] for u in U], axis=1)
        V = np.stack([v[0] for v in V], axis=1)
        F.append(U @ S)
        K = V @ S
    F.append(K)
    return F


def vkrf(k: np.ndarray,
         M: tp.Union[int, list]) -> tp.List[np.ndarray]:
    """
    Vector Khatri-Rao factorization

    Parameters
    ----------
    k : np.ndarray
        Khatri-Rao product vector.
    M : tp.Union[int, list]
        Vector length.

    Returns
    -------
    list
        List of vectors.

    """
    R = len(k)
    if isinstance(M, int):
        M = [M, R // M]
    N = len(M)
    f = []
    for n in range(N-1):
        K = k.reshape(M[n], np.prod(M[n+1:]))
        u, s, vh = la.svd(K, full_matrices=False)
        s = (s[0] ** 0.5)
        f.append(u[:, 0] * s)
        k = vh[0] * s
    f.append(k)
    return f


# %% Preprocessing


def lra(X: np.ndarray,
        R: tp.Union[int, None] = None) -> np.ndarray:
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
    if R is None:
        R = min(X.shape)
    U, s, VH = la.svd(X, full_matrices=False)
    return U[:, :R] @ np.diag(s[:R]) @ VH[:R]


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
    return np.hstack((np.zeros((Msub, shift)),
                      np.eye(Msub),
                      np.zeros((Msub, M - (Msub + shift)))))


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
        err_msg = "No. of samples must be a multiple of the no. of subarrays"
        raise SystemExit(err_msg)


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


# %% Beamforming


def beamformer(X: np.ndarray,
               A: tp.Union[list, np.ndarray] = None) -> list:
    """
    Interface function for beamformers.

    Parameters
    ----------
    X : NumPy array
        Beamformer kernel.
    A : NumPy array, optional
        Direction vector(s). The default is None.

    Returns
    -------
    NumPy array
        Beamformer pseudo-spectrum.

    """
    M = X.shape[0]
    if A is None:
        A = ula(M, np.linspace(-np.pi / 2, np.pi / 2, 181))
    if isinstance(A, list):
        return [np.conj(a.T) @ X @ a for a in A]
    return [np.conj(a.T) @ X @ a for a in A.T]


def bartlett(X: np.ndarray,
             A: tp.Optional[np.ndarray] = None) -> np.ndarray:
    """
    Bartlett's (conventional) beamformer

    Parameters
    ----------
    X : NumPy array
        Sample matrix or covariance matrix (square).
    A : NumPy array
        Azimuth steering vector matrix.

    Returns
    -------
    NumPy array
        Spectral power (dB) of Bartlett's beamformer.

    """
    M, N = X.shape
    if (M == N):
        Rxx = X
    else:
        Rxx = X @ np.conjugate(X.T) / N
    return 20 * np.log10(beamformer(Rxx, A)).real


def capon(X: np.ndarray,
          A: tp.Optional[np.ndarray] = None) -> np.ndarray:
    """
    Capon's beamformer

    Parameters
    ----------
    X : NumPy array
        Sample matrix or inverse of covariance matrix (square).
    A : NumPy array
        Azimuth steering vector matrix.

    Returns
    -------
    NumPy array
        Spectral power (dB) of Capon's beamformer.

    """
    M, N = X.shape
    if (M == N):
        invRxx = X
    else:
        Rxx = X @ np.conjugate(X.T) / N
        invRxx = la.inv(Rxx)
    return -20 * np.log10(beamformer(invRxx, A)).real


def linear_pred(X: np.ndarray,
                A: tp.Optional[np.ndarray] = None,
                u: tp.Optional[int] = None) -> np.ndarray:
    """
    Linear prediction beamformer

    Parameters
    ----------
    X : NumPy array
        Sample matrix or inverse of covariance matrix (square).
    A : NumPy array, optional
        Azimuth steering vector matrix.
    u : int, optional
        Selection vector or index

    Returns
    -------
    NumPy array
        Spectral power (dB) of Capon's beamformer.

    """
    M, N = X.shape
    if (M == N):
        invRxx = X
    else:
        Rxx = X @ np.conjugate(X.T) / N
        invRxx = la.inv(Rxx)
    if A is None:
        A = ula(M, np.linspace(-np.pi / 2, np.pi / 2, 181))[0]
    if u is None:
        return -40 * np.log10(np.abs([(invRxx @ a)[0]
                                      for a in A.T]))
    elif isinstance(u, int):
        return -40 * np.log10(np.abs([(invRxx @ a)[u]
                                      for a in A.T]))
    return -40 * np.log10(np.abs([(np.conj(u.T) @ invRxx) @ a
                                  for a in A.T]))


def music(X: np.ndarray, R: int,
          A: tp.Optional[np.ndarray] = None) -> np.ndarray:
    """
    MUSIC beamformer

    Parameters
    ----------
    X : NumPy array
        Sample matrix.
    R : int, optional
        Rank.
    A : NumPy array
        Azimuth steering vector matrix.

    Returns
    -------
    NumPy array
        Spectral power (dB) of MUSIC beamformer.

    """
    M, N = X.shape
    if N == M:
        if la.norm(X) > (M - R):
            En = la.eig(X)[1][:, R:]
            C = En @ np.conj(En.T)
        else:
            C = X
    elif N == M - R:
        C = X @ np.conj(X.T)
    else:
        Rxx = X @ np.conj(X.T) / N
        En = la.eig(Rxx)[1][:, R:]
        C = En @ np.conj(En.T)
    return -20 * np.log10(beamformer(C, A)).real


def music_roots(C: np.ndarray) -> np.ndarray:
    """
    Find roots of MUSIC polynomial

    Parameters
    ----------
    C : NumPy array
        Noise subspace covariance matrix.

    Returns
    -------
    NumPy array
        MUSIC roots (all 2*(M-1) roots).
    """
    M: int = C.shape[0]
    coefficients: tp.List[np.complex128] = [sum(np.diagonal(C, offset))
                                            for offset in range(M - 1, -M, -1)]
    return np.roots(coefficients)


def pair_roots(roots: np.ndarray) -> tp.List:
    """
    Sorts MUSIC polynomial roots.

    Parameters
    ----------
    roots : NumPy array
        MUSIC root.

    Returns
    -------
    List
        MUSIC root pairs.

    """
    angles = np.angle(roots)
    index = np.argsort(angles)
    roots = roots[index]
    pair_index = np.array([n for n in range(0, len(roots), 2)])
    pairs = np.array([roots[n: n + 2] for n in pair_index])
    diffs = [abs(np.diff(pair)[0]) for pair in pairs]
    diff_sort = np.argsort(diffs)
    return pairs[diff_sort]


def root_music(X: np.ndarray, R: int) -> tp.List:
    """
    Selects valid MUSIC roots.

    Parameters
    ----------
    roots : NumPy array
        DESCRIPTION.
    R : int
        DESCRIPTION.

    Returns
    -------
    NumPy array.

    """
    N = X.ndim
    if N == 1:
        return np.array([np.angle(pair[0])
                         for pair in pair_roots(X)[:R]])
    M, N = X.shape
    if N == M:
        roots = music_roots(X)
    else:
        Rxx = X @ np.conj(X.T) / N
        En = la.eig(Rxx)[0][:, R:]
        C = En @ np.conj(En.T)
        roots = music_roots(C)
    return np.array([np.angle(pair[0])
                     for pair in pair_roots(roots)[:R]])


# %% Array Processing

def una(M: int, mu: tp.Union[int, list]) -> np.ndarray:
    """
    Uniform (linear) array

    Parameters
    ----------
    M : int
        Number of elements.
    mu : tp.Union[int, list]
        Spatial frequencies.

    Returns
    -------
    NumPy array.

    """
    return np.exp(1j / 2 * np.outer(np.arange(1 - M, M, 2), mu))


def ula(M: int,
        D: tp.Union[int, list]) -> np.ndarray:
    """
    Uniform Linear Array generator.

    Parameters
    ----------
    M : int
        Number of array elements.
    D : int or list
        Number of Tx sources or directions of arrival (in radians).

    Returns
    -------
    A : NumPy array
        Array steering matrix.
    """
    if isinstance(D, int):
        az = (np.random.rand(D) - 0.5) * np.pi
    else:
        az = D
    mu = np.pi * np.sin(az)
    return una(M, mu)


def estimu(A: np.ndarray) -> np.ndarray:
    """
    Estimate spatial frequencies (from steering matrix)

    Parameters
    ----------
    A : NumPy array (matrix)
        Steering matrix.

    Returns
    -------
    mu : NumPy array (vector)
        Spatial frequencies.

    """
    return np.angle(A[1] / A[0])


def esprit1d(X: np.ndarray,
             D: tp.Union[int, None] = None) -> np.ndarray:
    """
    One-dimensional ESPRIT.

    Parameters
    ----------
    X : NumPy array
        Input matrix.
    D : int, optional
        Model order. Defaults to no. of columns.

    Returns
    -------
    mu : NumPy array
        Spatial frequencies vector

    """
    if X.ndim == 2:  # matrix
        N = X.shape[1]  # no. cols
        if N > D:
            Us = la.svd(X, full_matrices=False)[0][:, :D]
            Psi = la.pinv(Us[:-1]) @ Us[1:]
            Phi = la.eigvals(Psi)
        elif N == D:
            if (la.norm(X, axis=0) == np.ones(D)).all():
                Psi = la.pinv(X[:-1]) @ X[1:]
                Phi = la.eigvals(Psi)
            else:
                Phi = X[1] / X[0]
    else:
        Phi = X[1] / X[0]
    return np.angle(Phi)


def sort_sf(mu: np.ndarray,
            mu_hat: np.ndarray) -> tp.Tuple[np.ndarray, float]:
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


def mdl(X: np.ndarray,
        fba: tp.Optional[bool] = False) -> tp.Tuple[int,
                                                    np.ndarray,
                                                    np.ndarray]:
    """
    Minimum descriptor length model order estimator

    Parameters
    ----------
    X : np.ndarray
        Observation matrix.
    fba : bool, optional. Default is False
        If foward-backward averaging has been applied

    Returns
    -------
    R : int
        Estimated model order (NOT zero-indexed)
    L: NumPy array
        Likelihood function
    P : NumPy array
        Penalty function

    """
    M, N = X.shape
    Rxx = X @ np.conj(X.T) / N

    eigenvalues = la.eigvals(Rxx).real
    if eigenvalues.sum() < 0:
        return 0

    if np.any(np.iscomplex(Rxx)):
        if fba:
            p = [0.5 * r * (2 * M - r + 1) for r in range(M)]
        else:
            p = [r * (2 * M - r) for r in range(M)]
    else:
        if fba:
            p = [r * (2 * M + r + 1) for r in range(M)]
        else:
            p = [r * (2 * M - r + 1) for r in range(M)]
    P = 0.5 * np.array(p) / np.log(N)

    L = -np.log([gmean(eigenvalues[r:M] / np.mean(eigenvalues[r:M]))
                 for r in range(M)])

    MDL = L + P
    return np.argmin(MDL), L, P


def aic(X: np.ndarray,
        fba: tp.Optional[bool] = False) -> tp.Tuple[int,
                                                    np.ndarray,
                                                    np.ndarray]:
    """
    Akaike information criterion model order estimator

    Parameters
    ----------
    X : np.ndarray
        Observation matrix.
    fba : bool, optional. Default is False
        If foward-backward averaging has been applied

    Returns
    -------
    R : int
        Estimated model order
    L: NumPy array
        Likelihood function
    P : NumPy array
        Penalty function

    """
    M, N = X.shape
    if M == N:
        Rxx = X
    else:
        Rxx = X @ np.conj(X.T) / N

    eigenvalues = la.eigvals(Rxx).real
    if eigenvalues.sum() < 0:
        return 0

    if np.any(np.iscomplex(X)):
        if fba:
            p = [0.5 * r * (2 * M - r + 1) for r in range(M)]
        else:
            p = [r * (2 * M - r) for r in range(M)]
    else:
        if fba:
            p = [r * (M + r + 1) for r in range(M)]
        else:
            p = [r * (2 * M - r + 1) for r in range(M)]
    P = np.array(p)

    L = np.array([(r - M) * N * np.log(gmean(eigenvalues[r:M])
                                       / np.mean(eigenvalues[r:M]))
                  for r in range(M)])
    AIC = L + P
    return np.argmin(AIC) - 1, L, P


def eft(X: np.ndarray,
        tol: float = 1e-2,
        q: tp.Union[float, None] = None) -> int:
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

    evs = la.eigvals(R)
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
        d += 1
    return d
