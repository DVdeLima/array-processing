# %% Docstring

"""
Array Signal Processing.

------------------------

Includes:
    0. Convenience functions
        Column normalization (colnorm)
        Min/Max normalization (min_max)
        Random ranges (randges)
        Selection matrix generator (selM)
        Triangular reconditioning (tri)
    1. Basic operations
        Add noise to Rx matrix (noisy)
        Estimate SNR (est_snr)
        Root mean square error (rmse)
        Least-Squares Khatri-Rao factorization (lskrf)
    2. Preprocessing
        Low-rank approximation (lra)
        Selection matrix (selM)
        Forward-backward averaging (fba)
        Spatial smoothing (sps)
        De-sps (desps)
        Multiple denoising (MuDe)
    3. Recursive covariance estimation
        Covariance matrix update (cov_upd)
        Inverse covariance matrix update (inv_cov_upd)
        Inverse spectrum update (inv_spec_upd)
        Recursive Capon update
    4. Beamforming
        Bartlett's beamformer (bartlett)
        Capon's beamformer (capon)
        Miniumum variance distortionless response (mvdr)
        Linear predictor beamformer (linear_pred)
        Multiple signal classification (music)
        MUSIC polynomial roots (music_roots)
        MUSIC root selector (root_music)
    5. Array processing
        Uniform linear array (ula)
        1-D ESPRIT (esprit1D)
        Sort spatial frequencies (sort_sf)
    6. Model order estimation
        Minimum Descriptor Length (mdl)
        Akaike Information Criterion (aic)
        Exponential Fitting Test (eft)
"""

# %% __all__

__all__ = [
    "colnorm",
    "min_max",
    "randges",
    "selM",
    "tri",
    "colnorm",
    "noisy",
    "est_snr",
    "rmse",
    "lskrf",
    "lra",
    "fba",
    "sps",
    "desps",
    "MuDe",
    "beamformer",
    "bartlett",
    "capon",
    "mvdr",
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

import numpy as np
import numpy.linalg as la
import typing as tp

from itertools import permutations
from scipy.stats import gmean
from scipy.signal import find_peaks


# %% Convenience functions

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


def min_max(power: np.ndarray,
            min_power: bool = True) -> np.ndarray:
    """
    Min/Max normalization

    Parameters
    ----------
    power : NumPy array
        Data vector.

    Returns
    -------
    None.

    """
    min_power = min(power)
    max_power = max(power)
    diff = max_power - min_power
    if min_power:
        return (power - min_power) / diff
    return (power - max_power) / diff


def randges(R: int,
            start: float = -90.0,
            end: float = 90.0,
            biased: float = 3.0) -> np.ndarray:
    """
    Draw from random ranges for DoA estimation

    Parameters
    ----------
    R : int
        Number of ranges (R > 1).
    start : float, optional
        Start of distribution. The default is -90.0.
    end : float, optional
        End of distribution. The default is 90.0.
    biased : float, optional
        Number of standard deviations in each half
        range for Normal distribution draw. If set
        to 0 or False uses Uniform distribution.
        The default is 3.0.

    Returns
    -------
    draw : NumPy array
        Draw.

    """
    delta = abs(start - end) / R
    draw = start + delta * np.arange(R)
    if biased:
        draw += delta * (1 + np.random.randn(R) / biased) / 2
        draw[draw < start] = start
        draw[draw > end] = end
    else:
        draw += delta * np.random.rand(R)
    return draw


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


def tri(A: np.ndarray) -> np.ndarray:
    """
    Triangular (re)conditioning.

    Parameters
    ----------
    A : NumPy array
        Input (Hermitian) matrix.

    Returns
    -------
    NumPy array
        Reconditioned matrix.

    """
    return (A + A.conj().T) / 2


# %% Basic operations

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


def est_snr(X: np.ndarray, Xo: np.ndarray) -> float:
    return 20 * np.log10(la.norm(Xo) / la.norm(X - Xo))


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
    Ms = M - L + 1
    row_slices = [slice(ell, Ms + ell) for ell in range(L)]
    return np.hstack([X[row_slice] for row_slice in row_slices])


def desps(X_sps: np.ndarray, L: int = 2) -> np.ndarray:
    """
    Recover sample matrix from spatially smoothed samples

    Parameters
    ----------
    X : NumPy array
        DESCRIPTION.
    L : int, optional
        Number of subarrays. The default is 2.

    Returns
    -------
    NumPy array
        Sample matrix.

    """
    N = X_sps.shape[1] // L
    col_slices = [slice(ell * N, (ell + 1) * N) for ell in range(1, L)]
    return np.vstack((X_sps[:, :N], *[X_sps[-1, col_slice][None, :]
                                      for col_slice in col_slices]))


def MuDe(X: np.ndarray, D: int, L: int | list = 1) -> np.ndarray:
    """
    Multiple Denoising

    Parameters
    ----------
    X : np.ndarray
        DESCRIPTION.
    D : int
        DESCRIPTION.
    L : int | list, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    X_MuDe : TYPE
        DESCRIPTION.

    """
    X_MuDe = lra(X, D)
    if L == 1:
        return X_MuDe
    for ell in L:
        X_sps = sps(X_MuDe, ell)
        X_lra = lra(X_sps)
        X_MuDe = desps(X_lra, ell)
    return X_MuDe


# %% Recursive cov. est.


def cov_upd(covariance: np.ndarray,
            x: np.ndarray,
            forget: float = 0.8) -> tp.Tuple[np.ndarray]:
    """
    Covariance matrix update

    Parameters
    ----------
    covariance : NumPy array
        Previous covariance matrix.
    x : NumPy array
        Current data vector.
    forget : float, optional
        Forgetting factor. The default is 0.8.

    Returns
    -------
    NumPy array
        Updated covariance matrix.

    """
    if x.ndim == 1:
        return forget * covariance + np.outer(x, x.conj())
    return forget * covariance + x @ x.T.conj() / x.shape[1]


def inv_cov_upd(inv_covariance: np.ndarray,
                x: np.ndarray,
                forget: float = 0.8) -> np.ndarray:
    g = inv_covariance @ x  # a priori Kalman filter gain
    phi = forget / (forget + x.T.conj() @ g)
    inv_covariance = inv_covariance / forget \
        - np.outer(g, g.conj()) * phi / forget ** 2
    return tri(inv_covariance), g, phi


def inv_spec_upd(inv_spectrum: np.ndarray,
                 g: np.ndarray,
                 phi: float,
                 forget: float = 0.8,
                 F: np.ndarray = None) -> np.ndarray:
    """
    Inverse spectrum update

    Parameters
    ----------
    inv_spectrum : NumPy array
        Previous inverse spectrum.
    g : NumPy array
        Kalman gain vector.
    phi : float
        Scaling factor.
    forget : float, optional
        Forgetting factor. The default is 0.8.
    F : NumPy array, optional
        Frequency sweep matrix. The default is None.

    Returns
    -------
    inv_spectrum: NumPy array
        Updated inverse spectrum.

    """
    L = len(g)
    K = len(inv_spectrum)
    if F is None:
        F = ula(L, np.linspace(-np.pi / 2, np.pi / 2, K))
    phi = phi / (K * (1 - forget))
    v = F.T.conj() @ g
    return inv_spectrum / forget - phi * (abs(v) / forget) ** 2


def recapon_upd(inv_covariance: np.ndarray,
                x: np.ndarray,
                inv_spectrum: np.ndarray,
                forget: float = 0.8,
                F: np.ndarray = None) -> tp.Tuple[np.ndarray,
                                                  np.ndarray]:
    if F is None:
        L = len(x)
        K = 181
        F = ula(L, np.linspace(-np.pi / 2, np.pi / 2, 181))
    else:
        L, K = F.shape
    inv_covariance, g, phi = inv_cov_upd(x, inv_covariance)
    # Inv. cov. matrix est. upd.
    inv_spectrum = inv_spec_upd(inv_spectrum, g, phi, forget, F)
    return inv_covariance, inv_spectrum


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
    M = len(X)
    if A is None:
        A = ula(M, np.linspace(-np.pi / 2, np.pi / 2, 181))
    if isinstance(A, list):
        return [np.conj(a.T) @ X @ a for a in A]
    return [np.conj(a.T) @ X @ a for a in A.T]


def bartlett(X: np.ndarray,
             A: tp.Optional[list] = None) -> np.ndarray:
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
          A: tp.Optional[list] = None) -> np.ndarray:
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


def mvdr(mu: np.ndarray, invRxx: np.ndarray) -> np.ndarray:
    """
    Minimum variance distortionless response filter

    Parameters
    ----------
    mu : NumPy array
        Spatial frequency(ies).
    invRxx : NumPy array
        Inverse covariance matrix.

    Returns
    -------
    NumPy array
        Filter.

    """
    M = len(invRxx)
    return invRxx @ una(M, mu) / M


def linear_pred(X: np.ndarray,
                A: tp.Optional[list] = None,
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


def find_spec_peak_idxs(spec: np.ndarray,
                        R: tp.Optional[int] = None) -> np.ndarray:
    """
    Returns indexes of spectrum peaks.

    Parameters
    ----------
    spec : NumPy array
        Spectrum.
    R : int, optional
        Number of peaks (model order). The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    peaks = find_peaks(spec)[0]
    if R is None:
        return peaks
    return peaks[np.argsort(spec[peaks])[::-1][:R]]


def music(X: np.ndarray, R: int,
          A: tp.Optional[list] = None) -> np.ndarray:
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


def minimum_norm(X: np.ndarray, R: int,
                 A: tp.Optional[list] = None) -> np.ndarray:
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
    W = np.zeros((M, M))
    W[0, 0] = 1.0
    return -20 * np.log10(beamformer(C @ W @ C, A)).real


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
    M: int = len(C)
    coefficients: tp.List[np.complex128] = [np.trace(C, offset)
                                            for offset in range(1-M, M)]
    return np.roots(coefficients).conj()


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
        MUSIC polynomial roots.
    R : int
        Model order.

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


def root(X: np.ndarray, R: int) -> np.ndarray:
    """
    Calculate Root-MUSIC spatial frequencies

    Parameters
    ----------
    X : NumPy array
        Noise subspace covariance matrix.
    R : int
        Model order.

    Returns
    -------
    NumPy array
        Spatial frequencies.

    """
    return root_music(music_roots(X), R)


def mmse(invRxx: np.ndarray, a: np.ndarray,
         y: tp.Optional[np.ndarray] = None) -> np.ndarray:
    mmse_estimator = a.T.conj() @ invRxx / (a.T.conj() @ invRxx @ a)
    if y is None:
        return mmse_estimator
    return mmse_estimator @ y


# %% Estimate pairing

def binary_seq(R: int, zero: bool = False) -> np.ndarray:
    """
    Returns a binary sequence for R bits

    Parameters
    ----------
    R : int
        Number of bits.

    Returns
    -------
    seq: NumPy array
        Binary sequence.

    """
    seq = np.array([np.array(list(np.binary_repr(n, R))).astype(int)
                    for n in range(2**R)])
    if not zero:
        seq = 1 - 2 * seq
    return seq


def naive_pairing(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """
    Naïve pairing

    Parameters
    ----------
    previous : NumPy array
        Previous estimates.
    current : NumPy array
        Current estimates.

    Returns
    -------
    NumPy array
        Naïve paired current estimates.

    """
    candidates = tuple(permutations(current))
    idx_arg_min_pdn = la.norm(candidates - previous, axis=1).argmin()
    if idx_arg_min_pdn:
        return candidates[idx_arg_min_pdn]
    return current


def limit_switch(current: np.ndarray,
                 limit: tp.Optional[float] = None) -> np.ndarray:
    if limit is None:
        if np.any(np.abs(current) > np.pi/2):
            limit = 80
        else:
            limit = 1.4
    R = len(current)
    above_limit = abs(current) > limit
    L = len(np.nonzero(above_limit)[0])
    switch = np.ones((2 ** L, R))
    if L == 1:
        switch[:, above_limit] = np.asmatrix([1, -1]).T
    elif L > 1:
        ell = 0
        seq = binary_seq(L)
        for col, judgement in enumerate(above_limit):
            if judgement:
                switch[:, col] = seq[:, ell]
                ell += 1
    else:
        return current
    return switch @ np.diagflat(current)


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
    if isinstance(D, int) and D != 0:
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
             R: int = None) -> np.ndarray:
    """
    One-dimensional ESPRIT.

    Parameters
    ----------
    X : NumPy array
        Input matrix.
    R : int, optional
        Model order. If not specified assumed signal subspace was used.

    Returns
    -------
    mu : NumPy array
        Spatial frequencies vector

    """
    if R is None:
        Us = X
    else:
        Us = la.svd(X, full_matrices=False)[0][:, :R]
    if Us.ndim == 1:
        return estimu(Us)
    Psi = la.pinv(Us[:-1]) @ Us[1:]
    phi = la.eigvals(Psi)
    return np.angle(phi)


def sort_sf(mu: np.ndarray,
            mu_hat: np.ndarray = None) -> np.ndarray:
    if mu_hat is None:
        N, R = mu.shape
        mu_hat = np.zeros((N, R))
        mu_hat[0] = mu[0]
        for n in range(1, N):
            mu_hat[n] = sort_sf(mu_hat[n-1], mu[n])
        return mu_hat
    R = len(mu)
    perm_list = list(permutations(range(R)))
    error = la.norm([mu - mu_hat[list(perm)]
                     for perm in perm_list], axis=1)
    return mu_hat[list(perm_list[error.argmin()])]


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
    L : NumPy array
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
        fba: bool = False) -> tp.Tuple[int,
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
    L : NumPy array
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

    r = np.arange(M)
    if np.any(np.iscomplex(X)):
        if fba:
            P = 0.5 * r * (2 * M - r + 1)
        else:
            P = r * (2 * M - r)
    else:
        if fba:
            P = r * (M + r + 1)
        else:
            P = r * (2 * M - r + 1)

    L = np.array([(r - M) * N * np.log(gmean(abs(eigenvalues[r:M]))
                                       / np.mean(abs(eigenvalues[r:M])))
                  for r in range(M)])
    AIC = L + P
    return np.argmin(AIC), L, P


def eft(X: np.ndarray,
        tol: float = 1e2,
        q: tp.Union[float, None] = None) -> tp.Tuple[int, np.ndarray]:
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
    if M == N:
        Rxx = X
    else:
        Rxx = X @ np.conj(X.T) / N

    evs = la.eigvals(Rxx).real
    if q is None:
        q = np.exp(np.sqrt(30 / (M**2 + 2)
                           - np.sqrt(900 / ((M ** 2 + 2) ** 2)
                                     - 720 * M / (N * (M ** 4 + M ** 2 - 2)))))
    cost = evs[-1] * (q ** np.arange(M - 1, -1, -1))
    d = 0
    while abs(evs[d] - cost[d]) < tol:
        d += 1
    return d, cost
