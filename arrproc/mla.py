# %% Docstring

"""
Multilinear Algebra.

--------------------

Includes:
    1. Basic tensor operations
        Column normalization (colnorm)
        Kronecker product (kron)
        Khatri-Rao product (kr)
        Unfold tensor (unfold)
        Fold tensor (fold)
        Tensor-matrix mode product (modeprod)
        Tensor-matrix multimode product (mmprod)
        Generate identity tensor (eyeNR)
        Generate tensor from CPD (cpdgen)
        Generate L-rank identity tensor (eyeNL)
        Generate L, L, 1-rank tensor (ll1gen)
        Estimate SNR (estSNR)
        Add noise to tensor/matrix (noisy)
    2. Preprocessing
        Low-rank approximation (lra)
        Forward-backward averaging (fba)
        Generate unitary matrix (qunit)
        Unitary transformation (unitransf)
        "Cheap" unitary transformation (cheapUT)
        Spatial smoothing (sps)
    3. R-D array processing
        Generate R-dim. steering matrices (una)
        Generate lin/rect/par steering matrices (uxa)
        Standard Tensor ESPRIT (ste)
        GEVD-based spatial freq. pairing (gevdpair)
        Unitary Tensor ESPRIT (ute)
        UTE-based spatial freq. pairing (utepair)
    4. Multilinear SVD (MLSVD)
        Seq. Trunc. MLSVD (stmlsvd)
        Trunc. MLSVD (tmlsvd)
        Est. core tensor (estcore)
    5. Canonical Polyadic Decomposition (CPD)
        CPD via GEVD (cpdgevd)
        CPD via GEVD2 (cpdgevd2)
        CPD via SVD/EVD (cpdsevd)
        LS Khatri-Rao factorization (lskrf)
"""

# %% __all__

__all__ = [
    "frob",
    "colnorm",
    "kron",
    "kr",
    "unfold",
    "fold",
    "modeprod",
    "mmprod",
    "eyeNR",
    "cpdgen",
    "eyeNL",
    "ll1gen",
    "estSNR",
    "noisy",
    "lra",
    "fba",
    "qunit",
    "unitransf",
    "cheapUT",
    "sps",
    "una",
    "uxa",
    "ste",
    "gevdpair",
    "ute",
    "utepair",
    "sfcheck",
    "stmlsvd",
    "tmlsvd",
    "estcore",
    "cpdgevd",
    "cpdgevd2",
    "cpdsevd",
    "lskrf",
]

# %% Load dependencies

import numpy as np
import scipy.linalg as la
import typing as tp

# %% Load individual functions

from scipy.sparse.linalg import svds, eigs
from itertools import permutations

# %% Debug modules

# import pdb as ipdb
# ipdb.set_trace()


# %% Basic operations


def frob(X: np.ndarray, axis: tp.Union[int, None] = None) -> float:
    """
    Frobenius norm.

    Parameters
    ----------
    X : NumPy array
        Vector or matrix or tensor.
    axis : int, optional
        Axis along which the norm is calculated.

    Returns
    -------
    float
        Frobenius norm.

    """
    if isinstance(axis, int):
        return [frob(x) for x in unfold(X, axis)]
    return np.sqrt((X * X.conj()).real.sum())


def colnorm(X: np.ndarray) -> np.ndarray:
    """
    Column Normalization.

    Parameters
    ----------
    X : NumPy array
        Input matrix or tensor.

    Returns
    -------
    NumPy array
        Input matrix or tensor with normalized "columns".
    """
    N = X.ndim
    if N == 1:
        return X / la.norm(X)
    elif N == 2:
        return X @ np.diag(1 / la.norm(X, axis=0))
    else:
        D = np.diag(1 / np.array(frob(X, N - 1)))
        return modeprod(X, D, N - 1)


def kron(F: list) -> np.ndarray:
    """
    Kronecker product

    Parameters
    ----------
    F : list
        List of factor matrices.

    Raises
    ------
    Exception
        Number of factor matrices > 13.

    Returns
    -------
    NumPy array
        Kronecker product.
    """
    N = len(F)
    if N > 13:
        raise Exception("Too many factor matrices!")
    findvec = [f.ndim == 1 for f in F]
    if np.any(findvec):
        F = [np.matrix(f).T if vec else f for f, vec in zip(F, findvec)]
    kron_shape = np.prod([f.shape for f in F], axis=0)
    a = ord("a")  # 97
    row_list = [chr(m) for m in range(a, a + N)]
    col_list = [chr(n) for n in range(a + N + 1, a + 2 * N + 1)]
    alt_list = [r + c for r, c in zip(row_list, col_list)]
    subscripts = ",".join(alt_list) + "->" + "".join(row_list + col_list)
    return np.einsum(subscripts, *F).reshape(kron_shape)


def kr(F: list) -> np.ndarray:
    """
    Khatri-Rao product

    Parameters
    ----------
    F : list
        List of factor matrices (same no. of columns).

    Raises
    ------
    Exception
        Number of factor matrices > 25.

    Returns
    -------
    NumPy array
        Khatri-Rao product.
    """
    N = len(F)
    if N > 25:
        raise Exception("Too many factor matrices!")
    rows, cols = zip(*[f.shape for f in F])
    if not np.all([c == cols[0] for c in cols]):
        error_msg = "All factor matrice must have the same number of columns!"
        raise Exception(error_msg)
    kr_shape = [np.prod(rows), cols[0]]
    a = ord("a")  # 97
    row_list = [chr(m) for m in range(a, a + N)]
    subscripts = "z,".join(row_list) + "z->" + "".join(row_list) + "z"
    return np.einsum(subscripts, *F).reshape(kr_shape)


def unfold(T: np.ndarray, mode: int = 0):
    """
    Returns mode-unfolding of tensor.

    Parameters
    ----------
    T : NumPy array
        Tensor.
    mode : int, optional
        Unfolding mode. The default is 0 (first mode).

    Returns
    -------
    NumPy array
        mode-unfolded tensor.
    """
    size_tensor = list(T.shape)
    size_matrix = [size_tensor.pop(mode), np.array(size_tensor).prod()]
    if mode:
        N = T.ndim
        axis_order = np.insert(np.arange(1, N), mode, 0)
        T = np.moveaxis(T, np.arange(N), axis_order)
    return np.reshape(T, size_matrix)


def fold(M: np.ndarray, mode: int, sizes: list) -> np.ndarray:
    """
    Fold unfolded tensor back into tensor.

    Parameters
    ----------
    M : NumPy array
        Unfolded tensor (matrix).
    mode : int
        Mode (from unfolding).
    sizes : list
        Tensor dimension sizes.

    Returns
    -------
    NumPy array.
        Tensor
    """
    N = len(sizes)
    if mode:
        axis_order = list(range(N))
    else:
        return np.reshape(M, sizes)
    axis_order = [axis_order.pop(mode)] + axis_order
    moved_sizes = [sizes[a] for a in axis_order]
    return np.moveaxis(np.reshape(M, moved_sizes), np.arange(N), axis_order)


def modeprod(
    T: np.ndarray, M: np.ndarray, mode: int = 0, transpose: bool = False
) -> np.ndarray:
    """
    Tensor-matrix (single) mode product.

    Parameters
    ----------
    T : NumPy array
        Input tensor.
    M : NumPy array
        Input matrix.
    mode : int, optional
        Product mode. The default is 0.
    transpose : bool, optional
        Transpose input matrix. The default is False.

    Returns
    -------
    NumPy array.
        Tensor-matrix product.
    """
    size = np.array(T.shape)
    if transpose:
        M = M.T
    size[mode] = M.shape[0]
    return fold(M @ unfold(T, mode), mode, size)


def mmprod(
    T: np.ndarray,
    M: list,
    modes: tp.Union[list, None] = None,
    transpose: tp.Union[list, None] = None,
) -> np.ndarray:
    """
    Tensor-matrix multi-mode product.

    Parameters
    ----------
    T : NumPy array
        Input tensor.
    M : list
        List of input matrices.
    modes : list of ints, optional
        Modes. If not specified will default to ascending order.
    transpose : list of bools, optional
        Transpose input matrices. Defaults to False.

    Returns
    -------
    T : NumPy array
        Tensor-matrices product.
    """
    if not modes:
        modes = np.arange(len(M))
    if np.any(transpose):
        M = [m.T if t else m for m, t in zip(M, transpose)]
    for m, mode in zip(M, modes):
        T = modeprod(T, m, mode)
    return T


def eyeNR(N: int, R: int) -> np.ndarray:
    """
    Generate N-th order, rank R identity tensor.

    Parameters
    ----------
    N : int
        Order.
    R : int
        Rank.

    Returns
    -------
    NumPy array
        Identity tensor.
    """
    if N < 3:
        return np.eye(R)
    eye = np.zeros([R] * N, int)
    eye[tuple(np.arange(R) for n in range(N))] = 1
    return eye


def cpdgen(F: tp.Union[list, np.ndarray], opt: bool = False) -> np.ndarray:
    """
    Generate a tensor from list of factor matrices F.

    Parameters
    ----------
    F : list or NumPy array (of NumPy arrays)
        Factor matrices.
    opt : boolean, optional
        Attempt optimization, defaults to False
        (*possible* gain in speed at the cost of
        greater memory usage and slight loss of
        precision. Only works for N < 26. If your
        tensor has more than 25 dimensions this
        optimization is likely the least of
        your problems, though)

    Returns
    -------
    NumPy array
        Generated tensor.
    """
    R = F[0].shape[1]
    N = len(F)
    if N > 25:
        return mmprod(eyeNR(N, R), F)
    a = ord("a")  # 97
    seq_list = [chr(n) for n in np.arange(a, a + N)]
    subscripts = "z,".join(seq_list) + "z->" + "".join(seq_list)
    if opt:
        opt_path = np.einsum_path(subscripts, *F)[0]
        return np.einsum(subscripts, *F, optimize=opt_path)
    return np.einsum(subscripts, *F)


def eyeNL(N: int, L: list) -> np.ndarray:
    """
    Generate N-th order, rank-(L, L, 1) compressed identity tensor.

    Parameters
    ----------
    N : int
        Order.
    L : list
        Ranks.

    Returns
    -------
    NumPy array
        Compressed 'identity' tensor.
    """
    R = np.sum(L)
    INR = eyeNR(N, R)
    Lmod = [None, *L[:-1], None]
    proto = np.arange(R)
    slices = [proto[Lmod[i]: Lmod[i + 1]] for i in range(R - 1)]
    return np.array([INR[:, s].sum(axis=1) for s in slices]).transpose(
        [-1, *np.arange(R - 2, -1, -1)]
    )


def ll1gen(F: list, L: list) -> np.ndarray:
    """
    Generate a multilinear rank-(L, L, 1) tensor.

    Parameters
    ----------
    F : list (of NumPy arrays)
        Factor matrices.
    L : list
        Multilinear ranks.

    Returns
    -------
    NumPy array
        Generated tensor.
    """
    N = len(F)
    return mmprod(eyeNL(N, L), F)


def estSNR(X0: np.ndarray, X: np.ndarray) -> np.float64:
    """
    Estimates SNR (in dB) given noiseless X0 and noisy X.

    Parameters
    ----------
    X0 : NumPy array
        Noiseless tensor.
    X : NumPy array
        Noisy tensor.

    Returns
    -------
    float
        Estimated SNR.
    """
    return 20 * np.log10(la.norm(X0) / la.norm(X - X0))


def noisy(T: np.ndarray, SNR: float = 0.0) -> np.ndarray:
    """
    Add noise to a tensor of any order.

    Parameters
    ----------
    T : NumPy array
        Tensor.
    SNR : float, optional
        Signal-to-Noise Ratio. The default is 0.0.

    Returns
    -------
    NumPy array
        Noisy tensor.
    """
    size_tens = T.shape
    N = np.random.randn(*size_tens)
    if np.iscomplex(T).any():
        N = N + 1j * np.random.randn(*size_tens)
    scale = la.norm(T) * (10 ** (-SNR / 20)) / la.norm(N)
    N *= scale
    T = T + N
    return (T, N)


# %% Preprocessing


def lra(
    X: np.ndarray, R: tp.Union[int, list, np.ndarray], useTMLSVD: bool = False
) -> np.ndarray:
    """
    Low-Rank Approximation.

    Parameters
    ----------
    X : NumPy array
        Input tensor.
    R : int, list, or NumPy array
        Rank (int) or core size (list/array)
    useTMLSVD : bool, optional
        Use Truncated MLSVD instead of ST-MLSVD. The default is False.

    Returns
    -------
    Low-rank approximated tensor.
    """
    N = X.ndim

    if N == 2:
        U, s, VH = svds(X, R)
        return U @ np.diag(s) @ VH
    else:
        if type(R) is int:
            size_core = [R] * N
        else:
            size_core = R

        if useTMLSVD:
            U = tmlsvd(X, size_core)[0]
            S = estcore(X, U)
        else:
            U, S = stmlsvd(X, size_core)[:2]
        return mmprod(S, U)


def fba(X: np.ndarray, exp: bool = False) -> np.ndarray:
    """
    Forward-backward averaging.

    The last mode is assumed to be the temporal mode.

    Parameters
    ----------
    X : NumPy array
        Input tensor.
    exp : bool, optional, defaults to False
        Use expanded FBA.

    Returns
    -------
    NumPy array
        Forward-backward averaged tensor.
    """
    if X.ndim == 2:
        return np.hstack((X, np.rot90(X.conj(), 2)))
    else:
        PI = [np.fliplr(np.eye(s)) for s in X.shape]
        return np.concatenate((X, mmprod(X.conj(), PI)), X.ndim - 1)


def qunit(M: int) -> np.ndarray:
    """
    Generate unitary left-PI real matrices for unitary transformation.

    Parameters
    ----------
    M : int
        Number of array elements.

    Returns
    -------
    NumPy array
        Unitary transform matrix.
    """
    m = np.floor(M / 2).astype(int)
    n = np.mod(M, 2)
    Im = np.eye(m)
    PIm = np.fliplr(Im)

    Q = np.vstack((np.hstack((Im, np.zeros((m, n)), 1j * Im)),
                   np.hstack((np.zeros((n, m)),
                              np.sqrt(2) * np.ones((n, n)), np.zeros((n, m)))),
                   np.hstack((PIm, np.zeros((m, n)),
                              -1j * PIm)),)) / np.sqrt(2)
    return Q


def unitransf(X: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unitary transformation.

    Parameters
    ----------
    X : NumPy array
        Input tensor.

    Returns
    -------
    List of NumPy arrays
        Unitary transformed matrix.
        Forward-backward averaged matrix.
        Unitary transform residual matrix.
    """
    Z = fba(X)
    QH = [qunit(m).conj().T for m in Z.shape]
    phiZ = mmprod(Z, QH)
    resZ = np.imag(phiZ)
    phiZ = np.real(phiZ)

    return (phiZ, Z, resZ)


def cheapUT(X: np.ndarray) -> np.ndarray:
    """
    Cheap unitary transformation.

    Parameters
    ----------
    X : NumPy array
        Input tensor.

    Returns
    -------
    NumPy array
        Unitary transformed tensor.
    """
    R = X.ndim - 1
    if R == 1:
        Y = qunit(X.shape[0]).conj().T @ X
    else:
        QH = [qunit(m).conj().T for m in X.shape[:-1]]
        Y = mmprod(X, QH, modes=np.arange(R))
    return np.concatenate((Y.real, Y.imag), R)


def sps(X: np.ndarray, L: int = 2, expanded: bool = False) -> np.ndarray:
    """
    Spatial smoothing.

    Increases no. of samples using subarrays of the original data.

    Parameters
    ----------
    X : NumPy array
        Input tensor.
    L : int, optional
        Number of subarrays. The default is 2.
    expanded : bool, optional
        Expanded spatial smoothing. The default is False.

    Raises
    ------
    SystemExit
        If number of subarrays is equal to the number of elements.

    Returns
    -------
    NumPy array
        Spatially smoothed tensor.
    """
    R = X.ndim - 1
    M = np.array(X.shape[:R])

    if (L >= M).any():
        raise SystemExit("No. of subarrays cannot equal no. of elements")

    Msel = M - L + 1

    J = [[np.hstack((np.zeros((m, n)), np.eye(m), np.zeros((m, L - (n + 1)))))
          for m in Msel]
         for n in range(L)]

    if expanded:
        Y = np.stack(([mmprod(X, j) for j in J]), R + 1)
    else:
        Y = np.concatenate(([mmprod(X, j) for j in J]), R)
    return Y


# %% R-D array processing


def una(M: tp.Union[int, list, np.ndarray], mu: np.ndarray) -> np.ndarray:
    """
    Uniform N-dimensional Array.

    Generates left PI-real dimensionally separable array steering matrices
    given no. of elements in each dimension and spatial frequencies vector.

    Parameters
    ----------
    M : int, list, or NumPy array
        No. of array elements in each dimension.
    mu : NumPy array (vector)
        Spatial frequencies. Also defines model order.

    Returns
    -------
    NumPy array
        List of steering matrices (NumPy array).
    """
    if type(M) is int:
        deltaM = np.arange(1 - M, M, 2) / 2
        A = np.exp(1j * np.outer(deltaM, mu))
    else:
        deltaM = [np.arange(1 - m, m, 2) / 2 for m in M]
        A = [np.exp(1j * np.outer(d, m)) for d, m in zip(deltaM, mu)]
    return A


def uxa(
    M: tp.Union[int, list, np.ndarray], D: tp.Union[int, list, np.ndarray] = 1
) -> tp.Tuple[list, list]:
    """
    Uniform Linear/Rectangular/Cubic Array.

    Generates left PI-real dimensionally separable array
    steering matrices given no. of elements in each dimension,
    and model order or azimuth (and elevation, if applicable).

    Parameters
    ----------
    M : int, list, NumPy array
        No. of elements.
    D : int, list, NumPy array, optional
        Model order (int) or azimuth (and elevation).

    Returns
    -------
    List
        List of steering matrices and list of spatial frequencies.
    """
    try:
        R = len(M)
    except TypeError:
        R = 1

    if isinstance(D, int):
        if R == 1:
            az = np.pi * (np.random.rand(D) - 0.5)
        else:
            az = 2 * np.pi * np.random.rand(D)
            el = np.pi * (np.random.rand(D) - 0.5)
    else:
        if R == 1:
            az = D
        else:
            az = D[0]
            el = D[1]

    if R == 1:
        mu = np.pi * np.sin(az)
    else:
        mu = [np.pi * np.cos(az) * np.sin(el)]
        mu.append(np.pi * np.sin(az) * np.sin(el))
    if R == 3:
        mu.append(np.pi * np.sin(el))
    A = una(M, mu)

    return (A, mu)


def ste(X: np.ndarray, D: int, **varargin: bool) -> list:
    """
    Std. Tensor ESPRIT.

    Parameters
    ----------
    X : NumPy array
        Observation tensor.
    D : int
        Model order.
    **varargin : bool.
        useTMLSVD: Use TMLSVD.
        fullSVD: Use full matrices for MLSVD.

    Returns
    -------
    List
        Shift invariance eigenstructures.
    """
    trunc = bool(varargin.get("useTMLSVD"))
    usefull = bool(varargin.get("fullSVD"))

    R = X.ndim - 1
    M = np.asarray(X.shape[:R])
    P = [*np.minimum(M, D), D]

    # selection matrices
    Msel = M - 1

    J_0 = [np.hstack((np.eye(m), np.zeros((m, 1)))) for m in Msel]
    J_1 = [np.hstack((np.zeros((m, 1)), np.eye(m))) for m in Msel]

    I_left = [np.eye(np.prod(M[:r]).astype(int)) for r in range(R)]
    I_right = [np.eye(np.prod(M[r + 1: R]).astype(int)) for r in range(R)]

    J_0 = [kron([i_left, j_0, i_right])
           for i_left, j_0, i_right in zip(I_left, J_0, I_right)]
    J_1 = [kron([i_left, j_1, i_right])
           for i_left, j_1, i_right in zip(I_left, J_1, I_right)]

    # subspace estimation
    if not trunc:
        perm = np.argsort(X.shape)
        U, S = stmlsvd(X, P, perm, FullSVD=usefull)[:2]
        ISSE = unfold(mmprod(S, U[:R]), R).T
    else:
        if not usefull:
            Us, Sigmas = svds(
                unfold(X, R).T, D, return_singular_vectors="u", solver="lobpcg"
            )[:2]
        else:
            Us, Sigmas = la.svd(unfold(X, R).T, full_matrices=usefull)[:2]
            Us = Us[:, :D]
            Sigmas = Sigmas[:D]
        U = tmlsvd(X, P, FullSVD=usefull)[0]
        U_Kron = kron(U[:R])
        ISSE = U_Kron @ U_Kron.T.conj() @ Us @ np.diag(Sigmas)

    # shift invariance equations
    Psi = [la.pinv(j_0 @ ISSE) @ (j_1 @ ISSE) for j_0, j_1 in zip(J_0, J_1)]

    return Psi


def gevdpair(Psi: list, proj: bool = False) -> list:
    """
    GEVD-based spatial frequency pairing.

    Parameters
    ----------
    Psi : list
        Shift invariance eigenstructures.
    proj : bool, optional
        Use projection. The default is False.

    Returns
    -------
    list
        Shift-invariant matrices.
    """

    R = len(Psi)

    if R == 2:
        R = la.eig(Psi[0], Psi[1])[1]
        Phi = [la.inv(R) @ psi @ R for psi in Psi]
    elif proj:
        D = Psi[0].shape[0]
        T = np.stack(Psi, 2)
        P = la.svd(unfold(T, 2))[0][:, :2].conj().T
        T = fold(P @ unfold(T, 2), 2, (D, D, 2))
        R = la.eig(T[:, :, 0], T[:, :, 1])[1]
        Phi = [la.inv(R) @ psi @ R for psi in Psi]
    else:
        Phi = [gevdpair([Psi[0], Psi[r]]) for r in range(1, R)]
    return Phi


def ute(X: np.ndarray, D: int, **varargin: bool) -> list:
    """
    Unitary Tensor ESPRIT.

    Parameters
    ----------
    X : NumPy array
        Observation tensor.
    D : int
        Model order.
    **varargin : bool:
        useTMLSVD: Use TMLSVD.
        fullSVD: Use full matrices for MLSVD.
        performUT: perform unitary transform.
        cheapUT: perform cheap unitary transform.

    Returns
    -------
    list
        Shift invariance eigenstructures.

    """
    trunc = bool(varargin.get("useTMLSVD"))
    usefull = bool(varargin.get("fullSVD"))
    doUT = bool(varargin.get("performUT"))
    cheap = bool(varargin.get("cheapUT"))

    if np.iscomplex(X).any() ^ doUT:
        if cheap:
            X = cheapUT(X)
        else:
            X = unitransf(X)[0].real

    R = X.ndim - 1
    M = np.asarray(X.shape[:R])
    P = [*np.minimum(M, D), D]

    # selection matrix
    Msel = M - 1

    K = [qunit(m).conj().T @ np.hstack((np.zeros((m, 1)),
                                        np.eye(m))) @ qunit(m + 1)
         for m in Msel]

    I_left = [np.eye(np.prod(M[:r]).astype(int)) for r in range(R)]
    I_right = [np.eye(np.prod(M[r + 1: R]).astype(int)) for r in range(R)]

    K = [kron([i_left, k, i_right])
         for i_left, k, i_right in zip(I_left, K, I_right)]

    # subspace estimation
    if not trunc:
        perm = np.argsort(X.shape)
        U, S = stmlsvd(X, P, perm, FullSVD=usefull)[:2]
        ISSE = unfold(mmprod(S, U[:R]), R).T
    else:
        if not usefull:
            Us, Sigmas = svds(
                unfold(X, R).T, D, return_singular_vectors="u", solver="lobpcg"
            )[:2]
        else:
            Us, Sigmas = la.svd(unfold(X, R).T, full_matrices=usefull)[:2]
            Us = Us[:, :D]
            Sigmas = Sigmas[:D]
        U = tmlsvd(X, P, FullSVD=usefull)[0]
        U_Kron = kron(U[:R])
        ISSE = U_Kron @ U_Kron.T.conj() @ Us @ np.diag(Sigmas)

    # shift invariance equations
    Psi = [np.real(la.pinv(k.real @ ISSE) @ (k.imag @ ISSE)) for k in K]

    return Psi


def utepair(Psi: list) -> list:
    """
    UTE spatial frequency pairing.

    Parameters
    ----------
    Psi : list
        Shift invariance eigenstructure.

    Returns
    -------
    list
        Shift invariant matrices.

    """
    R = len(Psi)

    Psi_complex = Psi[0] + 1j * Psi[1]

    Phi_complex = la.eig(Psi_complex)[0]
    Phi = [Phi_complex.real]
    Phi.append(Phi_complex.imag)
    if R > 2:
        idx = Phi[0].argsort()
        Phi[0] = Phi[0][idx]
        Phi[1] = Phi[1][idx]
        for r in range(2, R):
            Psi_complex = Psi[0] + 1j * Psi[r]

            Phi_complex = la.eig(Psi_complex)[0]
            idx = Phi_complex.real.argsort()
            Phi.append(Phi_complex.imag[idx])

    return [np.diag(phi) for phi in Phi]


def sfcheck(mu: tp.Union[list, np.ndarray],
            mu_hat: tp.Union[list, np.ndarray]) -> list:
    """
    Spatial frequencies check.

    Unimplemented

    Parameters
    ----------
    mu : tp.Union[list, np.ndarray]
        True spatial frequencies.
    mu_hat : tp.Union[list, np.ndarray]
        Estimated spatial frequencies.

    Returns
    -------
    List
        Est. spatial frequencies, squared estimation error.

    """
    mu = np.array(mu)
    mu_hat = np.array(mu_hat)
    D = mu.shape[1]
    combs = list(permutations(range(D)))
    e = [la.norm(mu - mu_hat[:, comb], True) for comb in combs]
    idx = np.array(e).argmin()
    return mu_hat[:, combs[idx]], e[idx]


# %% Multilinear SVD


def stmlsvd(
    T: np.ndarray,
    size_core: tp.Union[None, int, list] = None,
    perm: tp.Union[None, list, np.ndarray] = None,
    **varargin: bool,
) -> tuple:
    """
    Sequentially truncated Multilinear SVD.

    Parameters
    ----------
    T : NumPy array
        Tensor.
    size_core : tp.Union[None, list, np.ndarray], optional
        Core size. The default is None.
    perm : tp.Union[None, list, np.ndarray], optional
        Permutation order. The default is None.
    **varargin : bool
        LargeScale: use eigendecomposition for SVD
        FullSVD: use full SVD
        Fast: only compute largest singular values (overrides FullSVD)

    Returns
    -------
    tuple
        Singular vector matrices, tensor core, singular values vector.

    """
    """Sequentially Truncated Multilinear SVD

    Usage:
        U, S, sv = stmlsvd(T)
        U, S, sv = stmlsvd(T, size_core)"""

    size_tens = list(T.shape)
    N = T.ndim

    if size_core is None:
        size_core = size_tens
    if isinstance(size_core, int):
        size_core = [size_core] * N

    if perm is None:
        perm = np.arange(N)

    large = bool(varargin.get("LargeScale"))
    usefull = bool(varargin.get("FullSVD"))
    fast = bool(varargin.get("Fast"))

    U = [None] * N
    sv = [None] * N

    S = T
    if large:
        for p in perm:
            Sp = unfold(S, p)
            SHS = Sp @ Sp.conj().T
            if usefull:
                ev, U[p] = la.eig(SHS)
                U[p] = U[p][:, : size_core[p]]
                sv[p] = np.sqrt(abs(ev[: size_core[p]]))
            else:
                ev, U[p] = la.eigs(SHS, size_core[p])
                sv[p] = np.sqrt(abs(ev))
            S = modeprod(S, U[p].conj().T, p)
    else:
        for p in perm:
            if fast:
                U[p], sv[p] = svds(unfold(S, p),
                                   k=size_core[p],
                                   solver="lobpcg")[:2]
            else:
                U[p], sv[p] = la.svd(
                    unfold(S, p), full_matrices=usefull, lapack_driver="gesvd"
                )[:2]
                U[p] = U[p][:, : size_core[p]]
                sv[p] = sv[p][: size_core[p]]
            S = modeprod(S, U[p].conj().T, p)
    return (U, S, sv)


def tmlsvd(
    T: np.ndarray,
    size_core: tp.Union[None, list, np.ndarray] = None,
    **varargin: bool
) -> tuple:
    """
    Truncated multilinear SVD.

    Parameters
    ----------
    T : NumPy array
        Tensor.
    size_core : tp.Union[None, list, np.ndarray], optional
        Core . The default is None.
    **varargin : bool
        LargeScale: use eigendecomposition for SVD.
        FullSVD: use full SVD.

    Returns
    -------
    tuple
        Singular vector matrices, singular values

    """
    size_tens = list(T.shape)
    N = T.ndim

    if size_core is None:
        size_core = size_tens

    large = bool(varargin.get("LargeScale"))
    usefull = bool(varargin.get("FullSVD"))

    if large:
        Sp = [unfold(T, n) for n in range(N)]
        SHS = [s @ s.T.conj() for s in Sp]
        if usefull:
            ev, U = zip(*[la.eig(s) for s in SHS])
            sv = [np.sqrt(abs(e[:size])) for e, size in zip(ev, size_core)]
            U = [u[:, :size] for u, size in zip(U, size_core)]
        else:
            ev, U = zip(*[eigs(s) for s in SHS])
            sv = [np.sqrt(abs(ev))]
    else:
        if usefull:
            U, sv = zip(
                *[
                    la.svd(unfold(T, n),
                           full_matrices=usefull,
                           lapack_driver="gesvd")[:2]
                    for n in range(N)
                ]
            )
            U = [u[:, :size] for u, size in zip(U, size_core)]
            sv = [s[:size] for s, size in zip(sv, size_core)]
        else:
            U, sv = zip(
                *[
                    svds(unfold(T, n), size, solver="lobcpg")[:2]
                    for n, size in zip(range(N), size_core)
                ]
            )
    return (U, sv)


def estcore(T: np.ndarray,
            U: list, perm: tp.Union[None, list] = None) -> np.ndarray:
    """
    Core tensor estimation.

    Parameters
    ----------
    T : NumPy array
        Tensor.
    U : tp.Union[list, np.ndarray]
        Singular value matrices.
    perm : tp.Union[None, list, np.ndarray], optional
        Permutation order. The default is None.

    Returns
    -------
    S : NumPy array
        Core tensor.
    """
    if perm is None:
        S = mmprod(T, [u.conj().T for u in U])
    else:
        if type(U) is not np.ndarray:
            U = np.array(U)
        S = mmprod(T, [u.conj().T for u in U[perm]], perm)
    return S


# %% CP Decomposition


def ampcpd(Y: np.ndarray, F: list, normcols: bool = False) -> np.ndarray:
    """
    Calculate CPD component amplitudes.

    Parameters
    ----------
    Y : NumPy array
        Tensor.
    F : list
        Factor matrices.
    normcols : bool, optional
        Normalize columns. The default is False.

    Returns
    -------
    NumPy array
        CPD component amplitudes.

    """
    if normcols:
        F = [colnorm(f) for f in F]
    R = F[0].shape[1]
    Fcols = [[f[:, r] for f in F] for r in range(R)]
    G = [np.einsum("i,j,k->ijk", *f) for f in Fcols]
    return np.array([(Y.ravel() / g.ravel()).sum() for g in G])


def cpdgevd(T: np.ndarray,
            R: int,
            normcols: bool = False,
            fast: bool = False) -> list:
    """
    Canonical Polyadic Decomposition via GEVD (CPD-GEVD).

    Decomposes a third-order tensor into its
    canonical polyadic components via generalized
    eigenvalue decomposition.

    Parameters
    ----------
    T : NumPy array
        Tensor.
    R : int
        Rank.
    normcols : bool, optional
        Normalize columns. The default is False.

    Returns
    -------
    list
        Estimated factor matrices.

    """
    U, S = stmlsvd(T, (R, R, np.max((R, 2))), Fast=fast)[:2]
    R = la.eig(S[:, :, 0].T, S[:, :, 1].T)[1]

    F23 = unfold(T, 0).T @ U[0].conj() @ R
    F = [U[0] @ la.inv(R.T)] + lskrf(F23, T.shape[1])

    if normcols:  # normalize columns
        F_col_norm = [np.sqrt((f * f.conj()).sum(0)) for f in F]
        F = [f @ np.diag(1 / f_col_norm)
             for f, f_col_norm in zip(F, F_col_norm)]
        allcolnorm = np.vstack(F_col_norm).prod(0) ** (1 / 3)
        F = [f @ np.diag(allcolnorm) for f in F]
    return F


def cpdgevd2(
    T: np.ndarray,
    R: int,
    normcols: bool = False,
    fast: bool = False,
    thirdonly: bool = False,
) -> tp.Tuple[np.array, np.array, np.array]:
    """
    Canonical Polyadic Decomposition via symmetric GEVD.

    Parameters
    ----------
    T : NumPy array
        Tensor.
    R : int
        Rank.
    normcols : bool, optional
        Normalize columns. The default is False.
    fast : bool, optional
        Something fast, I think.
    thirdonly : bool, optional
        Return only third factor matrix. The default is False.

    Returns
    -------
    list (of NumPy arrays)
        Estimated factor matrices.

    """
    U, S = stmlsvd(T, (R, R, np.max((R, 2))), Fast=fast)[:2]
    L, R = la.eig(S[:, :, 0], S[:, :, 1], left=True)[1:3]

    T = [None] * 3
    T[2] = np.einsum("ijk->kj", mmprod(S, [L.T.conj(), R.T]))
    if thirdonly:
        return U[2] @ T[2]
    T[:2] = [la.inv(lr) for lr in (L.T.conj(), R.T)]
    F = [u @ t for u, t in zip(U, T)]
    if normcols:  # normalize columns
        F_col_norm = [np.sqrt((f * f.conj()).sum(0)) for f in F]
        F = [f @ np.diag(1 / f_col_norm)
             for f, f_col_norm in zip(F, F_col_norm)]
        allcolnorm = np.vstack(F_col_norm).prod(0) ** (1 / 3)
        F = [f @ np.diag(allcolnorm) for f in F]
    return F


def cpdsevd(
    T: np.ndarray, R: int, normcols: bool = False, fast: bool = False
) -> tp.Tuple[np.array, np.array, np.array]:
    """
    Canonical polyadic decomposition via Singular
        and Eigenvalue decomposition (CPD-S/EVD)
    Don't use this, it's probably very imprecise

    Parameters
    ----------
    T : NumPy array
        Tensor.
    R : int
        Rank.
    normcols : bool, optional
        Normalize columns. The default is False.

    Returns
    -------
    list (of NumPy arrays)
        Estimated factor matrices.
    """
    U, S = stmlsvd(T, (R, R, 2), Fast=fast)[:2]
    S_0 = S[:, :, 0]
    S_v = np.vstack((S_0, S[:, :, 1]))
    U_v = la.svd(S_v, full_matrices=False)[0]
    U_1 = U_v[:R, :]
    U_2 = U_v[R: (2 * R), :]

    R_1 = U_1.conj().T @ U_1
    R_2 = U_1.conj().T @ U_2

    d, E = la.eig(R_2 @ la.inv(R_1))

    T_temp = la.inv(R_1) @ E

    T = [U_1 @ T_temp]
    T.append((la.pinv(T[0]) @ S_0).T)
    T.append(unfold(S, 2) @ (la.pinv(kr(T))).T)

    F = [U[n] @ T[n] for n in range(len(U))]
    if normcols:  # normalize columns
        F_col_norm = [np.sqrt((f * f.conj()).sum(0)) for f in F]
        F = [f @ np.diag(1 / f_col_norm)
             for f, f_col_norm in zip(F, F_col_norm)]
        allcolnorm = np.vstack(F_col_norm).prod(0) ** (1 / 3)
        F = [f @ np.diag(allcolnorm) for f in F]
    return F


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
        N = 2
    else:
        N = len(M)

    T = [np.reshape(K.T[r], M) for r in range(R)]
    u, s = zip(*[stmlsvd(t, 1)[:2] for t in T])
    S = np.diagflat(s).astype(complex) ** (1 / N)
    return [np.squeeze(a).T @ S for a in list(zip(*u))]
