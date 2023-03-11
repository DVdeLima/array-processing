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
        Tensor-matrix product (tmprod)
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
    "tmprod",
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
    "estDOAs",
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

from itertools import permutations

# %% Debug modules

# import pdb as ipdb
# ipdb.set_trace()

# %% Basic operations


def frob(X: np.ndarray,
         squared: bool = False,
         axis: tp.Union[int, None] = None) -> float:
    """
    Frobenius norm.

    Parameters
    ----------
    X : NumPy array
        Vector or matrix or tensor.
    squared : bool, optional
        Return squared Frobenius norm. The default is False.
    axis : tp.Union[int, None], optional
        Axis along which the norm is calculated. The default is None.

    Returns
    -------
    float
        Frobenius norm.

    """
    if isinstance(axis, int):
        return [frob(x, squared) for x in unfold(X, axis)]
    frob2 = (X * X.conj()).real.sum()  # squared Frob. norm
    if squared:
        return frob2
    return np.sqrt(frob2)


def colnorm(X: np.ndarray) -> np.ndarray:
    """
    Column Normalization.

    Parameters
    ----------
    X : NumPy array
        Input matrix or tensor. In the case of
        a tensor, normalization occurs along the
        fibers of the last mode of the tensor.

    Returns
    -------
    NumPy array
        Input matrix with normalized columns.
        The squared Frobenius norm is (approx)
        the order of the tensor.
    """
    N = X.ndim
    if N == 1:
        return X / frob(X)
    elif N == 2:
        return X @ np.diag(1 / np.array(frob(X, axis=1)))
    else:
        D = np.diag(1 / np.array(frob(X, axis=(N - 1))))
        return tmprod(X, D, N - 1)


def oprod(F: list) -> np.ndarray:
    """
    Outer product.

    Parameters
    ----------
    F : list
        List of vectors or matrices or tensors.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    N = [f.ndim for f in F]
    Nacc = [0] + [sum(N[:n]) for n in range(1, len(F) + 1)]
    modes = list(range(Nacc[-1]))
    modes = [modes[Nacc[m]: Nacc[m+1]] for m in range(len(F))]
    operands = [comb for pair in zip(F, modes) for comb in pair]
    return np.einsum(*operands, range(sum(N)))


def kron(F: list) -> np.ndarray:
    """
    Kronecker product.

    Parameters
    ----------
    F : list
        List of factor matrices.

    Returns
    -------
    NumPy array
        Kronecker product.

    """
    N = len(F)
    F = [np.matrix(f).T if f.ndim == 1 else f for f in F]
    kron_shape = np.prod([f.shape for f in F], axis=0)
    modes = [[m, n] for m, n in zip(range(N), range(N, 2*N))]
    operands = [comb for pair in zip(F, modes) for comb in pair]
    if kron_shape[1] == 1:
        return np.einsum(*operands, range(2*N)).ravel()
    return np.einsum(*operands, range(2*N)).reshape(kron_shape)


def kr(F: list) -> np.ndarray:
    """
    Khatri-Rao product.

    Parameters
    ----------
    F : list
        List of factor matrices.

    Raises
    ------
    Exception
        If number of columns isn't the same for all matrices.

    Returns
    -------
    NumPy array
        Khatri-Rao product.

    """
    N = len(F)
    if np.all([f.ndim == 1 for f in F]):
        modes = [[n] for n in range(N)]
        operands = [comb for pair in zip(F, modes) for comb in pair]
        return np.einsum(*operands).ravel()
    rows, cols = zip(*[f.shape for f in F])
    if not np.all([c == cols[0] for c in cols]):
        error_msg = "All factor matrice must have the same number of columns!"
        raise Exception(error_msg)
    kr_shape = [np.prod(rows), cols[0]]
    modes = [[n, N] for n in range(N)]
    operands = [comb for pair in zip(F, modes) for comb in pair]
    return np.einsum(*operands, range(N+1)).reshape(kr_shape)


def unfold(T: np.ndarray,
           modes_left: tp.Union[int, list] = 0,
           modes_right: tp.Union[list, None] = None):
    """
    Returns mode-unfolding of tensor.

    Parameters
    ----------
    T : NumPy array
        Tensor.
    mode : int, list, optional
        Unfolding mode. The default is 0 (first mode).

    Returns
    -------
    NumPy array
        mode-unfolded tensor.
    """
    modes_left_is_int = isinstance(modes_left, (int, np.integer))
    modes_right_is_none = modes_right is None
    if modes_left_is_int and modes_right_is_none:
        tensor_shape = list(T.shape)
        matrix_shape = [tensor_shape.pop(modes_left),
                        np.array(tensor_shape).prod()]
        if modes_left:
            N = T.ndim
            axis_order = list(range(N))
            axis_order = [axis_order.pop(modes_left), *axis_order]
            T = np.einsum(T, range(N), axis_order)
        return np.reshape(T, matrix_shape)
    else:  # arbitrary unfolding, undocumented
        N = T.ndim
        if modes_left_is_int:
            modes_left = [modes_left]
        if modes_right_is_none:
            modes_right = [n for n in range(N) if n not in modes_left]
        elif isinstance(modes_right, int):
            modes_right = [modes_right]
        tensor_shape = list(T.shape)
        matrix_shape = [np.prod([tensor_shape[m] for m in modes_left]),
                        np.prod([tensor_shape[m] for m in modes_right])]
        return np.einsum(T, range(N),
                         modes_left + modes_right).reshape(matrix_shape)


def fold(M: np.ndarray,
         modes: tp.Union[int, list],
         tensor_shape: list) -> np.ndarray:
    """
    Fold unfolded tensor back into tensor.

    Parameters
    ----------
    M : NumPy array
        Unfolded tensor (matrix).
    mode : int
        Mode (from unfolding).
    shape : list
        Tensor shape (size of each dimension).

    Returns
    -------
    NumPy array.
        Tensor
    """
    N = len(tensor_shape)
    if isinstance(modes, int):
        if modes:
            axis_order = list(range(N))
        else:
            return np.reshape(M, tensor_shape)
        axis_order = [axis_order.pop(modes)] + axis_order
        moved_shape = [tensor_shape[ax] for ax in axis_order]
        return np.einsum(np.reshape(M, moved_shape), axis_order, range(N))
    else:  # arbitrary mode folding, undocumented
        all_modes = range(N)
        if np.all([isinstance(m, int) for m in modes]):
            modes_left = modes
            modes_right = [n for n in all_modes if n not in modes_left]
        elif np.any(intput := [isinstance(m, int) for m in modes]):
            modes_left, modes_right = [[m] if iu else m
                                       for m, iu in zip(modes, intput)]
        axis_order = modes_left + modes_right
        moved_shape = [tensor_shape[ax] for ax in axis_order]
        return np.einsum(M.reshape(moved_shape), axis_order, all_modes)


def tmprod(T: np.ndarray,
           M: tp.Union[np.ndarray, list],
           modes: tp.Union[int, list, None] = None) -> np.ndarray:
    """
    Tensor-matrix product

    Parameters
    ----------
    T : NumPy array
        Input tensor.
    M : NumPy array or list of NumPy arrays
        Input matrices.
    modes : int or list of ints, optional
        Modes. The default is None.

    Returns
    -------
    NumPy array
        Tensor-matrix product.

    """
    N = T.ndim
    if modes is None:
        if isinstance(M, list):
            modes = range(len(M))
        else:
            modes = 0
    if isinstance(M, np.ndarray):
        M = [M]
    if isinstance(modes, (int, np.integer)):
        modes = [modes]
    no_modes = len(modes)
    new_shapes = range(N, N + no_modes)
    modict = {m: n for (m, n) in zip(modes, new_shapes)}
    output = [n if n not in modes else modict.get(n) for n in range(N)]
    modes = [[n, m] for m, n in zip(modes, range(N, N + no_modes))]
    operands = [comb for pair in zip(M, modes) for comb in pair]
    return np.einsum(T, range(N), *operands, output)


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


def cpdgen(F: list, opt: bool = False) -> np.ndarray:
    """
    Generate tensor from list of factor matrices F.

    Parameters
    ----------
    F : list
        DESCRIPTION.
    opt : bool, optional
        Attempt path optimization. The default is False.
        *Possible* gain in speed at the cost of greater
        memory usage and slight loss of precision.

    Returns
    -------
    NumPy array
        Generator tensor.

    """
    N = len(F)
    modes = [[n, N] for n in range(N)]
    operands = operands = [comb for pair in zip(F, modes) for comb in pair]
    return np.einsum(*operands, range(N))


def eyeNL(N: int, L: list) -> np.ndarray:
    """
    Generate N-th order, rank-(L_1, ..., L_P) compressed identity tensor.

    Parameters
    ----------
    N : int
        Order.
    L : list of ranks
        list of rank of each factor matrix.

    Returns
    -------
    NumPy array
        Compressed 'identity' tensor.
    """
    P = len(L)
    R = sum(L)
    INR = eyeNR(N, R)
    Lacc = [0] + [sum(L[:p+1]) for p in range(P)]
    IL = [np.sum(INR[Lacc[p]:Lacc[p+1]], axis=0) for p in range(P)]
    return np.stack(IL, axis=N-1)


def ll1gen(F: list,
           L: list) -> np.ndarray:
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
    return tmprod(eyeNL(N, L), F)


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
    return 20 * np.log10(frob(X0) / frob(X - X0))


def noisy(T: np.ndarray,
          SNR: float = 0.0) -> tp.Tuple[np.ndarray,
                                        np.ndarray]:
    """
    Generate AWGN noise for a tensor.

    Parameters
    ----------
    T : NumPy array
        Input tensor.
    SNR : float, optional
        Signal-to-Noise Ratio. The default is 0.0.

    Returns
    -------
    T : NumPy array
        Noisy tensor.
    N : NumPy array
        Noise tensor.
    """
    tensor_shape = T.shape
    if np.iscomplex(T).any():
        N = np.random.randn(*tensor_shape) + \
            1j * np.random.randn(*tensor_shape)
    else:
        N = np.random.randn(*tensor_shape)
    scale = frob(T) * (10 ** (-SNR / 20)) / frob(N)
    N *= scale
    T = T + N
    return (T, N)


def truncate(T: np.ndarray,
             R: int,
             axis: int = 1) -> np.ndarray:
    N = T.ndim
    if N == 1:
        return T[:R]
    elif N == 2:
        if axis:
            return T.T[:R].T
        else:
            return T[:R]

# %% Preprocessing


def lra(X: np.ndarray,
        R: tp.Union[int, list, None] = None,
        useTMLSVD: bool = False) -> np.ndarray:
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
        U, s, VH = np.linalg.svd(X, full_matrices=False)
        if isinstance(R, (int, np.integer)):
            U = U[:, :R]
            S = np.diagonal(s[:R])
            VH = VH[:R]
        elif isinstance(R, list):
            U = U[:, :R[0]]
            S = np.diagonal(s[:min(R)])
            VH = VH[:R[1]]
        return U @ S @ VH
    else:
        if R is None:
            size_core = None
        elif isinstance(R, (int, np.integer)):
            size_core = [np.min((R, m)) for m in X.shape]
        else:
            size_core = R

        if useTMLSVD:
            U = tmlsvd(X, size_core)[0]
            S = estcore(X, U)
        else:
            U, S = stmlsvd(X, size_core)[:2]
        return tmprod(S, U)


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
    N = X.ndim
    if N == 2:
        return np.hstack((X, np.fliplr(np.flipud(X.conj()))))
    else:
        PI = [np.fliplr(np.eye(s)) for s in X.shape]
        return np.concatenate((X, tmprod(X.conj(), PI)), N - 1)


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

    Q = np.vstack((np.hstack((Im,
                              np.zeros((m, n)),
                              1j * Im)),
                   np.hstack((np.zeros((n, m)),
                              np.sqrt(2) * np.ones((n, n)),
                              np.zeros((n, m)))),
                   np.hstack((PIm,
                              np.zeros((m, n)),
                              -1j * PIm)),)) / np.sqrt(2)
    return Q


def unitransf(X: np.ndarray) -> tp.Tuple[np.ndarray,
                                         np.ndarray]:
    """
    Unitary transformation.

    Parameters
    ----------
    X : NumPy array
        Input tensor.

    Returns
    -------
    Tuple of NumPy arrays
        Unitary transformed matrix (complex)
        Forward-backward averaged matrix.
        Unitary transform residual matrix.
    """
    Y = fba(X)
    QH = [qunit(m).conj().T for m in Y.shape]
    Z = tmprod(Y, QH)

    return (Z, QH)


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
        Y = tmprod(X, QH, modes=np.arange(R))
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

    J = [[np.hstack((np.zeros((m, n)),
                     np.eye(m),
                     np.zeros((m, L - (n + 1)))))
          for m in Msel] for n in range(L)]

    if expanded:
        Y = np.stack(([tmprod(X, j) for j in J]), R + 1)
    else:
        Y = np.concatenate(([tmprod(X, j) for j in J]), R)
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


def uxa(M: tp.Union[int, list],
        D: tp.Union[int, list] = 1) -> tp.Tuple[list, list]:
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
    if isinstance(M, int):
        R = 1
    else:
        R = len(M)

    if isinstance(D, int):
        if R == 1:
            az = np.pi * (np.random.rand(D) - 0.5)
        else:
            az = 2 * np.pi * (np.random.rand(D) - 0.5)
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
        mu.append(np.pi * np.cos(el))
    A = una(M, mu)

    return (A, mu)


def ste(X: np.ndarray,
        D: int,
        matrix_sse: bool = False,
        kron_sse: bool = False) -> list:
    """
    Standard Tensor ESPRIT

    Parameters
    ----------
    X : NumPy array
        Input tensor.
    D : int
        Model order.
    matrix_sse : bool, optional
        Use matrix-based signal subspace est.
        The default is False.
    kron_sse : bool, optional
        Use Kronecker structured projections to
        estimate subspace. The default is False.

    Returns
    -------
    list
        Shift invariance eigenstructures (unpaired).
    """
    R = X.ndim - 1
    M = np.asarray(X.shape[:R])
    P = [*np.minimum(M, D), D]

    # selection matrices
    Msel = M - 1

    J_0 = [np.hstack((np.eye(m),
                      np.zeros((m, 1))))
           for m in Msel]
    J_1 = [np.hstack((np.zeros((m, 1)),
                      np.eye(m)))
           for m in Msel]

    I_left = [np.eye(np.prod(M[:r]))
              for r in range(R)]
    I_right = [np.eye(np.prod(M[r + 1: R]))
               for r in range(R)]

    J_0 = [kron([i_left, j_0, i_right])
           for i_left, j_0, i_right
           in zip(I_left, J_0, I_right)]
    J_1 = [kron([i_left, j_1, i_right])
           for i_left, j_1, i_right
           in zip(I_left, J_1, I_right)]
    # subpace estimation
    if (kron_sse ^ matrix_sse):
        Us, sigmas = [truncate(m, D)
                      for m in np.linalg.svd(unfold(X, R).T,
                                             full_matrices=False)[:2]]
        Sigmas = np.diag(sigmas)
        if matrix_sse:
            ISSE = Us @ Sigmas
        else:
            U = [truncate(m, D)
                 for m in [np.linalg.svd(unfold(X, r))[0]
                           for r in range(R)]]
            U_Kron = kron(U)
            ISSE = U_Kron @ U_Kron.T.conj() @ (Us @ Sigmas)
    else:
        perm = np.flip(np.argsort(X.shape))
        U, S = stmlsvd(X, P, perm)[:2]
        ISSE = unfold(tmprod(S, U[:R]), R).T

    # shift invariance equations
    Psi = [la.pinv(j_0 @ ISSE) @ (j_1 @ ISSE)
           for j_0, j_1 in zip(J_0, J_1)]
    return Psi


def gevdpair(Psi: list) -> list:
    """
    GEVD-based shift-invariant eigenstructure pairing.

    Parameters
    ----------
    Psi : list
        Shift invariance eigenstructures.

    Returns
    -------
    list
        Shift-invariant matrices.
    """
    R = len(Psi)
    Rev = la.eig(*Psi[:2])[1]
    Phi = [Rev.T.conj() @ psi @ Rev for psi in Psi[:2]]
    if R > 2:
        idx = np.angle(Phi[0].diagonal()).argsort()
        Phi = [phi[idx] for phi in Phi]
        for r in range(2, R):
            pair = [Psi[0], Psi[r]]
            Rev = la.eig(*pair)[1]
            newPhi = [Rev.T.conj() @ psi @ Rev
                      for psi in pair]
            idx = np.angle(newPhi[0].diagonal()).argsort()
            Phi.append(newPhi[1][idx])
    return Phi


def sfPhi(Phi: list) -> list:
    """
    Calculate spatial frequencies from
    Shift invariance eigenstructures

    Parameters
    ----------
    Phi : list
        List of Shift invariance eigenstructures (paired).

    Returns
    -------
    list
        Spatial frequencies (paired).
    """
    if isinstance(Phi[0], np.ndarray):
        return [np.angle(np.diag(phi)) for phi in Phi]
    else:
        mus = [[np.angle(np.diag(p))
                for p in Phi[n]] for n in range(len(Phi))]
        idxs = [np.argsort(m[0]) for m in mus]
    if np.all([idxs[0] == ids for ids in idxs[1:]]):
        return mus[0] + [m[1] for m in mus[1:]]
    else:
        return [m[idxs[0]]
                for m in mus[0]] + [m[1][idx]
                                    for m, idx in zip(mus[1:], idxs[1:])]


def ute(X: np.ndarray,
        D: int,
        matrix_sse: bool = False,
        kron_sse: bool = False,
        cheap: bool = True) -> list:
    """
    Unitary Tensor ESPRIT

    Parameters
    ----------
    X : np.ndarray
        Input tensor.
    D : int
        Model order.
    matrix_sse : bool, optional
        Use matrix-based signal subspace estimation.
        The default is False.
    kron_sse : bool, optional
        Use structured Kronecker projection to
        estimate signal subspace. The default is False.
    cheap : bool, optional
        Use cheap unitary transform. The default is True.

    Returns
    -------
    list
        Shift invariance eigenstructures (unpaired).
    """
    if np.iscomplex(X).any():
        if cheap:
            X = cheapUT(X)
        else:
            X = unitransf(X)[0].real

    R = X.ndim - 1
    M = np.asarray(X.shape[:R])
    P = [*np.minimum(M, D), D]

    # selection matrix
    Msel = M - 1

    J_1 = [np.hstack((np.zeros((m, 1)),
                      np.eye(m)))
           for m in Msel]
    K = [qunit(m).conj().T @ j1 @ qunit(m + 1)
         for m, j1 in zip(Msel, J_1)]

    I_left = [np.eye(np.prod(M[:r])) for r in range(R)]
    I_right = [np.eye(np.prod(M[r + 1: R])) for r in range(R)]

    K = [kron([i_left, k, i_right])
         for i_left, k, i_right
         in zip(I_left, K, I_right)]

    if (kron_sse ^ matrix_sse):
        Us, sigmas = [truncate(m, D)
                      for m in np.linalg.svd(unfold(X, R).T,
                                             full_matrices=False)[:2]]
        Sigmas = np.diag(sigmas)
        if matrix_sse:
            ISSE = Us @ Sigmas
        else:
            U = [truncate(m, D)
                 for m in [np.linalg.svd(unfold(X, r))[0]
                           for r in range(R)]]
            U_Kron = kron(U)
            ISSE = U_Kron @ U_Kron.T.conj() @ (Us @ Sigmas)
    else:
        perm = np.flip(np.argsort(X.shape))
        U, S = stmlsvd(X, P, perm)[:2]
        ISSE = unfold(tmprod(S, U[:R]), R).T

    # shift invariance equations
    Psi = [np.real(la.pinv(k.real @ ISSE) @ (k.imag @ ISSE))
           for k in K]

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
    Phi = [Phi_complex.real, Phi_complex.imag]
    if R > 2:
        idx = Phi[0].argsort()
        Phi = [phi[idx] for phi in Phi]
        for r in range(2, R):
            Psi_complex = Psi[0] + 1j * Psi[r]

            Phi_complex = la.eig(Psi_complex)[0]
            idx = Phi_complex.real.argsort()
            Phi.append(Phi_complex.imag[idx])

    return [np.diag(phi) for phi in Phi]


def sfcheck(mu: tp.Union[list, np.ndarray],
            mu_hat: tp.Union[list, np.ndarray]) -> tp.Tuple[np.ndarray,
                                                            float]:
    """
    Check and sort estimated spatial frequencies.

    Parameters
    ----------
    mu : tp.Union[list, np.ndarray]
        True spacial frequencies.
    mu_hat : tp.Union[list, np.ndarray]
        Estimated spacial frequencies.

    Returns
    -------
    NumPy array
        Matrix of sorted estimated spatial frequencies.
    float
        Error (Euclidian distance).
    """
    if isinstance(mu, (list, tuple)):
        mu = np.stack(mu)
    D = mu.shape[0]
    if isinstance(mu_hat, (list, tuple)):
        mu_hat = np.stack(mu_hat)
    perms = list(permutations(range(D)))
    e = [frob(mu - mu_hat[:, perm]) for perm in perms]
    idx = np.argmin(e)
    return mu_hat[:, perms[idx]], e[idx]


def estDOAs(mu: tp.Union[list, np.ndarray]) -> np.ndarray:
    if isinstance(mu, list):
        mu = np.stack(mu)
    if mu.ndim == 1:
        return np.arcsin(mu / np.pi)
    else:
        az = np.arctan2(mu[1], mu[0])
        el = np.arcsin(mu[1] / (np.pi * np.sin(az)))
    return np.stack([az, el])


# %% Multilinear SVD

def stmlsvd(T: np.ndarray,
            size_core: tp.Union[int, list, None] = None,
            perm: tp.Union[list, np.ndarray, None] = None,
            **varargin: bool) -> tp.Tuple[np.ndarray,
                                          np.ndarray,
                                          np.ndarray]:
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
    tensor_shape = list(T.shape)
    N = T.ndim

    if size_core is None:
        size_core = tensor_shape
    if isinstance(size_core, int):
        size_core = [size_core] * N

    if perm is None:
        perm = range(N)

    large = bool(varargin.get("LargeScale"))
    usefull = bool(varargin.get("FullSVD"))

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
                ev, U[p] = [truncate(m, size_core[p])
                            for m in np.linalg.eig(SHS)]
                sv[p] = np.sqrt(abs(ev))
            S = tmprod(S, U[p].conj().T, p)
    else:
        for p in perm:
            U[p], sv[p] = np.linalg.svd(unfold(S, p),
                                        full_matrices=usefull)[:2]
            U[p] = U[p][:, :size_core[p]]
            sv[p] = sv[p][: size_core[p]]
            S = tmprod(S, U[p].conj().T, p)
    return (U, S, sv)


def tmlsvd(T: np.ndarray,
           size_core: tp.Union[None, list, np.ndarray] = None,
           **varargin: bool) -> tp.Tuple[np.ndarray,
                                         np.ndarray,
                                         np.ndarray]:
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
    tensor_shape = list(T.shape)
    N = T.ndim

    if size_core is None:
        size_core = tensor_shape

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
            ev, U = zip*([np.linalg.eig(s) for s in SHS])
            ev = [truncate(e) for e in ev]
            U = [truncate(u) for u in U]
            sv = [np.sqrt(abs(ev))]
    else:
        if usefull:
            U, sv = zip(*[la.svd(unfold(T, n),
                                 full_matrices=usefull,
                                 lapack_driver="gesvd")[:2]
                          for n in range(N)])
            U = [u[:, :size] for u, size in zip(U, size_core)]
            sv = [s[:size] for s, size in zip(sv, size_core)]
        else:
            U, sv = zip(*[np.linalg.svd(unfold(T, n),
                                        full_matrices=False)[:2]
                          for n, size in zip(range(N), size_core)])
    return (U, sv)


def estcore(T: np.ndarray,
            U: list,
            perm: tp.Union[None, list] = None) -> np.ndarray:
    """
    Core tensor estimation.

    Parameters
    ----------
    T : NumPy array
        Tensor.
    U : list
        Singular value matrices.
    perm : list, optional
        Permutation order. The default is None.

    Returns
    -------
    S : NumPy array
        Core tensor.
    """
    if perm is None:
        S = tmprod(T, [u.conj().T for u in U])
    else:
        if type(U) is not np.ndarray:
            U = np.array(U)
        S = tmprod(T, [u.conj().T for u in U[perm]], perm)
    return S


# %% Canonical Polyadic Decomposition


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
            normcols: bool = False) -> tp.Tuple[np.array,
                                                np.array,
                                                np.array]:
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
    U, S = stmlsvd(T, (R, R, np.max((R, 2))))[:2]
    R = la.eig(S[:, :, 0].T, S[:, :, 1].T)[1]

    F23 = unfold(T, 0).T @ U[0].conj() @ R
    F = [U[0] @ la.inv(R.T)] + lskrf(F23, T.shape[1])

    if normcols:  # normalize columns
        F = [colnorm(f) for f in F]
    return F


def cpdgevd2(
    T: np.ndarray,
    R: int,
    normcols: bool = False,
    thirdonly: bool = False) -> tp.Tuple[np.array,
                                         np.array,
                                         np.array]:
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
    U, S = stmlsvd(T, (R, R, np.max((R, 2))))[:2]
    L, R = la.eig(S[:, :, 0], S[:, :, 1], left=True)[1:]

    T = [None] * 3
    T[2] = tmprod(S, [L.T.conj(), R.T]).diagonal()
    if thirdonly:
        return U[2] @ T[2]
    T[:2] = [la.inv(lr) for lr in (L.T.conj(), R.T)]
    F = [u @ t for u, t in zip(U, T)]
    if normcols:  # normalize columns
        F = [colnorm(f) for f in F]
    return F


def cpdsevd(T: np.ndarray,
            R: int,
            normcols: bool = False) -> tp.Tuple[np.array,
                                                np.array,
                                                np.array]:
    """
    Canonical polyadic decomposition via Singular
    and Eigenvalue decomposition (CPD-S/EVD).

    Don't use this, probably very imprecise in most cases.

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
    U, S = stmlsvd(T, (R, R, 2))[:2]
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
        F = [colnorm(f) for f in F]
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
