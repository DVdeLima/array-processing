# %% Docstring

"""
Multilinear Algebra.

--------------------

Includes:
    1. Basic tensor operations
        Column normalization (colnorm)
        Kronecker product (kron)
        Khatri-Rao product (kr)
        Kronecker to KR prod. sel. matrix (kron2kr)
        Unfold tensor (unfold)
        Fold tensor (fold)
        Tensor-matrix product (tmprod)
        Tensor inner product (tip)
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
        Spatial frequency calculator (sf_calc)
        Generate lin./rect./par. steering matrices (uxa)
        Maximum overlap selection matrices (mosm)
        Standard Tensor ESPRIT (ste)
        GEVD-based spatial freq. pairing (gevdpair)
        Unitary Tensor ESPRIT (ute)
        UTE-based spatial freq. pairing (utepair)
    4. Multilinear SVD (MLSVD)
        Seq. Trunc. MLSVD (stmlsvd)
        Trunc. MLSVD (tmlsvd)
        Est. core tensor (estcore)
    5. Alternating Least Squares (ALS)
        ALS update (als_upd)
        ALS (als)
    6. Canonical Polyadic Decomposition (CPD)
        CPD via GEVD (cpdgevd)
        CPD via symmetric GEVD2
        CPD via SVD/EVD (cpdsevd)
        LS Khatri-Rao factorization (lskrf)
"""

# %% __all__

__all__ = [
    "colnorm",
    "outer_prod",
    "kron",
    "kr",
    "kron2kr",
    "unfold",
    "fold",
    "tmprod",
    "tip",
    "eyeNR",
    "cpdgen",
    "eyeNL",
    "ll1gen",
    "estSNR",
    "noisy",
    "lmlra",
    "fba",
    "qunit",
    "unitransf",
    "cheapUT",
    "una",
    "sf_calc",
    "uxa",
    "mosm",
    "ste",
    "gevd_pair",
    "ute",
    "ute_pair",
    "sf_check",
    "estDOAs",
    "stmlsvd",
    "tmlsvd",
    "estcore",
    "als_upd",
    "als",
    "cpdgevd",
    "cpdgevd2",
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
        return X / la.norm(X)
    elif N == 2:
        return X @ np.diagflat(1 / la.norm(X, axis=0))
    else:
        return fold(colnorm(unfold(X, N-1)), N-1, [*X.shape])


def outer_prod(F: list) -> np.ndarray:
    """
    Outer product.

    Parameters
    ----------
    F : list
        List of vectors or matrices or tensors.

    Returns
    -------
    NumPy array
        Outer product.
    """
    N = [f.ndim for f in F]
    Nacc = [0] + [sum(N[:n]) for n in range(1, len(F) + 1)]
    modelist = list(range(Nacc[-1]))
    modes = [modelist[Nacc[m]: Nacc[m+1]] for m in range(len(F))]
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


def kron2kr(C: int) -> np.array:
    """
    Kronecker to Khatri-Rao product selection matrix.

    Parameters
    ----------
    C : int
        No. of columns of Khatri-Rao product.

    Returns
    -------
    NumPy array
        Kronecker to Khatri-Rao product selection matrix.

    """
    K = np.zeros((C ** 2, C))
    K[(range(0, C ** 2, C + 1), range(C))] = 1
    return K


def unfold(T: np.ndarray,
           modes_left: tp.Union[int, list] = 0,
           modes_right: tp.Optional[list] = None):
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


def tip(tensors: tp.List[np.ndarray],
        modes: tp.List[int] = None) -> np.ndarray:
    """
    Tensor inner product.

    Parameters
    ----------
    tensors : List of NumPy arrays
        Input tensors.
    modes : List of ints, optional
        List of inner product modes. The default is None.
        If not set defaults to the last mode for all tensors.

    Returns
    -------
    NumPy array
        Inner product tensor.

    """
    ndims = [tensor.ndim for tensor in tensors]
    mode_op_indices = [[sum(ndims[:end]), sum(ndims[:end+1])]
                       for end in range(len(ndims))]
    mode_ops = [list(range(*indices)) for indices in mode_op_indices]
    if modes is None:
        modes = [n - 1 for n in ndims]
    input_ops_list = [[tensor, mode_op]
                      for tensor, mode_op in zip(tensors, mode_ops)]
    input_ops = [x for op in input_ops_list for x in op]
    del input_ops_list
    output_ops_list = [mode_op[:mode] + mode_op[mode+1:]
                       for mode_op, mode in zip(mode_ops, modes)]
    output_ops = [x for op in output_ops_list for x in op]
    return np.einsum(*(input_ops + [output_ops]))


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
    if R == 1:
        return 1
    if N < 3:
        return np.eye(R, dtype=int)
    eye = np.zeros([R] * N, int)
    eye[tuple(np.arange(R) for _ in range(N))] = 1
    return eye


def hyperdiag(vec: np.array, N: int = 3) -> np.array:
    """
    Hyperdiagonal

    Parameters
    ----------
    vec : NumPy array (vector)
        Input vector.
    N : int, optional
        Number of dimensions. The default is 3.

    Returns
    -------
    NumPy array
        Hyperdiagonal tensor.
    """
    R = len(vec)
    H = np.zeros([vec.shape[0]] * N, dtype=complex)
    H[tuple(np.arange(R) for _ in range(N))] = vec
    return H


def cpdgen(F: list, opt: bool = False) -> np.ndarray:
    """
    Generate tensor from list of factor matrices F.

    Parameters
    ----------
    F : list
        Factor matrices.
    opt : bool, optional
        Attempt path optimization. The default is False.
        *Possible* gain in speed at the cost of greater
        memory usage and slight loss of precision.

    Returns
    -------
    NumPy array
        Generator tensor.
    """
    if np.all([f.ndim == 2 for f in F]):
        N = len(F)
        modes = [[n, N] for n in range(N)]
        operands = [comb for pair in zip(F, modes) for comb in pair]
        return np.einsum(*operands, range(N))
    return tmprod(F[0], F[1:])


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
    return 20 * np.log10(la.norm(X0) / la.norm(X - X0))


def noise(T: np.ndarray,
          SNR: float = 0.0) -> np.ndarray:
    """
    Generate AWGN noise tensor.

    Parameters
    ----------
    T : NumPy array
        Input tensor.
    SNR : float, optional
        Signal to noise ratio. The default is 0.0.

    Returns
    -------
    N : NumPy array
        Noise tensor.

    """
    tensor_shape = T.shape
    if np.iscomplex(T).any():
        N = np.random.randn(*tensor_shape) + \
            1j * np.random.randn(*tensor_shape) / np.sqrt(2)
    else:
        N = np.random.randn(*tensor_shape)
    scale = la.norm(T) * (10 ** (-SNR / 20)) / la.norm(N)
    N *= scale
    return N


def truncate(M: np.ndarray,
             R: tp.Union[int, list],
             axis: int = 1) -> np.ndarray:
    """
    Truncate a matrix or vector.

    Parameters
    ----------
    M : NumPy array
        Tensor.
    R : int
        Rank.
    axis : int, optional
        Axis for trunction. The default is 1 (columnwise).

    Returns
    -------
    NumPy array
        Truncated matrix or vector.

    """
    N = M.ndim
    if isinstance(R, list):
        return M[:R[0], :R[1]]
    if axis and N == 2:
        return M[:, :R]
    return M[:R]


# %% Preprocessing

def lmlra(T: np.ndarray, R: int | list) -> np.ndarray:
    U, S = stmlsvd(T, R)[:2]
    return tmprod(S, U)


def fba(T: np.ndarray, mode: int = -1) -> np.ndarray:
    """
    Foward-backward averaging

    Parameters
    ----------
    T : NumPy array
        Input tensor.
    mode : int, optional
        FB-average "direction." The default is the last mode.
        Will stack onto a new dimension if the specified mode
        is equal to or greater than T.ndim

    Returns
    -------
    NumPy array
        FB-averaged tensor.

    """
    N = T.ndim
    slices = [slice(None, None, -1)] * N
    if mode >= N:
        return np.einsum("i... -> ...i", np.stack((T, T.conj()[*slices])))
    return np.concatenate((T, T.conj()[*slices]), axis=mode)


def sps(T: np.ndarray, L: int | list = None) -> np.ndarray:
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
    M = list(T.shape)
    N = T.ndim
    if L is None:
        L = [2] * N
    T_sps = T.copy()
    for n, ell in enumerate(L):
        if ell > 1:
            Ms = M[n] - ell + 1
            M[n] = Ms
            M[-1] = ell * M[-1]
            row_slices = [slice(eel, Ms + eel) for eel in range(ell)]
            T_sps = fold(np.hstack([unfold(T_sps, n)[row_slice]
                                    for row_slice
                                    in row_slices]), n, M)
    return T_sps


def desps(T_sps: np.ndarray, L: int | list = None) -> np.ndarray:
    """
    Recover sample tensor from spatially smoothed samples

    Parameters
    ----------
    T : NumPy array
        Spatially smoothed tensor.
    L : int, list, optional
        Number of subarrays. The default is 2.

    Returns
    -------
    NumPy array
        Sample matrix.

    """
    N = T_sps.ndim
    # modes = np.arange(N - 2, -1, -1)
    N = T_sps.shape[1] // L
    col_slices = [slice(ell * N, (ell + 1) * N) for ell in range(1, L)]
    return np.vstack((T_sps[:, :N], *[T_sps[-1, col_slice][None, :]
                                      for col_slice in col_slices]))


def MuDe(X: np.ndarray, D: int, L: int | list = 2) -> np.ndarray:
    """
    Multiple Denoising

    Parameters
    ----------
    X : NumPy array
        Input tensor.
    D : int
        Model order.
    L : int | list, optional
        No of subarrays. The default is 2.

    Returns
    -------
    X_MuDe : NumPy array
        Multiply denoised.

    """
    X_MuDe = lmlra(X, D)
    if L == 1:
        return X_MuDe
    for ell in L:
        X_sps = sps(X_MuDe, ell)
        X_lra = lmlra(X_sps)
        X_MuDe = desps(X_lra, ell)
    return X_MuDe


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
    m = M // 2
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


def unitransf(X: np.ndarray, mode: int = -1) -> tp.Tuple[np.ndarray,
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
    Y = fba(X, mode)
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


# %% R-D array processing


def una(M: int | list | np.ndarray,
        mu: float | list | np.ndarray) -> np.ndarray | list:
    """
    Uniform N-dimensional Array.

    Generates left PI-real dimensionally separable array steering matrices
    given no. of elements in each dimension and spatial frequencies vector.

    Parameters
    ----------
    M : int, list, or NumPy array
        No. of array elements in each dimension.
    mu : float, list, or NumPy array
        Spatial frequencies. Also defines model order.

    Returns
    -------
    NumPy array
        List of steering matrices (NumPy array).
    """
    if isinstance(M, int):
        deltaM = np.array([m/2 for m in range(1 - M, M, 2)])
        return np.exp(1j * np.outer(deltaM, mu))
    return [una(m, moo) for m, moo in zip(M, mu.T)]


def sf_calc(R: int,
            D: int | float | tuple | list | np.ndarray) -> float | np.ndarray:
    """
    Spatial Frequency calculator.

    Parameters
    ----------
    R : int
        Number of array dimensions.
    D : int | float | tuple | list | NumPy array
        Model order or DoA(s).

    Raises
    ------
    TypeError
        If R or D not of an accepted type.
    NotImplementedError
        If R > 3.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if R > 3:
        raise NotImplementedError("No implementation for R > 3")
    if isinstance(D, (tuple, list, np.ndarray)):
        if R == 1:
            azimuth = D
        else:
            if isinstance(D[0], float):
                azimuth = D
                elevation = np.pi / 2 * np.ones(len(D))
            else:
                azimuth, elevation = zip(*D)
    elif isinstance(D, float):
        if R == 1:
            azimuth = D
        else:
            azimuth = D
            elevation = np.pi / 2
    else:
        raise TypeError("Invalid input type: model order/DoAs")
    match R:
        case 1:
            mu = np.sin(azimuth)
        case 2:
            mu = np.einsum(np.array((np.cos(azimuth),
                                     np.sin(azimuth))), [1, 0],
                           np.sin(elevation), [0], [0, 1])
        case 3:
            mu = np.hstack((np.einsum(np.array((np.cos(azimuth),
                                                np.sin(azimuth))), [1, 0],
                                      np.sin(elevation), [0], [0, 1]),
                            np.cos(elevation)[:, None]))
        case _:
            raise NotImplementedError("No implementation for R > 3")
    return np.pi * mu


def uxa(M: int | list,
        D: int | float | list) -> (list, list):
    """
    Uniform Linear/Rectangular/Cuboid

    Parameters
    ----------
    M : int | list
        DESCRIPTION.
    D : int | float | list
        DESCRIPTION.

    Raises
    ------
    TypeError
        DESCRIPTION.
    NotImplementedError
        DESCRIPTION.

    Returns
    -------
    (list, list)
        DESCRIPTION.

    """
    R = len(M)
    if isinstance(D, int):
        if R == 1:
            azimuth = np.pi * (np.random.rand(D) - 0.5)
            mu = sf_calc(R, azimuth)
        else:
            azimuth = 2 * np.pi * (np.random.rand(D) - 0.5)
            elevation = np.pi * np.random.rand(D)
            mu = sf_calc(R, np.array([azimuth, elevation]).T)
    else:
        mu = sf_calc(R, D)
    return una(M, mu), mu


def mosm(M: int | list) -> np.ndarray | list:
    """
    Maximum overlap selection matrix

    Parameters
    ----------
    M : int or list
        No. of array elements.

    Returns
    -------
    NumPy array or list of NumPy array
        Max. overlap selection matrices.

    """
    if isinstance(M, int):
        return [np.vstack((np.eye(M - 1), np.zeros(M - 1))).T,
                np.vstack((np.zeros(M - 1), (np.eye(M - 1)))).T]
    J = [mosm(m) for m in M]
    eyes = [np.eye(m) for m in M]
    return [[kron(eyes[:n] + [jay] + eyes[n+1:])
             for jay in j]
            for n, j in enumerate(J)]


def ste(X: np.ndarray, D: int,
        matrix_sse: bool = True,
        kron_sse: bool = False,
        J: list = None) -> list:
    """
    Standard Tensor ESPRIT

    Parameters
    ----------
    X : NumPy array
        Data tensor.
    D : int
        Model order.
    matrix_sse : bool, optional
        Matrix-based subspace estimate. The default is True.
    kron_sse : bool, optional
        Kronecker projection-based subspace estimate. The default is False.
    J : list
        Selection matrix list. The default is None.

    Returns
    -------
    list
        Shift-invariant eigenstructures (Psi, unpaired).

    """
    R = X.ndim - 1
    M = X.shape[:R]
    P = [*np.minimum(M, D), D]

    if J is None:
        J = mosm(M)

    # subspace estimation
    if matrix_sse:
        Us, sigmas = [m.T[:D].T for m
                      in la.svd(unfold(X, R).T, full_matrices=False)[:2]]
        Sigmas = np.diag(sigmas)
        ISSE = Us @ Sigmas
        if kron_sse:
            U = [la.svd(unfold(X, r), full_matrices=False)[0][:, :p]
                 for r, p in zip(range(R), P[:R])]
            U_Kron = kron(U)
            ISSE = U_Kron @ U_Kron.T.conj() @ ISSE
    else:
        if kron_sse:
            U = [la.svd(unfold(X, r), full_matrices=False)[0][:, :p]
                 for r, p in zip(range(R), P[:R])]
            U_Kron = kron(U)
            U_KR = kr(U)
            ISSE = (U_Kron @ U_Kron.T.conj()) @ U_KR
        else:
            U, S = stmlsvd(X, P)[:2]
            ISSE = unfold(tmprod(S, U[:R]), R).T
    return [la.pinv(j[0] @ ISSE) @ (j[1] @ ISSE) for j in J]


def gevd_pair(Psi: tp.List[np.ndarray]) -> tp.List[np.ndarray]:
    """
    GEVD-based pairing.

    Parameters
    ----------
    Psi : list of NumPy arrays
        List of shift-invariant eigenstructures.

    Returns
    -------
    list of NumPy arrays (vectors)
        Paired spatial frequencies.

    """
    R = la.eig(*Psi[::-1])[1]
    return [np.angle(np.diag(la.inv(R) @ psi @ R)) for psi in Psi]


def qzpair(Psi: list) -> list:
    Phi = la.qz(*Psi)[0]
    return np.angle(Phi[1] / Phi[0])


def sfPhi(Phi: list) -> tp.List[np.ndarray]:
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
        cheap: bool = True) -> tp.List[np.ndarray]:
    """
    Unitary Tensor ESPRIT

    Parameters
    ----------
    X : NumPy array
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
                      for m in la.svd(unfold(X, R).T,
                                      full_matrices=False)[:2]]
        Sigmas = np.diag(sigmas)
        if matrix_sse:
            ISSE = Us @ Sigmas
        else:
            U = [truncate(m, D)
                 for m in [la.svd(unfold(X, r))[0]
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


def ute_pair(Psi: list) -> tp.List[np.ndarray]:
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


def sf_check(mu: tp.Union[list, np.ndarray],
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
    e = [la.norm(mu - mu_hat[:, perm]) for perm in perms]
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
            **varargin: bool) -> tp.List[np.ndarray]:
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

    Returns
    -------
    tuple
        Singular vector matrices, tensor core, singular values vectors.
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
                            for m in la.eig(SHS)]
                sv[p] = np.sqrt(abs(ev))
            S = tmprod(S, U[p].conj().T, p)
    else:
        for p in perm:
            U[p], sv[p] = la.svd(unfold(S, p),
                                 full_matrices=usefull)[:2]
            U[p] = U[p][:, :size_core[p]]
            sv[p] = sv[p][: size_core[p]]
            S = tmprod(S, U[p].conj().T, p)
    return (U, S, sv)


def tmlsvd(T: np.ndarray,
           size_core: tp.Union[None, list, np.ndarray] = None,
           **varargin: bool) -> tp.List[np.ndarray]:
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
    if isinstance(size_core, int):
        size_core = [size_core] * N

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
            ev, U = zip*([la.eig(s) for s in SHS])
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
            U, sv = zip(*[la.svd(unfold(T, n),
                                 full_matrices=False)[:2]
                          for n, size in zip(range(N), size_core)])
    return U, sv


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


# %% Alternating Least Squares


def als_upd(T: np.ndarray, F: list) -> tp.List[np.ndarray]:
    """
    Alternating Least Squares update

    Parameters
    ----------
    T : NumPy array
        Input tensor.
    F : list
        Estimated factor matrices.

    Returns
    -------
    list
        Updated estimated factor matrices.

    """
    for dim in range(T.ndim):
        F[dim] = unfold(T, dim) @ la.pinv(kr(F[:dim] + F[(dim + 1):])).T
    return F


def als(T: np.ndarray, R: int,
        maxit: int = 1000,
        tol: float = 1e-3,
        F: list = None) -> tp.Tuple[list, int, float]:
    """
    Alternating Least Squares PARAFAC decomposition

    Parameters
    ----------
    T : NumPy array
        Input tensor.
    R : int
        Rank.
    maxit : int, optional
        Maximum no. of iterations. The default is 1000.
    tol : float, optional
        Reconstruction error tolarance. The default is 1e-3.
    F : list, optional
        Factor matrix initialization overrides. The default is None.

    Returns
    -------
    list of NumPy arrays
        Estimated factor matrices.

    """
    reconst_error = 1.0
    iteration = 0
    if F is None:
        F = [np.random.randn(m, R) for m in T.shape]
    while (reconst_error > tol) and (iteration < maxit):
        iteration += 1
        F = als_upd(T, F)
        reconst_error = la.norm(T - cpdgen(F))
    return F, iteration, reconst_error


# %% Canonical Polyadic Decomposition


def ampcpd(Y: np.ndarray, F: list,
           normcols: bool = False) -> np.ndarray:
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


def cpdgevd(T: np.ndarray, R: int,
            normcols: bool = False) -> tp.List[np.array]:
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


def cpdgevd2(T: np.ndarray, D: int,
             normcols: bool = False,
             thirdonly: bool = False) -> tp.List[np.array]:
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
    thirdonly : bool, optional
        Return only third factor matrix. The default is False.

    Returns
    -------
    list (of NumPy arrays)
        Estimated factor matrices.

    """
    U, S = stmlsvd(T, (D, D, np.max((D, 2))))[:2]
    L, R = la.eig(S[:, :, 0], S[:, :, 1], left=True)[1:]

    if thirdonly:
        return U[2] @ tmprod(S, [L.T.conj(), R.T]).diagonal()
    T = [None] * 3
    T[:2] = [la.inv(lr) for lr in (L.T.conj(), R.T)]
    T[2] = tmprod(S, [L.T.conj(), R.T]).diagonal()
    F = [u @ t for u, t in zip(U, T)]
    if normcols:  # normalize columns
        F = [colnorm(f) for f in F]
    return F


def pigevd2(T: np.ndarray, R: int) -> list:
    U, S = stmlsvd(T, (R, R, np.max((R, 2))))[:2]
    L, R = la.eig(S[:, :, 0], S[:, :, 1], left=True)[1:]
    return (U[0] @ L).T.conj(), (U[1].conj() @ R).T


def lskrf(K: np.ndarray,
          M: tp.Union[int, list, np.ndarray]) -> tp.List[np.ndarray]:
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
    N = len(M)

    T = [np.reshape(K.T[r], M) for r in range(R)]
    u, s = zip(*[stmlsvd(t, 1)[:2] for t in T])
    S = np.diagflat(s).astype(complex) ** (1 / N)
    return [np.squeeze(a).T @ S for a in list(zip(*u))]


def htkrf(K: np.ndarray,
          M: tp.Union[int, list, np.ndarray]) -> tp.List[np.ndarray]:
    """
    Hierarchical Tensor Khatri-Rao factorization

    Parameters
    ----------
    K : NumPy array
        Khatri-Rao product.
    M : tp.Union[int, list, np.ndarray]
        Size of first dimension or list
        of sizes (of first dimension).

    Returns
    -------
    list (of NumPy arrays)
        Factor matrices.
    """
    N = len(M)
    F = []
    C = K.shape[1]
    for n in range(N - 1):
        T = fold(K, [[0, 1], 2], [M[n], np.prod(M[(n+1):]), C])
        U, S = stmlsvd(T, C, np.argsort(T.shape)[::-1])[:2]
        R, L = la.eig(S[:, :, 0], S[:, :, 1], left=True)[1:]
        F.append(np.einsum(tmprod(T, R @ U[1].T.conj(), 1), [0, 1, 2], [0, 2]))
        K = np.einsum(tmprod(T, L @ U[0].T.conj(), 0), [0, 1, 2], [1, 2])
    F.append(K)
    return F
