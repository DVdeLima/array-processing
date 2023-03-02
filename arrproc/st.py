# %% Docstring

"""
Subpace Tracking.

--------------------

Includes:
    1. Projection-based
        Projection Approximation Subspace Tracking (PASTupd)
        PAST via deflation (PASTdupd)
        Orthogonal PAST (OPASTupd)
        Fast Constrained PAST (FCPASTupd)
        Generalized YAST (GYASTupd)
    2. Power iteration-based
        Fast Approximate Power Iteration (FAPIupd)
    3. Raileigh quotient-based
        Fast Rayleigh quotient Adaptive Noise Subspace (FRANSupd)
        Fast Data Projection Method (FDPMupd)
"""

# %% Load dependencies

import numpy as np
import scipy.linalg as la
import typing as tp

from scipy.sparse.linalg import eigs
from arrproc.asp import colnorm


# %% Auxiliary functions


def Tri(A: np.ndarray) -> np.ndarray:
    """
    Triangular conditioning.

    Parameters
    ----------
    A : np.ndarray
        Input (Hermitian) matrix.

    Returns
    -------
    NumPy array
        Reconditioned matrix.

    """
    return (A + A.conj().T) / 2


def SVDupd(
    X: np.ndarray, U: np.ndarray, evs: np.ndarray, ff: float = 0.97
) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    SVD subspace update.

    Parameters
    ----------
    X : NumPy array
        Observation vector/matrix.
    U : np.ndarray
        Current subspace estimate.
    evs : NumPy array (vector)
        Current eigenvalues.
    ff : float, optional
        Forgetting factor. The default is .97.

    Returns
    -------
    NumPy array
        Updated subspace.
    NumPy array (vector)
        Updated eigenvalues.

    """
    C = U @ np.diagflat(evs) @ U.conj().T
    if X.ndim == 1:
        C = ff * C + np.outer(X, X.conj())
    else:
        C = ff * C + X @ X.conj().T
    evs, U = eigs(C, len(evs))
    return (U, evs)


# %% PAST-based


def PASTupd(
    X: np.ndarray, U: np.ndarray, C: np.ndarray, ff: float = 0.97
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Projection Approximation Subspace Tracking update.

    Usage:
        U, C = PASTupd(X, U, C, ff)

    Parameters
    ----------
    X : np.ndarray
        Observation vector/matrix.
    U : np.ndarray
        Current subspace estimate.
    C : np.ndarray
        Current covariance matrix.
    ff : float, optional
        Forgetting factor. The default is 0.97.

    Returns
    -------
    NumPy array
        Updated subspace estimate
    NumPy array
        Updated covariance matrix estimate

    """
    if X.ndim == 1:
        y = U.conj().T @ X
        e = X - U @ y
        h = C @ y
        g = h / (ff + y.conj().T @ h)
        U = U + np.outer(e, g.conj())
        C = Tri(C - np.outer(g, h.conj()))
    else:
        Y = U.conj().T @ X
        E = X - U @ Y
        C = ff * C + Y @ Y.conj().T
        G = la.inv(C) @ Y
        U = U + E @ G.conj().T
    return (U, C)


def BPASTupd(
    X: np.ndarray, U: np.ndarray, C: np.ndarray, ff: float = 0.97
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Batch PAST update.

    Usage:
        U, C = PASTupd(X, U, C, ff)

    Parameters
    ----------
    X : np.ndarray
        Observation vector/matrix.
    U : np.ndarray
        Current subspace estimate.
    C : np.ndarray
        Current covariance matrix.
    ff : float, optional
        Forgetting factor. The default is 0.97.

    Returns
    -------
    NumPy array
        Updated subspace estimate
    NumPy array
        Updated covariance matrix estimate
    """
    U, C = PASTupd(X.T[0], U, C, ff)
    for n in range(1, X.shape[1]):
        U, C = FAPIupd(X.T[n], U, C, 1.0)
    return (U, C)


def PASTdupd(
    X: np.ndarray, U: np.ndarray, evs: np.ndarray, ff: float = 0.97
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Deflationary PAST update.

    Usage:
        U, evs = PASTdupd(x, U, evs, ff)

    Parameters
    ----------
    x : NumPy array
        Observation vector.
    U : np.ndarray
        Current subspace estimate.
    evs : np.ndarray
        Estimated eigenvalues.
    ff : float, optional
        Forgetting factor. The default is 0.97.

    Returns
    -------
    NumPy array
        Updated subspace estimate.
    NumPy array
        Updated covariance matrix estimate.

    """
    for d in range(U.shape[1]):
        u = U.T[d]
        y = X.T @ u.conj()
        evs[d] = ff * evs[d] + (la.norm(y) ** 2)
        if X.ndim == 1:
            E = X - u * y
            U[:, d] = u + E * y.conj() / evs[d]
            X = X - U[:, d] * y
        else:
            E = X - np.outer(u, y)
            U[:, d] = u + E @ y.conj() / evs[d]
            X = X - np.outer(U[:, d], y)
    return (U, evs)


def OPASTupd(
    x: np.ndarray, U: np.ndarray, C: np.ndarray, ff: float = 0.97
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Orthogonal PAST.

    Parameters
    ----------
    x : np.ndarray
        Input vector/matrix.
    U : np.ndarray
        Current subspace estimate.
    C : np.ndarray
        Current covariance matrix.
    ff : float, optional
        Forgetting factor. The default is 0.97.

    Returns
    -------
    NumPy array
        Updated subspace estimate.
    NumPy array
        Updated covariance matrix estimate.

    """
    y = U.conj().T @ x
    q = C @ y / ff
    q2norm2 = la.norm(q) ** 2
    gamma = 1 / (1 + y.conj().T @ q)
    tau = (1 / np.sqrt(1 + q2norm2 * (la.norm(x) ** 2 - la.norm(y) ** 2)) - 1) / q2norm2
    p = (
        U @ (tau * q - gamma * (1 + tau * q2norm2) * y)
        + (1 + tau * q2norm2) * gamma * x
    )
    U += np.outer(p, q.conj())
    C = C / ff - gamma * np.outer(q, q.conj())
    return (U, C)


def BOPASTupd(
    x: np.ndarray, U: np.ndarray, C: np.ndarray, ff: float = 0.97
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Orthogonal PAST.

    Parameters
    ----------
    x : np.ndarray
        Input vector/matrix.
    U : np.ndarray
        Current subspace estimate.
    C : np.ndarray
        Current covariance matrix.
    ff : float, optional
        Forgetting factor. The default is 0.97.

    Returns
    -------
    NumPy array
        Updated subspace estimate.
    NumPy array
        Updated covariance matrix estimate.

    """
    U, C = OPASTupd(x.T[0], U, C, ff)
    for n in range(1, x.shape[1]):
        U, C = OPASTupd(x.T[n], U, C, 1.0)
    return (U, C)


def FCPASTupd(
    X: np.ndarray, U: np.ndarray, C: np.ndarray, ff: float = 0.97
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Fast Constrained PAST.

    Parameters
    ----------
    X : np.ndarray
        DESCRIPTION.
    U : np.ndarray
        DESCRIPTION.
    C : np.ndarray
        DESCRIPTION.
    ff : float, optional
        DESCRIPTION. The default is 0.97.

    Returns
    -------
    None.

    """
    if X.ndim == 1:
        y = U.conj().T @ X
        f = C.conj().T @ y
        D = np.outer(y, f.conj().T)
        D /= ff + np.trace(D)
        C = C / ff @ (np.eye(U.shape[1]) - D)
        U = U - U @ D + np.outer(X, y.conj().T @ C)
    else:
        Y = U.conj().T @ X
        F = C.conj().T @ Y
        D = Y @ F.conj().T
        D /= ff + np.trace(D)
        C = C / ff @ (np.eye(U.shape[1]) - D)
        U = U - U @ D + X @ (Y.conj().T @ C)
    return (U, C)


def BFCPASTupd(
    X: np.ndarray, U: np.ndarray, C: np.ndarray, ff: float = 0.97
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Batch Fast Constrained PAST.

    Parameters
    ----------
    X : np.ndarray
        DESCRIPTION.
    U : np.ndarray
        DESCRIPTION.
    C : np.ndarray
        DESCRIPTION.
    ff : float, optional
        DESCRIPTION. The default is 0.97.

    Returns
    -------
    None.

    """
    U, C = FCPASTupd(X.T[0], U, C, ff)
    for n in range(1, X.shape[1]):
        U, C = FCPASTupd(X.T[n], U, C, 1.0)
    return (U, C)


def GYASTupd(
    x: np.ndarray, U: np.ndarray, P: np.ndarray, sigma_n: float, ff: float = 0.97
) -> tp.Tuple[np.ndarray, np.ndarray, float]:
    """
    Generalized YAST.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION.
    U : np.ndarray
        DESCRIPTION.
    P : np.ndarray
        DESCRIPTION.
    sigma_n : float
        DESCRIPTION.
    ff : float
        DESCRIPTION.

    Returns
    -------
    None.

    """
    D = U.shape[1]

    y = U.conj().T @ x
    P1 = ff * P + np.outer(y, y.conj())
    sigma = np.sqrt((x.conj().T @ x - y.conj().T @ y).real)
    z = sigma * y
    gamma = (ff * sigma_n**2 + sigma**2).real
    P2 = np.row_stack((np.column_stack((P1, z)), np.append(z.conj(), gamma)))
    e, Q = la.eig(P2)
    q_n = Q[:, e.real.argmin()]
    q = q_n[:D]
    r = q_n[-1]
    ell = 1 / (1 + abs(r))
    k = r / (abs(r) * sigma)
    e1 = U @ (ell * q - k * y) + k * x
    U = U - np.outer(e1, q.conj())
    q1 = P1 @ q
    C = np.outer((r / abs(r)) * z + ell * q1, q.conj())
    c = (
        ell**2 * q.conj().T @ q1 + gamma + 2 * ell / abs(r) * (r * q.conj().T @ z)
    ).real
    P = P1 + c * np.outer(q, q.conj()) - C - C.conj().T
    lambda_m = np.trace(P2) - np.trace(P)
    sigma_n = np.sqrt(min((lambda_m, sigma_n**2)))
    return U, P, sigma_n


def BGYASTupd(
    x: np.ndarray, U: np.ndarray, P: np.ndarray, sigma_n: float, ff: float = 0.97
) -> tp.Tuple[np.ndarray, np.ndarray, float]:
    """
    Batch GYAST.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION.
    U : np.ndarray
        DESCRIPTION.
    P : np.ndarray
        DESCRIPTION.
    sigma_n : float
        DESCRIPTION.
    ff : float
        DESCRIPTION.

    Returns
    -------
    None.

    """
    U, P, sigma_n = GYASTupd(x.T[0], U, P, sigma_n, ff)
    for n in range(1, x.shape[1]):
        U, P, sigma_n = GYASTupd(x.T[n], U, P, sigma_n, 1.0)
    return U, P, sigma_n


# %% Power Iteration-based


def FAPIupd(
    x: np.ndarray, U: np.ndarray, C: np.ndarray, ff: float = 0.97
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Fast Approximate Power Iteration.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION.
    U : np.ndarray
        DESCRIPTION.
    P : np.ndarray
        DESCRIPTION.
    ff : float, optional
        DESCRIPTION. The default is 0.97.

    Returns
    -------
    None.

    """
    y = U.conj().T @ x
    h = C @ y
    g = h / (ff + y.conj().T @ h)

    g2norm2 = la.norm(g) ** 2
    epsilon2 = la.norm(x) ** 2 - la.norm(y) ** 2
    tau = epsilon2 / (1 + epsilon2 * g2norm2 + np.sqrt(1 + epsilon2 * g2norm2))
    eta = 1 - tau * g2norm2

    y_prime = eta * y + tau * g
    h_prime = C.conj().T @ y_prime
    epsilon = (tau / eta) * (C @ g - (h_prime.conj().T @ g) * g)
    C = (C - np.outer(g, h_prime.conj()) + np.outer(epsilon, g.conj())) / ff

    e = eta * x - U @ y_prime
    U += np.outer(e, g.conj())
    return (U, C)


def BFAPIupd(
    x: np.ndarray, U: np.ndarray, C: np.ndarray, ff: float = 0.97
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Batch FAPI.

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION.
    U : np.ndarray
        DESCRIPTION.
    P : np.ndarray
        DESCRIPTION.
    ff : float, optional
        DESCRIPTION. The default is 0.97.

    Returns
    -------
    None.

    """
    U, C = FAPIupd(x.T[0], U, C, ff)
    for n in range(1, x.shape[1]):
        U, C = FAPIupd(x.T[n], U, C, 1.0)
    return (U, C)


# %% Rayleigh quotient-based


def FRANSupd(
    x: np.ndarray, U: np.ndarray, mu: tp.Union[float, None] = None
) -> np.ndarray:
    """
    Fast Rayleigh quotient-based Adaptive Noise Subpace.

    Parameters
    ----------
    x : np.ndarray
        Observation vector.
    U : np.ndarray
        Current subspace estimate.
    mu : tp.Union[float, None], optional
        Forgetting factor. The default is None.

    Returns
    -------
    NumPy array
        Updated subspace estimate.

    """
    if x.ndim == 1:
        if mu is None:
            mu = la.norm(x) ** -2
        y = U.conj().T @ x
        T = U + mu * np.outer(x, y.conj())
        y2norm2 = la.norm(y) ** 2
        rho = (
            1 - 1 / np.sqrt(1 + (2 * mu + (mu**2) * la.norm(x) ** 2) * y2norm2)
        ) / y2norm2
        U = T - rho * np.outer((T @ y), y.conj())
    else:
        if mu is None:
            U = FRANSupd(x.T[0], U)
        else:
            U = FRANSupd(x.T[0], U, mu)
        for n in range(1, x.shape[1]):
            U = FRANSupd(x.T[n], U, 1.0)
    return U


def FDPMupd(
    x: np.ndarray, U: np.ndarray, mu: tp.Union[float, None] = None
) -> np.ndarray:
    """
    Fast Data Projection Method.

    Parameters
    ----------
    x : np.ndarray
        Observation vector.
    U : np.ndarray
        Current subspace estimate.
    mu : tp.Union[float, None], optional
        Forgetting factor. The default is None.

    Returns
    -------
    NumPy array
        Updated subspace estimate.

    """
    D = U.shape[1]
    if x.ndim == 1:
        if mu is None:
            mu = la.norm(x) ** -2
        y = U.conj().T @ x
        T = U + mu * np.outer(x, y.conj())
        if D == 1:
            U = T / la.norm(T)
        else:
            a = y - la.norm(y) * np.eye(D)[:, 0]
            Z = T - 2 / (la.norm(a) ** 2) * np.outer(T @ a, a.conj())
            U = colnorm(Z)
    else:
        N = x.shape[1]
        if mu is None:
            U = FRANSupd(x.T[0], U)
        else:
            U = FRANSupd(x.T[0], U, mu)
        for n in range(1, N):
            U = FDPMupd(x.T[n], U, 1.0)
    return U


def FSDPMupd(x: np.ndarray, U: np.ndarray, ff: float = 0.97) -> np.ndarray:
    """
    Fast and stable DPM update.

    Parameters
    ----------
    x : np.ndarray
        Data vector (or matrix).
    U : np.ndarray
        Current estimated subspace.
    ff : float, optional
        Forgetting factor. The default is 0.97.

    Returns
    -------
    U : np.ndarray
        Updated subspace estimate.

    """
    if x.ndim == 1:
        y = U.conj().T @ x
        z = U @ y
        y_norm = la.norm(y)
        w = z / y_norm + ff * x * y_norm
        q = w / la.norm(w) - z / y_norm
        U += np.outer(q, y.conj()) / y_norm
    else:
        U = FSDPMupd(x.T[0], U)
        for n in range(1, x.shape[1]):
            U = FSDPMupd(x.T[n], U, 1.0)
        # Y = U.conj().T @ x
        # Z = U @ Y
        # Y_col_norm = la.norm(Y, axis=0)
        # W = Z @ np.diag(1 / Y_col_norm) + ff * x @ np.diag(Y_col_norm)
        # W_col_norm = la.norm(W, axis=0)
        # Q = W @ np.diag(1 / W_col_norm) - Z @ np.diag(1 / Y_col_norm)
        # U += Q @ (Y @ np.diag(1 / Y_col_norm)).conj().T
    return U


# %% If main

if __name__ == "__main__":
    pass
