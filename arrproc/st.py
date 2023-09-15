# %% Docstring

"""
Subpace Tracking.

--------------------

Includes:
    1. Projection-based
        Projection Approximation Subspace Tracking (past_upd)
        PAST via deflation (pastd_upd)
        Fast Approximate Subpsace Tracking (fast_upd)
        Orthogonal PAST (opast_upd)
        Fast Constrained PAST (fcpast_upd)
        Generalized YAST (gyast_upd)
    2. Power iteration-based
        Fast Approximate Power Iteration (fapi_upd)
    3. Raileigh quotient-based
        Fast Rayleigh quotient Adaptive Noise Subspace (frans_upd)
        Fast Data Projection Method (fdpm_upd)
        Fast and Stable DPM (fsdpm_upd)
"""

# %% Load dependencies

import numpy as np
import numpy.linalg as la
import typing as tp


# %% Auxiliary functions

def stripe(M: int, N: int, normalize: bool = True) -> np.ndarray:
    if M == N:
        return np.eye(M)
    if M > N:
        res = np.vstack((np.tile(np.eye(N, dtype=complex), (M//N, 1)),
                         np.eye(N, M % N, dtype=complex).T))
    else:
        res = stripe(N, M).T
    if normalize:
        return res @ np.diagflat(res.sum(axis=0) ** -0.5)
    return res


def tri(A: np.ndarray) -> np.ndarray:
    """
    Triangular conditioning.

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


def svd_upd(X: np.ndarray,
            U: np.ndarray,
            evs: np.ndarray,
            ff: float = 0.97) -> tp.Tuple[np.ndarray,
                                          np.ndarray]:
    """
    SVD subspace update.

    Parameters
    ----------
    X : NumPy array
        Observation vector/matrix.
    U : NumPy array
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
    R = len(evs)
    C = U @ np.diagflat(evs) @ U.conj().T
    if X.ndim == 1:
        C = ff * C + np.outer(X, X.conj())
    else:
        C = ff * C + X @ X.conj().T
    evs, U = la.eig(C)
    return (U[:, :R], evs[:R])


# %% PAST-based

def past_upd(X: np.ndarray,
             U: np.ndarray,
             C: np.ndarray,
             ff: float = 0.97) -> tp.Tuple[np.ndarray,
                                           np.ndarray]:
    """
    Projection Approximation Subspace Tracking update.

    Usage:
        U, C = past_upd(X, U, C, ff)

    Parameters
    ----------
    X : NumPy array
        Observation vector/matrix.
    U : NumPy array
        Current subspace estimate.
    C : NumPy array
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
        y = U.T.conj() @ X
        e = X - U @ y
        h = C @ y
        g = h / (ff + y.conj().T @ h)
        U = U + np.outer(e, g.conj())
        C = tri(C - np.outer(g, h.conj()))
    else:
        Y = U.conj().T @ X
        E = X - U @ Y
        C = ff * C + Y @ Y.conj().T
        G = la.inv(C) @ Y
        U = U + E @ G.conj().T
    return (U, C)


def bpast_upd(X: np.ndarray,
              U: np.ndarray,
              C: np.ndarray,
              ff: float = 0.97) -> tp.Tuple[np.ndarray,
                                            np.ndarray]:
    """
    Batch PAST update.

    Usage:
        U, C = past_upd(X, U, C, ff)

    Parameters
    ----------
    X : NumPy array
        Observation vector/matrix.
    U : NumPy array
        Current subspace estimate.
    C : NumPy array
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
    U, C = past_upd(X.T[0], U, C, ff)
    for n in range(1, X.shape[1]):
        U, C = past_upd(X.T[n], U, C, 1.0)
    return (U, C)


def pastd_upd(X: np.ndarray,
              U: np.ndarray,
              evs: np.ndarray,
              ff: float = 0.97) -> tp.Tuple[np.ndarray,
                                            np.ndarray]:
    """
    Deflationary PAST update.

    Usage:
        U, evs = pastd_upd(x, U, evs, ff)

    Parameters
    ----------
    x : NumPy array
        Observation vector.
    U : NumPy array
        Current subspace estimate.
    evs : NumPy array
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


def fast_upd(x: np.ndarray,
             U: np.ndarray) -> tp.Tuple[np.ndarray,
                                        np.ndarray]:
    """
    Fast Approximate Subspace Tracking (FAST)

    Parameters
    ----------
    x : NumPy array
        Updated data vector/matrix.
    U : NumPy array
        Old subspace estimate.

    Returns
    -------
    U : NumPy array
        Updated subspace estimate.
    sv : NumPy array (vector)
        Updated singular values.
    """
    if ((x.ndim == 2) & (x.shape[1] > 1)):
        X = x[:, :-1]
        x = np.asmatrix(x[:, -1]).T
    else:
        X = None
    a = U.T.conj() @ x
    z = x - U @ a
    b = la.norm(z)
    q = z / b
    if X is None:
        E = np.vstack((a, b))
    else:
        N = X.shape[1]
        E = np.vstack((np.hstack((U.T.conj() @ X, a)),
                       np.hstack((np.zeros(N), b))))
    U_E, s_E = la.svd(E @ E.T.conj(),
                      hermitian=True)[:2]
    return (np.hstack((U, q)) @ U_E[:, :-1], s_E[:-1])


def opast_upd(x: np.ndarray,
              U: np.ndarray,
              C: np.ndarray,
              ff: float = 0.97) -> tp.Tuple[np.ndarray,
                                            np.ndarray]:
    """
    Orthogonal PAST.

    Parameters
    ----------
    x : NumPy array
        Input vector/matrix.
    U : NumPy array
        Current subspace estimate.
    C : NumPy array
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
    tau = (1 / np.sqrt(1
                       + q2norm2 * (la.norm(x) ** 2
                                    - la.norm(y) ** 2))
           - 1) / q2norm2
    p = (U @ (tau * q - gamma * (1 + tau * q2norm2) * y)
         + (1 + tau * q2norm2) * gamma * x)
    U += np.outer(p, q.conj())
    C = C / ff - gamma * np.outer(q, q.conj())
    return (U, C)


def bopast_upd(x: np.ndarray,
               U: np.ndarray,
               C: np.ndarray,
               ff: float = 0.97) -> tp.Tuple[np.ndarray,
                                             np.ndarray]:
    """
    Orthogonal PAST.

    Parameters
    ----------
    x : NumPy array
        Input vector/matrix.
    U : NumPy array
        Current subspace estimate.
    C : NumPy array
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
    U, C = opast_upd(x.T[0], U, C, ff)
    for n in range(1, x.shape[1]):
        U, C = opast_upd(x.T[n], U, C, 1.0)
    return (U, C)


def fcpast_upd(X: np.ndarray,
               U: np.ndarray,
               C: np.ndarray,
               ff: float = 0.97) -> tp.Tuple[np.ndarray,
                                             np.ndarray]:
    """
    Fast Constrained PAST.

    Parameters
    ----------
    X : NumPy array
        DESCRIPTION.
    U : NumPy array
        DESCRIPTION.
    C : NumPy array
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


def bfcpast_upd(X: np.ndarray,
                U: np.ndarray,
                C: np.ndarray,
                ff: float = 0.97) -> tp.Tuple[np.ndarray,
                                              np.ndarray]:
    """
    Batch Fast Constrained PAST.

    Parameters
    ----------
    X : NumPy array
        DESCRIPTION.
    U : NumPy array
        DESCRIPTION.
    C : NumPy array
        DESCRIPTION.
    ff : float, optional
        DESCRIPTION. The default is 0.97.

    Returns
    -------
    None.

    """
    U, C = fcpast_upd(X.T[0], U, C, ff)
    for n in range(1, X.shape[1]):
        U, C = fcpast_upd(X.T[n], U, C, 1.0)
    return (U, C)


def gyast_upd(x: np.ndarray,
              U: np.ndarray,
              P: np.ndarray,
              sigma_n: float,
              ff: float = 0.97) -> tp.Tuple[np.ndarray,
                                            np.ndarray,
                                            float]:
    """
    Generalized YAST.

    Parameters
    ----------
    x : NumPy array
        DESCRIPTION.
    U : NumPy array
        DESCRIPTION.
    P : NumPy array
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
    c = (ell**2 * q.conj().T @ q1
         + gamma + 2 * ell / abs(r) * (r * q.conj().T @ z)).real
    P = P1 + c * np.outer(q, q.conj()) - C - C.conj().T
    lambda_m = np.trace(P2) - np.trace(P)
    sigma_n = np.sqrt(min((lambda_m, sigma_n**2)))
    return U, P, sigma_n


def bgyast_upd(x: np.ndarray,
               U: np.ndarray,
               P: np.ndarray,
               sigma_n: float,
               ff: float = 0.97) -> tp.Tuple[np.ndarray,
                                             np.ndarray,
                                             float]:
    """
    Batch GYAST.

    Parameters
    ----------
    x : NumPy array
        DESCRIPTION.
    U : NumPy array
        DESCRIPTION.
    P : NumPy array
        DESCRIPTION.
    sigma_n : float
        DESCRIPTION.
    ff : float
        DESCRIPTION.

    Returns
    -------
    None.

    """
    U, P, sigma_n = gyast_upd(x.T[0], U, P, sigma_n, ff)
    for n in range(1, x.shape[1]):
        U, P, sigma_n = gyast_upd(x.T[n], U, P, sigma_n, 1.0)
    return U, P, sigma_n


# %% Power Iteration-based

def fapi_upd(x: np.ndarray,
             U: np.ndarray,
             C: np.ndarray,
             ff: float = 0.97) -> tp.Tuple[np.ndarray,
                                           np.ndarray]:
    """
    Fast Approximate Power Iteration.

    Parameters
    ----------
    x : NumPy array
        DESCRIPTION.
    U : NumPy array
        DESCRIPTION.
    P : NumPy array
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


def bfapi_upd(x: np.ndarray,
              U: np.ndarray,
              C: np.ndarray,
              ff: float = 0.97) -> tp.Tuple[np.ndarray,
                                            np.ndarray]:
    """
    Batch FAPI.

    Parameters
    ----------
    x : NumPy array
        DESCRIPTION.
    U : NumPy array
        DESCRIPTION.
    P : NumPy array
        DESCRIPTION.
    ff : float, optional
        DESCRIPTION. The default is 0.97.

    Returns
    -------
    None.

    """
    U, C = fapi_upd(x.T[0], U, C, ff)
    for n in range(1, x.shape[1]):
        U, C = fapi_upd(x.T[n], U, C, 1.0)
    return (U, C)


# %% Rayleigh quotient-based

def frans_upd(x: np.ndarray,
              U: np.ndarray,
              mu: float = None) -> np.ndarray:
    """
    Fast Rayleigh quotient-based Adaptive Noise Subpace.

    Parameters
    ----------
    x : NumPy array
        Observation vector.
    U : NumPy array
        Current subspace estimate.
    mu : float, optional
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
        rho = (1
               - 1 / np.sqrt(1
                             + (2 * mu
                                + (mu**2)
                                * la.norm(x) ** 2)
                             * y2norm2)) / y2norm2
        U = T - rho * np.outer((T @ y), y.conj())
    else:
        if mu is None:
            U = frans_upd(x.T[0], U)
        else:
            U = frans_upd(x.T[0], U, mu)
        for n in range(1, x.shape[1]):
            U = frans_upd(x.T[n], U, 1.0)
    return U


def fdpm_upd(x: np.ndarray,
             U: np.ndarray,
             mu: float = None) -> np.ndarray:
    """
    Fast Data Projection Method.

    Parameters
    ----------
    x : NumPy array
        Observation vector.
    U : NumPy array
        Current subspace estimate.
    mu : float, optional
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
            U = Z @ np.diag(la.norm(Z, axis=0) ** -1)
    else:
        N = x.shape[1]
        if mu is None:
            U = fdpm_upd(x.T[0], U)
        else:
            U = fdpm_upd(x.T[0], U, mu)
        for n in range(1, N):
            U = fdpm_upd(x.T[n], U, 1.0)
    return U


def fsdpm_upd(x: np.ndarray,
              U: np.ndarray,
              ff: float = 0.97) -> np.ndarray:
    """
    Fast and stable DPM update.

    Parameters
    ----------
    x : NumPy array
        Data vector (or matrix).
    U : NumPy array
        Current estimated subspace.
    ff : float, optional
        Forgetting factor. The default is 0.97.

    Returns
    -------
    U : NumPy array
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
        U = fsdpm_upd(x.T[0], U)
        for n in range(1, x.shape[1]):
            U = fsdpm_upd(x.T[n], U, 1.0)
    return U
