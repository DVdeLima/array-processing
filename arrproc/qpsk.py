# %% Docstring

"""
Array Signal Processing.

------------------------

Includes:
    0. QPSK alphabet
        alphabet
    1. Auxiliary functions
        sign
    2. De/modulator (not implemented yet)
        modulator
        demodulator
    3. Bit error rate estimation
        ber_estimator
"""

# %% __all__

__all__ = [
    "alphabet",
    "modulator",
    "demodulator",
    "ber_estimator"
]

# %% Dependencies

import typing as tp
from numpy import angle, arange, exp, ndarray, pi, sign, sqrt, stack
from sklearn.cluster import KMeans

# %% Alphabet

alphabet = exp(1j * arange(pi / 4, 2 * pi, pi / 2))


# %% Auxiliary functions

def csign(number: complex) -> complex:
    """
    Complex sign function

    Parameters
    ----------
    number : complex
        Complex number (or NumPy array).

    Returns
    -------
    complex
        Sign of real and imaginary portions.

    """
    return sign(number.real) + 1j * sign(number.imag)


# %% De/modulator

def modulator(binary_sequence: tp.Union[list, ndarray],
              constellation: ndarray = alphabet) -> ndarray:
    pass


def demodulator(signals: tp.Union[list, ndarray],
                constellation: ndarray = alphabet) -> ndarray:
    pass


def phase_shift_estimator(signals: ndarray) -> ndarray:
    pass


# %% BER estimator

def ber_estimator(true: ndarray,
                  received: ndarray) -> ndarray:
    """
    Bit Error Rate estimator for QPSK

    Parameters
    ----------
    true : NumPy array
        True QPSK symbol sequence.
    received : NumPy array
        Received QPSK symbol sequence.

    Returns
    -------
    float
        Estimated BER.

    """
    diff = abs(csign(true) - csign(received))
    diff[diff == 2] = sqrt(2)
    return sum(diff / sqrt(2)) / len(diff)
