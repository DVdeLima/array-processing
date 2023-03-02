# %% Docstring

"""
PRS generator module.

---------------------

Includes:
    1. C/A code generator (cacode)
    2. FFT calculator (_genFFT)
    3. Sampled signal generator
        Sampled PRS (genC)
        Sampled Correlator bank (genQ)
    4. Compressed signal generator
        Compressed correlator bank (genQtil)
        Compressed PRS (genCtil)
"""

# %% __all__

__all__ = ["cacode", "genC", "genQ", "genQtil", "genCtil"]

# %% Load dependencies

import typing as tp
import os.path as op
import numpy as np
import scipy as sp

# %% Debug modules

# import pdb as ipdb
# ipdb.set_trace()

# %% PRS generator


def cacode(satno: int) -> np.ndarray:
    """
    C/A code generator.

    Parameters
    ----------
    satno : int
        Satellite number

    Returns
    -------
    C/A code

    """
    # phase assignments
    phase = np.array(
        [
            [2, 6],
            [3, 7],
            [4, 8],
            [5, 9],
            [1, 9],
            [2, 10],
            [1, 8],
            [2, 9],
            [3, 10],
            [2, 3],
            [3, 4],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [1, 4],
            [2, 5],
            [3, 6],
            [4, 7],
            [5, 8],
            [6, 9],
            [1, 3],
            [4, 6],
            [5, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [1, 6],
            [2, 7],
            [3, 8],
            [4, 9],
        ]
    )

    # select taps for G2 delay
    s1, s2 = phase[satno - 1] - 1

    # initial state
    G1 = -1 * np.ones(10, dtype=int)
    G2 = -1 * np.ones(10, dtype=int)

    G = np.zeros(1023)

    for i in range(1023):
        # Gold code
        G[i] = G2[s1] * G2[s2] * G1[9]
        # generator 1 - shift register 1
        tmp = G1[0]
        G1[0] = G1[2] * G1[9]
        G1[1:10] = np.hstack((tmp, G1[1:9]))
        # generator 2 - shift register 2
        tmp = G2[0]
        G2[0] = G2[1] * G2[2] * G2[5] * G2[7] * G2[8] * G2[9]
        G2[1:10] = np.hstack((tmp, G2[1:9]))

    return G


def Legendre(L: int = 10223) -> np.ndarray:
    """
    Generate Legendre sequence.

    Parameters
    ----------
    L : int, optional
        Legendre sequence length. The default is 10223.

    Returns
    -------
    Legendre sequence length.

    """
    LS = np.zeros(L)
    idx = (np.arange(1, L / 2) ** 2 % L).astype(int)
    LS[idx] = 1
    return LS


def Weil_idx(satno: int) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Generate Weil indexes.

    Parameters
    ----------
    satno : int
        Satellite number.

    Returns
    -------
    Tuple.
        Weil index for L1CP and L1CD code.
    """
    satno = satno + 1
    W_L1CP = [
        5111,
        5109,
        5108,
        5106,
        5103,
        5101,
        5100,
        5098,
        5095,
        5094,
        5093,
        5091,
        5090,
        5081,
        5080,
        5069,
        5068,
        5054,
        5044,
        5027,
        5026,
        5014,
        5004,
        4980,
        4915,
        4909,
        4893,
        4885,
        4832,
        4824,
        4591,
        3706,
        5092,
        4986,
        4965,
        4920,
        4917,
        4858,
        4847,
        4790,
        4770,
        4318,
        4126,
        3961,
        3790,
        4911,
        4881,
        4827,
        4795,
        4789,
        4725,
        4675,
        4539,
        4535,
        4458,
        4197,
        4096,
        3484,
        3481,
        3393,
        3175,
        2360,
        1852,
        5065,
        5063,
        5055,
        5012,
        4981,
        4952,
        4934,
        4932,
        4786,
        4762,
        4640,
        4601,
        4563,
        4388,
        3820,
        3687,
        5052,
        5051,
        5047,
        5039,
        5015,
        5005,
        4984,
        4975,
        4974,
        4972,
        4962,
        4913,
        4907,
        4903,
        4833,
        4778,
        4721,
        4661,
        4660,
        4655,
        4623,
        4590,
        4548,
        4461,
        4442,
        4347,
        4259,
        4256,
        4166,
        4155,
        4109,
        4100,
        4023,
        3998,
        3979,
        3903,
        3568,
        5088,
        5050,
        5020,
        4990,
        4982,
        4966,
        4949,
        4947,
        4937,
        4935,
        4906,
        4901,
        4872,
        4865,
        4863,
        4818,
        4785,
        4781,
        4776,
        4775,
        4754,
        4696,
        4690,
        4658,
        4607,
        4599,
        4596,
        4530,
        4524,
        4451,
        4441,
        4396,
        4340,
        4335,
        4296,
        4267,
        4168,
        4149,
        4097,
        4061,
        3989,
        3966,
        3789,
        3775,
        3622,
        3523,
        3515,
        3492,
        3345,
        3235,
        3169,
        3157,
        3082,
        3072,
        3032,
        3030,
        4582,
        4595,
        4068,
        4871,
        4514,
        4439,
        4122,
        4948,
        4774,
        3923,
        3411,
        4745,
        4195,
        4897,
        3047,
        4185,
        4354,
        5077,
        4042,
        2111,
        4311,
        5024,
        4352,
        4678,
        5034,
        5085,
        3646,
        4868,
        3668,
        4211,
        2883,
        2850,
        2815,
        2542,
        2492,
        2376,
        2036,
        1920,
    ]
    W_L1CD = [
        5097,
        5110,
        5079,
        4403,
        4121,
        5043,
        5042,
        5104,
        4940,
        5035,
        4372,
        5064,
        5084,
        5048,
        4950,
        5019,
        5076,
        3736,
        4993,
        5060,
        5061,
        5096,
        4983,
        4783,
        4991,
        4815,
        4443,
        4769,
        4879,
        4894,
        4985,
        5056,
        4921,
        5036,
        4812,
        4838,
        4855,
        4904,
        4753,
        4483,
        4942,
        4813,
        4957,
        4618,
        4669,
        4969,
        5031,
        5038,
        4740,
        4073,
        4843,
        4979,
        4867,
        4964,
        5025,
        4579,
        4390,
        4763,
        4612,
        4784,
        3716,
        4703,
        4851,
        4955,
        5018,
        4642,
        4840,
        4961,
        4263,
        5011,
        4922,
        4317,
        3636,
        4884,
        5041,
        4912,
        4504,
        4617,
        4633,
        4566,
        4702,
        4758,
        4860,
        3962,
        4882,
        4467,
        4730,
        4910,
        4684,
        4908,
        4759,
        4880,
        4095,
        4971,
        4873,
        4561,
        4588,
        4773,
        4997,
        4583,
        4900,
        4574,
        4629,
        4676,
        4181,
        5057,
        4944,
        4401,
        4586,
        4699,
        3676,
        4387,
        4866,
        4926,
        4657,
        4477,
        4359,
        4673,
        4258,
        4447,
        4570,
        4486,
        4362,
        4481,
        4322,
        4668,
        3967,
        4374,
        4553,
        4641,
        4215,
        3853,
        4787,
        4266,
        4199,
        4545,
        4208,
        4485,
        3714,
        4407,
        4182,
        4203,
        3788,
        4471,
        4691,
        4281,
        4410,
        3953,
        3465,
        4801,
        4278,
        4546,
        3779,
        4115,
        4193,
        3372,
        3786,
        3491,
        3812,
        3594,
        4028,
        3652,
        4224,
        4334,
        3245,
        3921,
        3840,
        3514,
        2922,
        4227,
        3376,
        3560,
        4989,
        4756,
        4624,
        4446,
        4174,
        4551,
        3972,
        4399,
        4562,
        3133,
        4157,
        5053,
        4536,
        5067,
        3905,
        3721,
        3787,
        4674,
        3436,
        2673,
        4834,
        4456,
        4056,
        3804,
        3672,
        4205,
        3348,
        4152,
        3883,
        3473,
        3669,
        3455,
        2318,
        2945,
        2947,
        3220,
        4052,
        2953,
    ]
    return (W_L1CP[satno], W_L1CD[satno])


# %% FFT & spectrum gen.


def _genFFT(B: float, T: float, satno: int) -> np.ndarray:
    """
    Generate FFT (internal use).

    Parameters
    ----------
    B : float
        Bandwidth.
    T : float
        Epoch period.
    satno : int
        Satellite number (1-32).

    Returns
    -------
    FFT of C/A code.

    """
    FFT_file = "CA_SPEC_FFT_" + str(satno) + "_" + str(round(B)) + ".npz"
    if op.isfile(FFT_file):
        data = np.load(FFT_file)
        CA_FFT = data["CA_FFT"]
        data.close()
    else:
        N = round(2 * B * T)

        ca_code = cacode(satno)
        ca_code_len = len(ca_code)

        CA_FFT = N // ca_code_len * np.tile(sp.fft.rfft(ca_code), N // ca_code_len)
        R_ca_code = (
            np.array(
                [
                    (ca_code * np.roll(ca_code, -shift)).sum()
                    for shift in range(ca_code_len)
                ]
            )
            / ca_code_len
        )

        CA_SPEC = sp.fft.fftshift(
            N // ca_code_len * np.tile(sp.fft.rfft(R_ca_code), N // ca_code_len).real
        )
        np.savez(
            FFT_file,
            CA_FFT=CA_FFT,
            CA_SPEC=CA_SPEC,
            SAT=satno,
            R_ca_code=R_ca_code,
            B=B,
        )
    return CA_FFT


# %% Sampled PRS and CB


def genC(
    B: float, T: float, satnos: tp.Union[int, list], delays: tp.Union[float, list]
) -> np.ndarray:
    """
    Generate sampled PRS sequence.

    Parameters
    ----------
    B : float
        Bandwidth.
    T : float
        Epoch period.
    satnos : int or list (of ints)
        Satellite number(s).
    delays : float or list (of floats)
        Time delay(s).

    Returns
    -------
    Sampled PRS sequence.

    """
    if np.isscalar(satnos) and not np.isscalar(delays):
        satnos = np.repeat(satnos, len(delays))
        D = 1
    else:
        sats = np.unique(satnos)
        D = len(sats)

    if np.isscalar(delays):
        L = 1
    else:
        L = len(delays)

    Tc = 1 / B
    N = round(2 * B * T)
    f0 = 2 * B / N
    samples = np.arange(-B, B, f0)

    PULSE_FFT = np.fft.fftshift(np.sqrt(Tc) * np.sinc(samples * Tc))
    E = sp.fft.fftshift(np.exp(-2j * np.pi * np.outer(samples, delays)), axes=0)

    if D == 1:
        if L == 1:
            CA_FFT = _genFFT(B, T, satnos)
        else:
            CA_FFT = _genFFT(B, T, satnos[0])
        C = sp.fft.fftshift(
            sp.fft.ifft(np.tile(PULSE_FFT * CA_FFT, (L, 1)).T * E, axis=0).real, axes=0
        )
    else:
        CA_FFT = np.vstack([_genFFT(B, T, sat) for sat in sats]).T
        S = np.tile(sats, (L, 1)).T == np.tile(satnos, (D, 1))
        C = sp.fft.fftshift(
            sp.fft.ifft(np.tile(PULSE_FFT, (L, 1)).T * (CA_FFT @ S * E), axis=0).real,
            axes=0,
        )

    if L == 1:
        return C * np.sqrt(N / (C @ C))
    else:
        return C @ np.diag(np.sqrt(N / np.diag(C.T @ C)))


def genQ(
    B: float,
    T: float,
    satnos: tp.Union[int, list],
    taps: tp.Union[int, list, np.ndarray] = 11,
) -> np.ndarray:
    """
    Generate correlator bank.

    Parameters
    ----------
    B : float
        Bandwidth.
    T : float
        Epoch period.
    satnos : int or list (of ints)
        Satellite number(s)
    taps : int, optional
        Number of correlator bank taps. The default is 11.

    Returns
    -------
    None.

    """
    if np.isscalar(taps):
        Tc = 1 / B
        delays = np.linspace(-Tc, Tc, taps)
    else:
        delays = taps
    return genC(B, T, satnos, delays)


# %% Compressed PRS and CB


def genQtil(
    B: float, T: float, satnos: tp.Union[int, list], taps: int = 11
) -> np.ndarray:
    """
    Generate compressed correlator bank.

    Parameters
    ----------
    B : float
        Bandwidth.
    T : float
        Epoch period.
    satnos : int or list (of ints)
        Satellite number(s).
    taps : int, optional
        Number of correlator bank taps. The default is 11.

    Returns
    -------
    Compressed correlator bank.

    """
    Qtil, Sigma, VH = np.linalg.svd(genQ(B, T, satnos, taps), full_matrices=False)
    return (Qtil, np.diag(Sigma) @ VH)


def genCtil(
    B: float,
    T: float,
    satnos: tp.Union[int, list],
    delays: tp.Union[float, list],
    Qtil: np.ndarray,
) -> np.ndarray:
    """
    Generate (compressed) correlated PRSs.

    Parameters
    ----------
    B : float
        Bandwidth.
    T : float
        Epoch period.
    satnos : tp.Union[int, list]
        Satellite number(s).
    delays : tp.Union[float, list]
        Time delay(s).
    Qtil : np.ndarray
        Compressed correlator bank.

    Returns
    -------
    Correlated PRSs.

    """
    return Qtil.T @ genC(B, T, satnos, delays)


# %% If main

if __name__ == "__main__":
    pass
