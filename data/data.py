import numpy as np
import mpmath as mp

# Global high-precision setting, corresponding to MATLAB's digits(20)
mp.mp.dps = 20   # 20 significant digits

def free_sub(d: int, real_val, imag_val=14.134725):
    """
    Corresponds to MATLAB:
      result = free_sub(d, real_val)
    except that here imag is fixed.
    """
    d = int(d)
    real_val = mp.mpf(real_val)
    imag_val = mp.mpf(imag_val)

    N = 2 ** d

    # Denominator: de = sum_{n=1}^{N} 1 / n^real_val
    de = mp.mpf("0")
    for n in range(1, N + 1):
        de += 1 / (n ** real_val)

    # s1 = real + i * imag
    s1 = mp.mpc(real_val, imag_val)

    # Alternating sum
    sum_expr = mp.mpc(0)
    for n in range(1, N + 1):
        sign = (n % 2) * 2 - 1   # Odd: +1, even: -1
        term = sign * (n ** (-s1))
        sum_expr += term

    app = -sum_expr
    abs_val = abs(app / de)

    # result = -1/real_val * ln|app/de|
    result = -1 / real_val * mp.log(abs_val)
    return result


def saveZ2_npz():
    # real_part = 0.1 + 0.0005 * (1:1600)
    real_part = 0.1 + 0.0005 * np.arange(1, 1601, dtype=float)

    # Compute pointwise for d = 4, 6, 8
    result_d4 = np.array([float(free_sub(4, x)) for x in real_part], dtype=float)
    result_d6 = np.array([float(free_sub(6, x)) for x in real_part], dtype=float)
    result_d8 = np.array([float(free_sub(8, x)) for x in real_part], dtype=float)

    # Save as an npz file
    np.savez(
        "data.npz",
        real_part=real_part,
        result_d4=result_d4,
        result_d6=result_d6,
        result_d8=result_d8,
    )


if __name__ == "__main__":
    saveZ2_npz()