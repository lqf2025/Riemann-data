import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mpmath as mp

# Set the precision globally here if needed
mp.mp.dps = 20  # Number of significant digits; adjust as needed

def free_sub(d, real_part, imag_val):
    """
    Corresponds to the MATLAB function:
    function result = free_sub(d, real_part, imag_val)
    """
    # Convert inputs to mpmath high-precision numbers
    d = int(d)
    real_part = mp.mpf(real_part)
    imag_val = mp.mpf(imag_val)

    N = 2 ** d  # N = 2^d

    # Denominator: sum_{n=1}^{2^d} 1 / n^real_part
    de = mp.mpf('0')
    for n in range(1, N + 1):
        de += 1 / (n ** real_part)

    # s1 = real_part + i * imag_val
    s1 = mp.mpc(real_part, imag_val)

    # Alternating sum
    sum_expr = mp.mpc(0)
    for n in range(1, N + 1):
        # ((mod(n,2))*2 - 1) -> odd: +1, even: -1
        sign = (n % 2) * 2 - 1
        term = sign * (n ** (-s1))
        sum_expr += term

    app = -sum_expr

    abs_val = abs(app / de)
    result = -1 / real_part * mp.log(abs_val)  # Natural logarithm

    return result

def compute_result_for_d(d_val):
    # Use np.vectorize for elementwise evaluation, then convert to float before saving
    f = np.vectorize(lambda x: float(free_sub(d_val, real_part, x)))
    return f(data)

mat_in = loadmat('tt_type11_full.mat')

# Corresponds to MATLAB:
# S = load('tt_type11_full.mat'); data = S.tt_type11_full;
data = mat_in['tt_type11_full']  # Change this if the variable name is different
data = np.array(data, dtype=float)

# Ensure data is a NumPy array
real_part = mp.mpf('0.3')
results_d4_type2 = compute_result_for_d(4)
results_d8_type2 = compute_result_for_d(8)

real_part = mp.mpf('0.5')

# === Load your data ===

# Define a helper function that applies free_sub to each element of data
results_d4_type1 = compute_result_for_d(4)
results_d8_type1 = compute_result_for_d(8)

np.savez(
    'data2.npz',
    results_d4_type2=results_d4_type2,
    results_d8_type2=results_d8_type2,
    results_d4_type1=results_d4_type1,
    results_d8_type1=results_d8_type1
)