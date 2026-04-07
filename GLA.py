import mpmath as mp
from multiprocessing import Pool
import numpy as np
def theta_rs(t):
    z = mp.mpf('0.25') + 0.5j * mp.mpf(t)
    return mp.im(mp.loggamma(z)) - (mp.mpf(t)/2) * mp.log(mp.pi)
def Z_accurate(t):
    s = mp.mpf('0.5') + 1j * mp.mpf(t)
    theta_val = theta_rs(t)
    return mp.re(mp.e**(1j * theta_val) * mp.zeta(s))
def Z_riemann_siegel(t):
    N = int(mp.floor(mp.sqrt(t / (2 * mp.pi))))
    theta_val = theta_rs(t)
    s = mp.nsum(lambda n: (1/mp.sqrt(n)) * mp.cos(theta_val - t * mp.log(n)), [1, N])
    return 2 * s
def demo(t):
    N = int(mp.floor(mp.sqrt(t / (2 * mp.pi))))
    return mp.nsum(lambda n: 1 / mp.sqrt(n), [1, N])
if __name__ == '__main__':
    mp.dps = 50 
    start=420
    final=450
    num=1000000
    d=demo(start)
    t_vals = [mp.mpf(start) + i * (mp.mpf(final) - mp.mpf(start)) / num for i in range(num)]
    with Pool(processes=46) as pool:  # adjust number of processes
        results = pool.map(Z_accurate, t_vals)
        results2 = pool.map(Z_riemann_siegel, t_vals)
    results=np.array(results)
    results2=np.array(results2)
    Z_acc=results/d/2
    Z_app=results2/d/2
    zero_indices_acc = np.where(np.sign(Z_acc[:-1]) * np.sign(Z_acc[1:]) < 0)[0]
    zeros_t_acc = [(t_vals[i] + t_vals[i+1])/2 for i in zero_indices_acc]

    zero_indices_app = np.where(np.sign(Z_app[:-1]) * np.sign(Z_app[1:]) < 0)[0]
    zeros_t_app = [(t_vals[i] + t_vals[i+1])/2 for i in zero_indices_app]
    min_len = min(len(zeros_t_acc), len(zeros_t_app))
    zero_diffs = [zeros_t_acc[i] - zeros_t_app[i] for i in range(min_len)]
    np.savez("GLA1.npz",Z_acc=Z_acc,Z_app=Z_app,zeros_t_acc=zeros_t_acc,zeros_t_app=zeros_t_app,zero_diffs=zero_diffs)


    start=6595000
    final=6595000+10
    num=1000000
    d=demo(start)
    t_vals = [mp.mpf(start) + i * (mp.mpf(final) - mp.mpf(start)) / num for i in range(num)]
    with Pool(processes=46) as pool:  # adjust number of processes
        results = pool.map(Z_accurate, t_vals)
        results2 = pool.map(Z_riemann_siegel, t_vals)
    results=np.array(results)
    results2=np.array(results2)
    Z_acc=results/d/2
    Z_app=results2/d/2
    zero_indices_acc = np.where(np.sign(Z_acc[:-1]) * np.sign(Z_acc[1:]) < 0)[0]
    zeros_t_acc = [(t_vals[i] + t_vals[i+1])/2 for i in zero_indices_acc]

    zero_indices_app = np.where(np.sign(Z_app[:-1]) * np.sign(Z_app[1:]) < 0)[0]
    zeros_t_app = [(t_vals[i] + t_vals[i+1])/2 for i in zero_indices_app]
    min_len = min(len(zeros_t_acc), len(zeros_t_app))
    zero_diffs = [zeros_t_acc[i] - zeros_t_app[i] for i in range(min_len)]
    np.savez("GLA2.npz",Z_acc=Z_acc,Z_app=Z_app,zeros_t_acc=zeros_t_acc,zeros_t_app=zeros_t_app,zero_diffs=zero_diffs)

    start=267653395648
    final=267653395648+12
    num=1000000
    d=demo(start)
    t_vals = [mp.mpf(start) + i * (mp.mpf(final) - mp.mpf(start)) / num for i in range(num)]
    with Pool(processes=46) as pool:  # adjust number of processes
        results = pool.map(Z_accurate, t_vals)
        results2 = pool.map(Z_riemann_siegel, t_vals)
    results=np.array(results)
    results2=np.array(results2)
    Z_acc=results/d/2
    Z_app=results2/d/2
    zero_indices_acc = np.where(np.sign(Z_acc[:-1]) * np.sign(Z_acc[1:]) < 0)[0]
    zeros_t_acc = [(t_vals[i] + t_vals[i+1])/2 for i in zero_indices_acc]

    zero_indices_app = np.where(np.sign(Z_app[:-1]) * np.sign(Z_app[1:]) < 0)[0]
    zeros_t_app = [(t_vals[i] + t_vals[i+1])/2 for i in zero_indices_app]
    min_len = min(len(zeros_t_acc), len(zeros_t_app))
    zero_diffs = [zeros_t_acc[i] - zeros_t_app[i] for i in range(min_len)]
    np.savez("GLA3.npz",Z_acc=Z_acc,Z_app=Z_app,zeros_t_acc=zeros_t_acc,zeros_t_app=zeros_t_app,zero_diffs=zero_diffs)
