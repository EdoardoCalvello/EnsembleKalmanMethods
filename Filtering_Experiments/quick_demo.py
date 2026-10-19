"""
Quick demo for EnsembleKalmanMethods: short run (100 steps) using cheap predictor
and profiling to identify hotspots. Writes profile summary to
Filtering_Experiments/results/quick_demo_profile.txt and saves predictions.

This demo uses the simple G0 predictor (linear coupling) to avoid GP cost
and runs EnKF with a small ensemble. Designed to complete in seconds/minutes.
"""
import sys
sys.path.append('../')
import numpy as np
import scipy.integrate as scint
from Lorenz96.L96_multiscale import L96M
from Data_Assimilation.DA_Filtering import EnKF
import io, cProfile, pstats, time


def make_observation_operator(l96m):
    class observation_operator:
        def __init__(self):
            self.H = np.zeros((6, l96m.K))
            self.H[0,0] = 1
            self.H[1,1] = 1
            self.H[2,3] = 1
            self.H[3,4] = 1
            self.H[4,6] = 1
            self.H[5,7] = 1

        def forward(self, v):
            return self.H @ v
    return observation_operator()


class Psi:
    def __init__(self, l96m, TAU):
        self.l96m = l96m
        self.TAU = TAU

    def forward(self, v):
        y0 = np.asarray(v, dtype=float).ravel()
        sol = scint.solve_ivp(self.l96m.regressed, [0, self.TAU], y0, method='RK45', max_step=self.TAU)
        return sol.y[:, -1]


def main():
    # parameters
    STEPS = 100
    TAU = 0.001
    sigma = 0.1
    gamma = 0.1
    ENSEMBLE = 50

    # init model with cheap predictor to avoid GP cost
    l96m = L96M()
    l96m.set_stencil()
    l96m.set_G0_predictor()

    # generate short truth trajectory
    K = l96m.K
    states = np.zeros((K, STEPS))
    obs = np.zeros((6, STEPS))
    # initial slow state
    x = l96m.xk_star + 0.1 * np.random.randn(K)
    states[:,0] = x
    H = make_observation_operator(l96m)

    for n in range(STEPS-1):
        sol = scint.solve_ivp(l96m.regressed, [0, TAU], states[:,n], method='RK45', max_step=TAU)
        xn = sol.y[:, -1]
        states[:, n+1] = xn
        obs[:, n+1] = H.forward(xn) + np.random.normal(0, gamma, size=6)

    # set up EnKF with cheap forward operator
    psi = Psi(l96m, TAU)
    enkf = EnKF(psi, H, sigma=sigma, gamma=gamma, ensemble_size=ENSEMBLE)

    # initial ensemble around truth with spread
    ic = np.random.normal(loc=states[:,0:1], scale=1.0, size=(K, ENSEMBLE))

    # profile the EnKF run
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    predicted = enkf.run(obs, ic)
    pr.disable()
    t1 = time.perf_counter()

    # save results
    out_dir = './results'
    import os
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, 'quick_demo_predicted.npz'), prediction=predicted, truth=states, observations=obs)

    # write profile summary
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(30)
    profile_txt = s.getvalue()
    with open(os.path.join(out_dir, 'quick_demo_profile.txt'), 'w') as f:
        f.write(f"Quick demo total time: {t1-t0:.4f} seconds\n\n")
        f.write(profile_txt)

    print(f"Quick demo finished in {t1-t0:.4f}s. Profile written to {os.path.join(out_dir, 'quick_demo_profile.txt')}")


if __name__ == '__main__':
    main()
