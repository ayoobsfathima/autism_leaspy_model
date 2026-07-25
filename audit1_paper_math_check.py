"""
AUDIT 1: Does the simulator implement the paper's Eq. 3 correctly?

Paper Eq. 3:
  psi^l_ik(t) = e^xi_i (t - t0 - tau_i) + t0 - sum_{m=1..l} delta^m_k
  P(y>=l)     = [1 + (1/p_k - 1) exp( -(v_k * psi^l_ik(t) + w_ik) / (p_k(1-p_k)) )]^-1

So the linear predictor is:
  x = v_k * psi = v_k*e^xi*(t-t0-tau) + v_k*t0 - v_k*cum_delta + w_ik
                                        ^^^^^^^^ constant per item
Simulator uses:
  x = v_k*e^xi*(t-t0-tau)             - v_k*cum_delta + w_ik
"""
import numpy as np

v, p, t0, tau, xi, w = 0.18, 0.35, 3.0, 0.0, 0.0, 0.0
cum_delta = 1.1
age = 6.0

def sig(x, p): return 1/(1+(1/p-1)*np.exp(-x/(p*(1-p))))

x_paper = v*(np.exp(xi)*(age-t0-tau) + t0 - cum_delta) + w
x_sim   = v*np.exp(xi)*(age-t0-tau) - v*cum_delta + w

print("paper linear predictor:", round(x_paper,5))
print("sim   linear predictor:", round(x_sim,5))
print("difference            :", round(x_paper-x_sim,5), " (= v*t0 =", round(v*t0,5), ")")
print()
print("P(y>=l) paper:", round(sig(x_paper,p),4))
print("P(y>=l) sim  :", round(sig(x_sim,p),4))
