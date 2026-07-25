"""AUDIT 2: numerical cross-check simulator (numpy) vs fitter (torch)."""
import numpy as np, torch, pandas as pd
from fit_cars2_dcm import Cars2OrdinalDCM

# --- simulator-side reference implementation (copied from cars2_simulator) ---
L, T0 = 6, 3.0
def sig_np(x,p): return 1/(1+(1/p-1)*np.exp(-x/(p*(1-p))))
def probs_np(age,tau,xi,w,p,v,deltas):
    pge=[1.0]
    for level in range(1,L+1):
        cd=sum(deltas[:level]); rt=np.exp(xi)*(age-T0-tau)
        pge.append(sig_np(v*rt - v*cd + w, p))
    pge.append(0.0)
    pge=np.array(pge); pge=np.minimum.accumulate(pge)
    pr=-np.diff(pge); pr=np.clip(pr,1e-9,None); return pr/pr.sum()

# --- fitter-side, with parameters forced to identical values ---
p_true, v_true = 0.35, 0.18
deltas_true = [0.5,0.6,0.55,0.62,0.48,0.53]
tau_t, xi_t, w_t, age_t = 0.4, -0.12, 0.25, 6.3

m = Cars2OrdinalDCM(items=["X"], n_subjects=1, n_sources=1, n_levels=L, t0=T0)
with torch.no_grad():
    m.g[0]        = torch.logit(torch.tensor(p_true))
    m.log_v[0]    = torch.log(torch.expm1(torch.tensor(v_true-1e-3)))
    for j,d in enumerate(deltas_true):
        m.log_delta[0,j] = torch.log(torch.expm1(torch.tensor(d-1e-3)))
    m.tau[0]=tau_t; m.xi[0]=xi_t
    m.A[0,0]=w_t; m.sources[0,0]=1.0   # so w = A@s = w_t

    got = m.level_probs(torch.tensor([0]), torch.tensor([0]),
                        torch.tensor([age_t], dtype=torch.float32)).numpy()[0]

want = probs_np(age_t, tau_t, xi_t, w_t, p_true, v_true, deltas_true)

print("fitter (torch):", np.round(got,6))
print("simulator (np):", np.round(want,6))
print("max abs diff  :", float(np.abs(got-want).max()))
print("MATCH" if np.abs(got-want).max() < 1e-4 else "MISMATCH")
