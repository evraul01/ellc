# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import corner
import ellc
from ellc import lc

import edmcmc as edm
# import batman


# %%
# System parameters
r_1 = 0.04132231404
r_2 = 0.0011228161490683
incl = 89.45
a = 24.2
e = 0.05
f_c = np.sqrt(e) * math.cos(np.deg2rad(-10))
f_s = np.sqrt(e) * math.sin(np.deg2rad(-10))
q = 0.0000291991764706

# %%
t0_guess = 2460708.91005806
period_guess = 9.489
lambda1_guess = 0

planet_name = "HD-97658-b"
model_name = "initial"
csvfile = "../data/HD97658_2025Feb20.csv"
output_dir = "./edmcmc_output/"

df = pd.read_csv(csvfile, comment='#')
# ensure columns exist
for col in ("ccfjdsum", "ccfrvmod", "dvrms"):
    if col not in df.columns:
        raise ValueError(f"Input CSV missing required column: {col}")

time_obs = df['ccfjdsum'].values.astype(float)  # times in days
rv_data = df['ccfrvmod'].values.astype(float)   # observed RV in km/s
rv_err = df['dvrms'].values.astype(float)       # RV uncertainties in km/s


# %%
def loglikelihood(p, time, rv_obs, rv_err):
    t0_mod, period_mod, lambda1_mod = p  # free params

    rv_model, _ = ellc.rv(
        time,
        t_zero=t0_mod,
        period=period_mod,
        lambda_1=lambda1_mod,
        radius_1=r_1,
        radius_2=r_2,
        incl=incl,
        a=a,
        f_c=f_c,
        f_s=f_s,
        q=q
    )

    chi2 = np.sum((rv_obs - rv_model)**2 / (rv_err)**2)
    return -0.5 * chi2


# initial guess and step size
labels = ['time', 'period', 'lambda']
p0 = [t0_guess, period_guess, lambda1_guess]  # initial params: t0, period, lambda1
wid = [0.000001, 0.000001, 0.0000001]         # step sizes
ndim = len(p0)

# Run the MCMC
out = edm.edmcmc(
    loglikelihood,
    p0,
    wid,
    args=(time_obs, rv_data, rv_err),  # observed data
    nwalkers=15,                       # usually 2-4x number of free params
    nlink=1000,
    nburnin=80,
    ncores=4                         # number of cores
)

print(np.median(out.flatchains[:,0]), '+/-', np.std(out.flatchains[:,0]), ';    ', np.median(out.flatchains[:,1]), '+/-', np.std(out.flatchains[:,1]))
print(type(out))

fig1, axes1 = plt.subplots(ndim, figsize=(10, 1+2*ndim), sharex=True)
for i in range(ndim):
            ax = axes1[i]
            ax.plot(out.whichlink, out.flatchains[:,i], '.')
            # ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            # ax.yaxis.set_label_coords(-0.1, 0.5)
axes1[-1].set_xlabel("Link number")
fig1_name = output_dir+planet_name+'_'+model_name+'_trace.pdf'
fig1.savefig(fig1_name)
print('walker trace plot:'+fig1_name)
plt.close(fig1)

fig2 = plt.figure(figsize=(1+3*ndim,1+3*ndim))
fig2 = corner.corner(out.flatchains,labels=labels)
fig2_name = output_dir+planet_name+'_'+model_name+'_corner.pdf'
fig2.savefig(fig2_name)
print('corner plot:'+fig2_name)
plt.close(fig2)