#!/usr/bin/env python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ellc

parser = argparse.ArgumentParser()
parser.add_argument("--eps", help="generate .eps file", action="store_true")
parser.add_argument("--csv", help="path to input CSV (default ../data/HD97658_2025Feb20.csv)",
                    default="../data/HD97658_2025Feb20.csv")
args = parser.parse_args()

if args.eps:
    import matplotlib
    matplotlib.use('Agg')

# -------------------------
# Small helper to handle ellc return types
# -------------------------
def as_array(x):
    """
    ellc.rv (and other ellc routines) may return a single array or a tuple.
    If it's a tuple/list, this helper returns the first element as an ndarray.
    Otherwise it returns the input as an ndarray.
    """
    if isinstance(x, (tuple, list)):
        return np.asarray(x[0])
    return np.asarray(x)

# -------------------------
# User / model parameters
# -------------------------
csvfile = args.csv

# Stellar / planet physical params (example values; change these)
M_1 = 0.74         # stellar mass (solar masses) - tune as needed
M_2 = 0.02         # companion mass (solar masses) - small for a planet
R_1 = 0.74         # stellar radius (solar radii) - example
R_2 = 0.0148       # planetary radius (solar radii) - example
period = 8.0       # days (example)
t0 = 2460000.0     # epoch (same time system as ccfjdsum). adjust to your transit epoch
incl = 89.0        # degrees
vsini_1 = 3.0      # km/s (projected stellar rotation) — change as needed
lambda_1 = 0.0     # sky-projected obliquity in degrees (0 = aligned)
ld_1 = 'lin'       # limb darkening law
ldc_1 = 0.4        # limb-darkening coefficient (if linear)
sbratio = 0.0      # for planet-star, set to 0 (planet contributes negligible light)

# shape parameters
shape_1 = 'sphere'
shape_2 = 'sphere'

# optional plotting settings
phase_window = 0.1   # phase window (in phase units) used to mask in-transit when fitting the orbital trend

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(csvfile, comment='#')
# ensure columns exist
for col in ("ccfjdsum", "ccfrvmod", "dvrms"):
    if col not in df.columns:
        raise ValueError("Input CSV missing required column: {}".format(col))

t_obs = df['ccfjdsum'].values.astype(float)      # times in days (user-provided)
rv_obs = df['ccfrvmod'].values.astype(float)    # observed RV in km/s
rv_err = df['dvrms'].values.astype(float)        # RV uncertainties in km/s

# -------------------------
# Build ellc model
# -------------------------
a = 4.20944009361 * period**(2./3.) * (M_1 + M_2)**(1./3.)
r_1 = R_1 / a
r_2 = R_2 / a
q = M_2 / M_1

# compute "full" model including vsini (gives orbital + RM)
_model_full_raw = ellc.rv(t_obs,
                          radius_1=r_1, radius_2=r_2, sbratio=sbratio,
                          incl=incl,
                          t_zero=t0, period=period, a=a, q=q,
                          ld_1=ld_1, ldc_1=ldc_1,
                          vsini_1=vsini_1, lambda_1=lambda_1,
                          shape_1=shape_1, shape_2=shape_2)
model_rv_full = as_array(_model_full_raw)

# compute the same model but with vsini=0 -> gives the pure orbital motion (no RM)
_model_no_vsini_raw = ellc.rv(t_obs,
                              radius_1=r_1, radius_2=r_2, sbratio=sbratio,
                              incl=incl,
                              t_zero=t0, period=period, a=a, q=q,
                              ld_1=ld_1, ldc_1=ldc_1,
                              vsini_1=0.0, lambda_1=lambda_1,
                              shape_1=shape_1, shape_2=shape_2)
model_rv_no_vsini = as_array(_model_no_vsini_raw)

# RM anomaly (in same units as RVs, i.e. km/s)
rm_model = model_rv_full - model_rv_no_vsini

# Shift model to match observed systemic velocity (simple median offset)
sys_offset = np.median(rv_obs) - np.median(model_rv_full)
model_rv_full_shifted = model_rv_full + sys_offset
model_rv_no_vsini_shifted = model_rv_no_vsini + sys_offset
rm_model_shifted = model_rv_full_shifted - model_rv_no_vsini_shifted

# -------------------------
# Dense grid for plotting models
# -------------------------
phases_grid = np.linspace(-0.5, 0.5, 2000)
t_grid = t0 + phases_grid * period

_model_full_grid_raw = ellc.rv(t_grid,
                               radius_1=r_1, radius_2=r_2, sbratio=sbratio,
                               incl=incl, t_zero=t0, period=period, a=a, q=q,
                               ld_1=ld_1, ldc_1=ldc_1,
                               vsini_1=vsini_1, lambda_1=lambda_1,
                               shape_1=shape_1, shape_2=shape_2)
model_rv_full_grid = as_array(_model_full_grid_raw) + sys_offset

_model_no_vsini_grid_raw = ellc.rv(t_grid,
                                   radius_1=r_1, radius_2=r_2, sbratio=sbratio,
                                   incl=incl, t_zero=t0, period=period, a=a, q=q,
                                   ld_1=ld_1, ldc_1=ldc_1,
                                   vsini_1=0.0, lambda_1=lambda_1,
                                   shape_1=shape_1, shape_2=shape_2)
model_rv_no_vsini_grid = as_array(_model_no_vsini_grid_raw) + sys_offset

rm_model_grid = model_rv_full_grid - model_rv_no_vsini_grid

# -------------------------
# Fold times to phase for plotting
# -------------------------
phases = ((t_obs - t0) / period)
phases = (phases + 0.5) % 1.0 - 0.5  # map to [-0.5, +0.5]

# -------------------------
# Create a simple 'observed RM' by removing a linear fit to the out-of-transit RVs
# -------------------------
in_transit_mask = np.abs(phases) < phase_window
out_mask = ~in_transit_mask
if out_mask.sum() < 3:
    out_mask = np.ones_like(out_mask, dtype=bool)

coeff = np.polyfit(t_obs[out_mask], rv_obs[out_mask], deg=1)
rv_trend = np.polyval(coeff, t_obs)
rv_obs_detrended = (rv_obs - rv_trend) * 1e3   # convert to m/s for plotting
rv_err_ms = rv_err * 1e3

rm_model_obs_ms = (rm_model_shifted) * 1e3
rm_model_grid_ms = rm_model_grid * 1e3

# -------------------------
# Plotting
# -------------------------
fig = plt.figure(1, figsize=(8, 8))

ax1 = plt.subplot(211)
ax1.errorbar(phases, rv_obs, yerr=rv_err, fmt='o', ms=5, label='NEID RVs', zorder=5)
ax1.plot(phases_grid, model_rv_full_grid, '-', lw=1.5, label='ellc model (orbital + RM)')
ax1.plot(phases_grid, model_rv_no_vsini_grid, '--', lw=1.2, label='ellc model (orbital only)')
ax1.set_xlim(-0.2, 0.2)
ax1.set_xlabel('Phase')
ax1.set_ylabel('Radial velocity (km/s)')
ax1.legend(loc='best')
# ax1.set_title('Observed RVs + ellc model (shifted by median offset)')

ax2 = plt.subplot(223)
ax2.errorbar(phases, rv_obs_detrended, yerr=rv_err_ms, fmt='o', ms=5, label='observed (detrended)', zorder=5)
ax2.plot(phases, rm_model_obs_ms, 'o', ms=3, label='RM model (obs times)', alpha=0.8)
ax2.plot(phases_grid, rm_model_grid_ms, '-', lw=1.5, label='RM model (dense grid)')
ax2.set_xlim(-0.05, 0.05)
ax2.set_ylim(-200, 200)   # m/s; adjust as needed
ax2.set_xlabel('Phase')
ax2.set_ylabel('RM anomaly (m/s)')
ax2.legend(loc='best')
# ax2.set_title('RM anomaly: data (detrended) vs ellc RM model')

ax3 = plt.subplot(224)
ax3.errorbar(phases, rv_obs - np.median(rv_obs), yerr=rv_err, fmt='o', ms=5, label='RV - median', zorder=5)
ax3.plot(phases_grid, model_rv_full_grid - np.median(model_rv_full_grid), '-', lw=1.5, label='model - median')
ax3.set_xlim(0.47, 0.53)
ax3.set_xlabel('Phase')
ax3.set_ylabel('Radial velocity difference (km/s)')
ax3.legend(loc='best')

plt.tight_layout()

if args.eps:
    fig.savefig("RM_singleplanet_overlay_fixed.eps", dpi=400)
else:
    plt.show()
