#!/usr/bin/env python3
# Joint ellc.lc + ellc.rv EDMCMC fit
# Uses ellc.lc for photometry and ellc.rv for spectroscopy in a single joint likelihood.
# Produces trace, corner, transit figure (two-panel), RV figure (two-panel), and CSV of best-fit medians+stds.

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
import ellc
from ellc import lc
import edmcmc as edm

# -------------------------
# USER EDITS (change as needed)
# -------------------------
planet_name = "TOI-5082-b"
model_name = "updated"

# Filepaths
PHOT_CSV = "../data/tic437011608flattened-2min.csv"   # photometry (BJD - 2457000)
PHOT_TIME_COL = "Time (BJD-2457000)"
PHOT_FLUX_COL = " Flattened Flux"

RV_CSV = "../data/NEID_TOI5082_RM_Event202502.csv"   # spectroscopy (JD)
RV_TIME_COL = "ccfjdsum"
RV_RV_COL = "ccfrvmod"
RV_ERR_COL = "dvrms"

output_dir = "./edmcmc_output/"
os.makedirs(output_dir, exist_ok=True)

# Boolean: whether to phase-cut photometry to a small window around expected transit
USE_PHASE_CUT = False      # default: fit entire photometry time array (per your request)
PHASE_CUT_HALF = 0.08     # if USE_PHASE_CUT True, use |phase| < PHASE_CUT_HALF

# Posterior samples to use for model bands plotting (affects runtime & smoothness)
NSAMP_BANDS = 500  # e.g. 500; will be clipped to available samples

# Fonts and plot sizes (you can change)
label_fontsize = 14
tick_fontsize = 12
legend_fontsize = 12
marker_size = 5
model_linewidth = 1.5
band_alpha_1sig = 0.14
band_alpha_2sig = 0.06

# -------------------------
# Fixed model ingredients (from your RM/LC code)
# -------------------------
# Stellar & orbit baseline parameters (kept fixed unless included as free params)
r_1 = 0.0844633938008258    # star radius in units of semi-major axis (fixed)
shape_1 = "sphere"
shape_2 = "sphere"
sbratio = 0.0  # surface brightness ratio
# Fixed limb-darkening (per your instruction to keep fixed)
ldc_1 = [0.1, 0.3]
ld_1 = "quad"
# Eccentricity fixed to 0 per your instruction (no eccentric fit)
e = 0.0
f_c = np.sqrt(e) * math.sin(np.deg2rad(-10))
f_s = np.sqrt(e) * math.sin(np.deg2rad(-10))
q = 0.0000211673084961  # mass ratio

# -------------------------
# Initial guesses (you gave these)
# -------------------------
# Use the RM t0 as the starting (JD)
t0_init = 2460708.82692481   # JD (shared between photometry and RV)
period_init = 4.240122        # days
radius2_init = 0.0021193441472191  # planet radius in units of a (radius_2)
a_init = 11.01068755603       # semi-major axis (same units you've been using)
incl_init = 90.0              # degrees
vsini_init = 1.5              # km/s
lambda_init = 45.0            # degrees

# -------------------------
# Load data
# -------------------------
# Photometry (phot file times are BJD - 2457000, convert to JD by adding 2457000)
phot_df = pd.read_csv(PHOT_CSV)
if PHOT_TIME_COL not in phot_df.columns or PHOT_FLUX_COL not in phot_df.columns:
    print(f"Expected columns: {PHOT_TIME_COL}, {PHOT_FLUX_COL}")
    print(f"Available columns: {list(phot_df.columns)}")
    raise ValueError("Photometry CSV is missing expected columns.")
phot_time_bjd_minus = phot_df[PHOT_TIME_COL].values.astype(float)
phot_time = phot_time_bjd_minus + 2457000.0       # convert to JD
phot_flux = phot_df[PHOT_FLUX_COL].values.astype(float)

# Spectroscopy
rv_df = pd.read_csv(RV_CSV, comment="#")
for col in (RV_TIME_COL, RV_RV_COL, RV_ERR_COL):
    if col not in rv_df.columns:
        raise ValueError(f"RV CSV missing column: {col}")
time_obs = rv_df[RV_TIME_COL].values.astype(float)   # JD
rv_data = rv_df[RV_RV_COL].values.astype(float)      # km/s
rv_err = rv_df[RV_ERR_COL].values.astype(float)      # km/s

# -------------------------
# Photometric noise estimate (single yerr array) — computed once using initial t0 guess
# We compute out-of-transit std using an initial epoch to approximate the noise level.
# This follows your prior approach (no jitter).
# -------------------------
# Compute phases relative to initial t0 (in days)
phot_phase_init = ((phot_time - (t0_init)) / period_init + 0.5) % 1.0 - 0.5
# mask out central transit region for noise estimate
phot_oof_mask = np.abs(phot_phase_init) > 0.02   # exclude ±0.02 by default (rough)
if phot_oof_mask.sum() < 3:
    phot_oof_mask[:] = True
phot_noise = np.nanstd(phot_flux[phot_oof_mask])
phot_yerr = np.full_like(phot_flux, phot_noise)

# Optionally create phase-cut arrays (will be used later if USE_PHASE_CUT True)
phot_phase = ((phot_time - t0_init) / period_init + 0.5) % 1.0 - 0.5

# -------------------------
# EDMCMC: define joint log-likelihood (photometry + RV)
# Free parameters (in order): t0, period, radius_2, a, incl, vsini, lambda
# -------------------------
def loglikelihood_joint_debug(p, phot_t, phot_f, phot_err, rv_t, rv_obs, rv_e):
    """
    Diagnostic joint log-likelihood for ellc.lc + ellc.rv with extensive prints and try-blocks.
    Returns log-likelihood (phot + rv) or -np.inf on error.
    Use this in place of loglikelihood_joint when debugging.
    """
    import numpy as np
    import time
    import traceback
    import sys

    def now(): return time.time()

    def report_array(name, a):
        try:
            a_np = np.asarray(a)
            contig = a_np.flags['C_CONTIGUOUS']
            print(f"  {name}: dtype={a_np.dtype}, shape={a_np.shape}, contiguous={contig}", flush=True)
            if a_np.size <= 10:
                print(f"    values: {a_np}", flush=True)
            else:
                print(f"    sample start: {a_np.flat[:3]}, end: {a_np.flat[-3:]}", flush=True)
            if np.any(np.isnan(a_np)):
                print(f"    WARNING: {name} contains NaN(s)", flush=True)
            if np.any(np.isinf(a_np)):
                print(f"    WARNING: {name} contains Inf(s)", flush=True)
        except Exception as ex:
            print(f"  report_array({name}) failed: {ex}", flush=True)

    # small helper tests
    def small_lc_test():
        try:
            t_test = np.linspace(0.0, 0.01, 5)
            f_test = ellc.lc(
                t_obs=t_test,
                radius_1=0.1,
                radius_2=0.001,
                sbratio=0.0,
                incl=89.0,
                t_zero=t_test.mean(),
                period=10.0,
                a=10.0,
                ldc_1=[0.1, 0.3],
                f_c=0.0,
                f_s=0.0,
                ld_1="quad",
                shape_1="sphere",
                shape_2="sphere",
                verbose=0,
            )
            print("  small_lc_test succeeded. out.shape=", np.shape(f_test), flush=True)
            return True
        except Exception as ex:
            print("  small_lc_test FAILED:", repr(ex), flush=True)
            print(traceback.format_exc(), flush=True)
            return False

    def small_rv_test():
        try:
            t_test = np.linspace(0.0, 0.01, 5)
            rv_call = ellc.rv(
                t_test,
                t_zero=t_test.mean(),
                period=10.0,
                lambda_1=0.0,
                radius_1=0.1,
                radius_2=0.001,
                incl=89.0,
                a=10.0,
                f_c=0.0,
                f_s=0.0,
                q=0.001,
                shape_1="sphere",
                shape_2="sphere",
                vsini_1=1.0,
                vsini_2=0.0,
                flux_weighted=True,
                verbose=0,
            )
            print("  small_rv_test succeeded. type=", type(rv_call), "len?=", getattr(rv_call, "__len__", None),
                  flush=True)
            return True
        except Exception as ex:
            print("  small_rv_test FAILED:", repr(ex), flush=True)
            print(traceback.format_exc(), flush=True)
            return False

    # Begin diagnostic sequence
    start_time = now()
    print("\n--- DEBUG loglikelihood_joint_debug ENTRY ---", flush=True)
    print("  Received parameter vector p (len={}):".format(len(p)), p, flush=True)

    # Basic sanity checks on p
    try:
        if len(p) != 7:
            print(f"  WARNING: expected 7 parameters [t0, period, radius2, a, incl, vsini, lambda], got {len(p)}", flush=True)
    except Exception:
        pass

    # Unpack but robustly (guard against too-short lists)
    try:
        t0_mod = float(p[0])
        period_mod = float(p[1])
        radius2_mod = float(p[2])
        a_mod = float(p[3])
        incl_mod = float(p[4])
        vsini_mod = float(p[5])
        lambda_mod = float(p[6])
    except Exception as ex:
        print("  ERROR unpacking p:", repr(ex), flush=True)
        print(traceback.format_exc(), flush=True)
        return -np.inf

    # Quick validity checks on parameter values
    try:
        if not (0.0 < period_mod < 1e5):
            print(f"  WARNING: period_mod out of expected range: {period_mod}", flush=True)
        if radius2_mod <= 0 or radius2_mod > 1.0:
            print(f"  WARNING: radius2_mod suspicious: {radius2_mod}", flush=True)
        if a_mod <= 0:
            print(f"  WARNING: a_mod suspicious: {a_mod}", flush=True)
        if not (0.0 <= incl_mod <= 180.0):
            print(f"  WARNING: incl_mod out of physical range: {incl_mod}", flush=True)
        if not (0.0 <= vsini_mod <= 100.0):
            print(f"  WARNING: vsini_mod suspicious: {vsini_mod}", flush=True)
        if not (0.0 <= lambda_mod <= 360.0):
            print(f"  WARNING: lambda_mod suspicious: {lambda_mod}", flush=True)
    except Exception as ex:
        print("  ERROR during validity checks:", repr(ex), flush=True)

    # Report arrays that will be passed
    print("  Inspecting photometry arrays (phot_t, phot_f, phot_err):", flush=True)
    report_array("phot_t", phot_t)
    report_array("phot_f", phot_f)
    report_array("phot_err", phot_err)

    print("  Inspecting RV arrays (rv_t, rv_obs, rv_e):", flush=True)
    report_array("rv_t", rv_t)
    report_array("rv_obs", rv_obs)
    report_array("rv_e", rv_e)

    # Run small tests to validate ellc functions on this environment before the big calls
    print("  Running small ellc.lc test...", flush=True)
    if not small_lc_test():
        # if the small test fails, bail out early
        print("  Aborting: small ellc.lc test failed.", flush=True)
        return -np.inf

    print("  Running small ellc.rv test...", flush=True)
    if not small_rv_test():
        print("  Aborting: small ellc.rv test failed.", flush=True)
        return -np.inf

    # Try calling ellc.lc on the full photometry (catch and report errors)
    t0_call = now()
    try:
        # Ensure contiguous arrays & correct dtypes before call
        phot_t_c = np.ascontiguousarray(phot_t, dtype=np.float64)
        phot_f_c = np.ascontiguousarray(phot_f, dtype=np.float64)
        phot_err_c = np.ascontiguousarray(phot_err, dtype=np.float64)

        print("  Calling ellc.lc(...) on photometry (this may take a moment)...", flush=True)
        lc_start = now()
        flux_model = ellc.lc(
            t_obs=phot_t_c,
            radius_1=r_1,
            radius_2=radius2_mod,
            sbratio=sbratio,
            incl=incl_mod,
            t_zero=t0_mod,
            period=period_mod,
            a=a_mod,
            ldc_1=ldc_1,
            f_c=f_c,
            f_s=f_s,
            ld_1=ld_1,
            shape_1=shape_1,
            shape_2=shape_2,
            verbose=0,
        )
        lc_elapsed = now() - lc_start
        print(f"  ellc.lc returned shape {np.shape(flux_model)} (took {lc_elapsed:.3f} s).", flush=True)
    except Exception as ex:
        print("  ellc.lc call FAILED:", repr(ex), flush=True)
        print(traceback.format_exc(), flush=True)
        print("  Attempting defensive conversions & retry...", flush=True)
        try:
            phot_t_c = np.ascontiguousarray(phot_t, dtype=np.float64)
            flux_model = ellc.lc(
                t_obs=phot_t_c,
                radius_1=float(r_1),
                radius_2=float(radius2_mod),
                sbratio=float(sbratio),
                incl=float(incl_mod),
                t_zero=float(t0_mod),
                period=float(period_mod),
                a=float(a_mod),
                ldc_1=list(ldc_1),
                f_c=float(f_c),
                f_s=float(f_s),
                ld_1=str(ld_1),
                shape_1=str(shape_1),
                shape_2=str(shape_2),
                verbose=0,
            )
            print("  Retry ellc.lc succeeded.", flush=True)
        except Exception as ex2:
            print("  Retry ellc.lc FAILED:", repr(ex2), flush=True)
            print(traceback.format_exc(), flush=True)
            return -np.inf

    # Check model length matches phot_t length
    try:
        if np.shape(flux_model)[0] != np.shape(phot_t_c)[0]:
            print("  ERROR: flux_model length != phot_t length:", np.shape(flux_model), np.shape(phot_t_c), flush=True)
            return -np.inf
    except Exception as ex:
        print("  ERROR checking flux_model shape:", repr(ex), flush=True)
        print(traceback.format_exc(), flush=True)
        return -np.inf

    # Photometry likelihood (optionally use phase cut)
    try:
        if USE_PHASE_CUT:
            phase = ((phot_t_c - t0_mod) / period_mod + 0.5) % 1.0 - 0.5
            mask = np.abs(phase) <= PHASE_CUT_HALF
            if mask.sum() < 3:
                mask = np.ones_like(mask, dtype=bool)
            y = phot_f_c[mask]
            m = flux_model[mask]
            s = phot_err_c[mask]
            print(f"  Using phase cut: kept {mask.sum()} photometric points", flush=True)
        else:
            y = phot_f_c
            m = flux_model
            s = phot_err_c
    except Exception as ex:
        print("  ERROR preparing photometry likelihood:", repr(ex), flush=True)
        print(traceback.format_exc(), flush=True)
        return -np.inf

    # Compute chi2 photometric
    try:
        s_safe = np.where(s <= 0, np.median(s[s > 0]) if np.any(s > 0) else 1e-6, s)
        chi2_phot = np.sum((y - m) ** 2 / (s_safe ** 2))
        lnlike_phot = -0.5 * chi2_phot
        print(f"  Photometry: chi2 = {chi2_phot:.4g}, lnlike_phot = {lnlike_phot:.4g}", flush=True)
    except Exception as ex:
        print("  ERROR computing photometric chi2:", repr(ex), flush=True)
        print(traceback.format_exc(), flush=True)
        return -np.inf

    # RV model call
    try:
        rv_t_c = np.ascontiguousarray(rv_t, dtype=np.float64)
        rv_obs_c = np.ascontiguousarray(rv_obs, dtype=np.float64)
        rv_e_c = np.ascontiguousarray(rv_e, dtype=np.float64)

        print("  Calling ellc.rv(...) on spectroscopy...", flush=True)
        rv_start = now()
        rv_call = ellc.rv(
            rv_t_c,
            t_zero=t0_mod,
            period=period_mod,
            lambda_1=lambda_mod,
            radius_1=r_1,
            radius_2=radius2_mod,
            incl=incl_mod,
            a=a_mod,
            f_c=f_c,
            f_s=f_s,
            q=q,
            shape_1=shape_1,
            shape_2=shape_2,
            vsini_1=vsini_mod,
            vsini_2=0.0,
            flux_weighted=True,
            sbratio=sbratio,
            verbose=0,
        )
        rv_elapsed = now() - rv_start
        print(f"  ellc.rv returned (type={type(rv_call)}) (took {rv_elapsed:.3f} s).", flush=True)
    except Exception as ex:
        print("  ellc.rv call FAILED:", repr(ex), flush=True)
        print(traceback.format_exc(), flush=True)
        return -np.inf

    # Extract star_rv_model (first element of tuple)
    try:
        star_rv_model = np.asarray(rv_call[0])
        print(f"  Extracted star_rv_model with shape {star_rv_model.shape}", flush=True)
    except Exception as ex:
        print("  ERROR extracting star_rv_model:", repr(ex), flush=True)
        print(traceback.format_exc(), flush=True)
        return -np.inf

    # Align model and compute RV chi2
    try:
        sys_offset = np.median(rv_obs_c) - np.median(star_rv_model)
        star_rv_model = star_rv_model + sys_offset
        rv_e_safe = np.where(rv_e_c <= 0, np.median(rv_e_c[rv_e_c > 0]) if np.any(rv_e_c > 0) else 1e-6, rv_e_c)
        chi2_rv = np.sum((rv_obs_c - star_rv_model) ** 2 / (rv_e_safe ** 2))
        lnlike_rv = -0.5 * chi2_rv
        print(f"  RV: chi2 = {chi2_rv:.4g}, lnlike_rv = {lnlike_rv:.4g}", flush=True)
        print(f"  rv med data = {np.median(rv_obs_c):.6g}, rv med model(before offset) = {np.median(rv_call[0]):.6g}, sys_offset={sys_offset:.6g}", flush=True)
    except Exception as ex:
        print("  ERROR computing RV chi2 or aligning model:", repr(ex), flush=True)
        print(traceback.format_exc(), flush=True)
        return -np.inf

    # Sum likelihoods and return
    try:
        lnlike_total = lnlike_phot + lnlike_rv
        elapsed = now() - start_time
        print(f"  lnlike_total = {lnlike_total:.6g} (elapsed {elapsed:.3f} s) -- returning value.", flush=True)
        print("--- DEBUG loglikelihood_joint_debug EXIT ---\n", flush=True)
        return lnlike_total
    except Exception as ex:
        print("  ERROR summing likelihoods:", repr(ex), flush=True)
        print(traceback.format_exc(), flush=True)
        return -np.inf


# -------------------------
# Prepare EDMCMC inputs: initial p0, wid, parinfo (priors/limits)
# Order: [t0, period, radius2, a, incl, vsini, lambda]
# -------------------------
p0 = [
    t0_init,
    period_init,
    radius2_init,
    a_init,
    incl_init,
    vsini_init,
    lambda_init,
]

# proposal widths - sensible starting values
wid = [
    1e-4,    # t0 [days]
    1e-4,    # period [days]
    1e-6,    # radius2 (in units of a)
    0.01,    # a (same units you used)
    0.01,    # incl [deg]
    0.1,     # vsini [km/s]
    1.0,     # lambda [deg]
]

# parinfo: hard limits using EDMCMC format (limited True=apply)
parinfo = [
    {'fixed': False, 'limits': [p0[0] - 0.5, p0[0] + 0.5], 'limited': [True, True]},   # t0 ±0.5 days
    {'fixed': False, 'limits': [p0[1] - 0.1, p0[1] + 0.1], 'limited': [True, True]},   # period ±0.1 days
    {'fixed': False, 'limits': [1e-6, 0.10], 'limited': [True, True]},               # radius2 [1e-6, 0.1]
    {'fixed': False, 'limits': [5.0, 50.0], 'limited': [True, True]},                # a bounding (sensible)
    {'fixed': False, 'limits': [80.0, 90.0], 'limited': [True, True]},               # incl [80,90]
    {'fixed': False, 'limits': [0.0, 10.0], 'limited': [True, True]},                # vsini [0,20] km/s
    {'fixed': False, 'limits': [0.0, 90.0], 'limited': [True, True]},                # lambda [0,90] per your request
]

# sampler settings
ndim = len(p0)
nwalkers = 1 * ndim   # you requested 5x number of free params
nlink = 1
nburnin = 0
ncores = 1

print("[INFO] Running joint EDMCMC with ndim =", ndim, "nwalkers =", nwalkers)

# -------------------------
# Run EDMCMC
# -------------------------
out = edm.edmcmc(
    loglikelihood_joint_debug,
    p0,
    wid,
    args=(phot_time, phot_flux, phot_yerr, time_obs, rv_data, rv_err),
    parinfo=parinfo,
    nwalkers=nwalkers,
    nlink=nlink,
    nburnin=nburnin,
    ncores=ncores,
    quiet=False
)

# -------------------------
# Output & save parameter summaries
# -------------------------
params_names = ['t0', 'period', 'radius2', 'a', 'incl', 'vsini', 'lambda']
meds = np.median(out.flatchains, axis=0)
stds = np.std(out.flatchains, axis=0)

for name, med, std in zip(params_names, meds, stds):
    print(f"{name}: {med:.6g} +/- {std:.6g}")

# save CSV of medians+stds
df_params = pd.DataFrame({
    'parameter': params_names,
    'median': meds,
    'std': stds
})
csv_params = os.path.join(output_dir, f"{planet_name}_{model_name}_bestfit_params.csv")
df_params.to_csv(csv_params, index=False)
print("[INFO] Saved best-fit parameter CSV:", csv_params)

# -------------------------
# Trace plot
# -------------------------
fig1, axes1 = plt.subplots(ndim, figsize=(10, 1 + 2 * ndim), sharex=True)
for i in range(ndim):
    ax = axes1[i]
    ax.plot(out.whichlink, out.flatchains[:, i], '.')
    ax.set_ylabel(params_names[i])
axes1[-1].set_xlabel("Link number")
fig1_name = os.path.join(output_dir, f"{planet_name}_{model_name}_trace.pdf")
fig1.savefig(fig1_name)
plt.close(fig1)
print("[INFO] Trace plot saved:", fig1_name)

# -------------------------
# Corner plot
# -------------------------
fig2 = corner.corner(out.flatchains, labels=params_names)
fig2_name = os.path.join(output_dir, f"{planet_name}_{model_name}_corner.pdf")
fig2.savefig(fig2_name)
plt.close(fig2)
print("[INFO] Corner plot saved:", fig2_name)

# -------------------------
# Compute posterior model bands (photometry and RV) using a subset of posterior samples
# -------------------------
all_samples = out.flatchains
nsamples_total = all_samples.shape[0]
nsamp = min(NSAMP_BANDS, nsamples_total)
# Use stratified selection for even coverage:
sel_idx = np.linspace(0, nsamples_total - 1, nsamp, dtype=int)

# Pre-allocate arrays
ntime_phot = len(phot_time)
ntime_rv = len(time_obs)
phot_models = np.zeros((nsamp, ntime_phot))
rv_models = np.zeros((nsamp, ntime_rv))

for j, idx in enumerate(sel_idx):
    samp = all_samples[idx, :]
    t0_s, per_s, r2_s, a_s, inc_s, vs_s, lam_s = samp

    # phot model
    phot_m = ellc.lc(
        t_obs=phot_time,
        radius_1=r_1,
        radius_2=r2_s,
        sbratio=sbratio,
        incl=inc_s,
        t_zero=t0_s,
        period=per_s,
        a=a_s,
        ldc_1=ldc_1,
        f_c=f_c,
        f_s=f_s,
        ld_1=ld_1,
        shape_1=shape_1,
        shape_2=shape_2,
        verbose=0,
    )
    phot_models[j, :] = phot_m

    # rv model
    rv_call_s = ellc.rv(
        time_obs,
        t_zero=t0_s,
        period=per_s,
        lambda_1=lam_s,
        radius_1=r_1,
        radius_2=r2_s,
        incl=inc_s,
        a=a_s,
        f_c=f_c,
        f_s=f_s,
        q=q,
        shape_1=shape_1,
        shape_2=shape_2,
        vsini_1=vs_s,
        vsini_2=0.0,
        flux_weighted=True,
        sbratio=sbratio,
        verbose=0,
    )
    rv_models[j, :] = np.asarray(rv_call_s[0])
    # align each sampled RV model to data zeropoint
    rv_models[j, :] += (np.median(rv_data) - np.median(rv_models[j, :]))

# Compute percentiles for photometry & RV
phot_med = np.median(phot_models, axis=0)
phot_p16 = np.percentile(phot_models, 16, axis=0)
phot_p84 = np.percentile(phot_models, 84, axis=0)
phot_p025 = np.percentile(phot_models, 2.5, axis=0)
phot_p975 = np.percentile(phot_models, 97.5, axis=0)

rv_med = np.median(rv_models, axis=0)
rv_p16 = np.percentile(rv_models, 16, axis=0)
rv_p84 = np.percentile(rv_models, 84, axis=0)
rv_p025 = np.percentile(rv_models, 2.5, axis=0)
rv_p975 = np.percentile(rv_models, 97.5, axis=0)

# Best-fit (median-parameter) models
best = meds
t0_best, per_best, r2_best, a_best, inc_best, vs_best, lam_best = best

phot_best = ellc.lc(
    t_obs=phot_time,
    radius_1=r_1,
    radius_2=r2_best,
    sbratio=sbratio,
    incl=inc_best,
    t_zero=t0_best,
    period=per_best,
    a=a_best,
    ldc_1=ldc_1,
    f_c=f_c,
    f_s=f_s,
    ld_1=ld_1,
    shape_1=shape_1,
    shape_2=shape_2,
    verbose=0,
)
rv_call_best = ellc.rv(
    time_obs,
    t_zero=t0_best,
    period=per_best,
    lambda_1=lam_best,
    radius_1=r_1,
    radius_2=r2_best,
    incl=inc_best,
    a=a_best,
    f_c=f_c,
    f_s=f_s,
    q=q,
    shape_1=shape_1,
    shape_2=shape_2,
    vsini_1=vs_best,
    vsini_2=0.0,
    flux_weighted=True,
    sbratio=sbratio,
    verbose=0,
)
rv_best = np.asarray(rv_call_best[0])
rv_best += (np.median(rv_data) - np.median(rv_best))  # align zeropoint

# -------------------------
# Save transit plot (two-panel: data+median-model+bands, residuals)
# -------------------------
fig_phot, (ax_p_top, ax_p_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}
)

# Top: bands
ax_p_top.fill_between(phot_time, phot_p025, phot_p975, color='tab:blue', alpha=0.06)
ax_p_top.fill_between(phot_time, phot_p16, phot_p84, color='tab:blue', alpha=0.14)

# Data and best-fit
ax_p_top.errorbar(phot_time, phot_flux, yerr=phot_yerr, fmt='o', ms=marker_size, c='k', zorder=3, label='phot data')
ax_p_top.plot(phot_time, phot_best, '-', lw=model_linewidth, color='tab:blue', label='median-parameter model')

ax_p_top.tick_params(labelsize=tick_fontsize)
ax_p_top.legend(prop={'size': legend_fontsize}, loc='best')
# remove individual y labels; place shared label below
ax_p_top.set_ylabel('Normalized flux', fontsize=label_fontsize)

# Residuals to pointwise median model
resid_phot_ms = (phot_flux - phot_med)  # flux units
ax_p_bot.errorbar(phot_time, resid_phot_ms, yerr=phot_yerr, fmt='o', ms=marker_size, c='k')
ax_p_bot.axhline(0.0, color='tab:blue', linestyle='--', alpha=0.7)
ax_p_bot.set_xlabel('Time (JD)', fontsize=label_fontsize)
ax_p_bot.set_ylabel('Flux residual', fontsize=label_fontsize)
ax_p_bot.tick_params(labelsize=tick_fontsize)

fig_phot_name = os.path.join(output_dir, f"{planet_name}_{model_name}_transit.pdf")
fig_phot.savefig(fig_phot_name)
plt.close(fig_phot)
print("[INFO] Transit plot saved:", fig_phot_name)

# -------------------------
# Save RV plot (two-panel: data+median-model+bands, residuals)
# -------------------------
fig_rv, (ax_r_top, ax_r_bot) = plt.subplots(
    2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}
)

# Plot RV bands and best-fit (convert to m/s for plotting)
ax_r_top.fill_between(time_obs, rv_p025 * 1e3, rv_p975 * 1e3, color='red', alpha=band_alpha_2sig)
ax_r_top.fill_between(time_obs, rv_p16 * 1e3, rv_p84 * 1e3, color='red', alpha=band_alpha_1sig)

ax_r_top.errorbar(time_obs, rv_data * 1e3, yerr=rv_err * 1e3, fmt='o', ms=marker_size, c='k', label='RV data', zorder=5)
ax_r_top.plot(time_obs, rv_best * 1e3, '-', lw=model_linewidth, alpha=1.0, c='red', label='median-parameter model')

ax_r_top.set_ylabel('Radial velocity (m/s)', fontsize=label_fontsize)
ax_r_top.tick_params(labelsize=tick_fontsize)
ax_r_top.legend(prop={'size': legend_fontsize}, loc='best')

# Residuals relative to pointwise median model
resid_rv_ms = (rv_data - rv_med) * 1e3
ax_r_bot.errorbar(time_obs, resid_rv_ms, yerr=rv_err * 1e3, fmt='o', ms=marker_size, c='k')
ax_r_bot.axhline(0.0, color='red', linestyle='--', alpha=0.7)
ax_r_bot.set_xlabel('Time (JD)', fontsize=label_fontsize)
ax_r_bot.set_ylabel('Radial velocity (m/s)', fontsize=label_fontsize)
ax_r_bot.tick_params(labelsize=tick_fontsize)

fig_rv_name = os.path.join(output_dir, f"{planet_name}_{model_name}_rv_model.pdf")
fig_rv.savefig(fig_rv_name)
plt.close(fig_rv)
print("[INFO] RV plot saved:", fig_rv_name)

# Done
print("[INFO] Joint fit complete. All outputs saved in", output_dir)
