# %%
# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import sys
from pathlib import Path

import batman
import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _bootstrap_paths(start_path: Path) -> None:
    for candidate in [start_path, *start_path.parents]:
        run_candidate = candidate if (candidate / "edmcmc.py").exists() else candidate / "run"
        if (run_candidate / "edmcmc.py").exists():
            repo_candidate = run_candidate.parent
            if str(run_candidate) not in sys.path:
                sys.path.insert(0, str(run_candidate))
            if str(repo_candidate) not in sys.path:
                sys.path.insert(0, str(repo_candidate))
            return


def _find_run_dir(start_path: Path) -> Path:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "data").is_dir() and (candidate / "edmcmc.py").exists():
            return candidate
        run_candidate = candidate / "run"
        if (run_candidate / "data").is_dir() and (run_candidate / "edmcmc.py").exists():
            return run_candidate
    raise FileNotFoundError("Could not find the run/ directory containing data/ and edmcmc.py")


_bootstrap_paths(Path(__file__).resolve().parent)
import edmcmc as edm
import ellc


# %%
# ------------------------------------------------------------
# TOI-4137b system parameters
# ------------------------------------------------------------
planet_name = "TOI-4137b"
model_name = "joint_lc_rv_ld"

r_1_fixed = 0.1280615074825186  # R_star / a (initial)
r_2_fixed = 0.01107927677378  # R_planet / a (initial)
incl_fixed = 85.7  # deg (initial)
a_fixed = 11.228979169999995236  # solar radii
q_fixed = 0.0010469210379437
period_fixed = 3.8016122  # days
t0_bjd_fixed = 2461054.76  # BJD
ecc = 0.0
omega = -10.0  # deg
ld_u = [0.1, 0.3]
ld_model = "quadratic"

rp_guess = r_2_fixed / r_1_fixed
a_over_rstar_guess = 1.0 / r_1_fixed
t0_guess = t0_bjd_fixed
period_guess = period_fixed
inc_guess = incl_fixed
vsini_guess = 10.0
lambda_guess = 25.0

fit_b = True
fit_e = True
fit_K = True
fit_ld = True
fit_period = False

show_lc_sigma = True
write_chains = True
extend_chains = False
thin_n = 5
thin_burnin = 200

nlink = 20000
ncores = 6

# %%
# ------------------------------------------------------------
# Prior/step configuration
# ------------------------------------------------------------
auto_K_from_data = True
K_guess = 0.5
K_lims = [0.0, 5.0]

t0_lims = [t0_guess - 0.5, t0_guess + 0.5]
period_lims = [period_guess - 0.1, period_guess + 0.1]
rp_lims = [0.001, 0.3]
a_lims = [1.0, 20.0]

ecc_lims = [0.0, 0.9]
sqrt_e_max = np.sqrt(ecc_lims[1])
sqrt_e_cosw_guess = np.sqrt(ecc) * np.cos(np.deg2rad(omega))
sqrt_e_sinw_guess = np.sqrt(ecc) * np.sin(np.deg2rad(omega))
sqrt_e_cosw_lims = [-sqrt_e_max, sqrt_e_max]
sqrt_e_sinw_lims = [-sqrt_e_max, sqrt_e_max]

b_guess = (a_over_rstar_guess * np.cos(np.deg2rad(inc_guess)) *
           (1.0 - ecc**2) / (1.0 + ecc * np.sin(np.deg2rad(omega))))
b_lims = [0.0, 1.5]

vsini_lims = [max(0.0, vsini_guess - 5.0), vsini_guess + 5.0]
lambda_lims = [lambda_guess - 30.0, lambda_guess + 30.0]
inc_lims = [70.0, 95.0]
ld_u_lims = [(0.0, 1.0) for _ in ld_u]

use_rho_star_prior = False
rho_star_mu = 2.69
rho_star_sigma = 0.13

param_config = {
    "t0_bjd": {"guess": t0_guess, "wid": 0.01, "prior": t0_lims},
    "period_d": {"guess": period_guess, "wid": 0.001, "prior": period_lims},
    "rp_over_rstar": {"guess": rp_guess, "wid": 0.001, "prior": rp_lims},
    "a_over_rstar": {"guess": a_over_rstar_guess, "wid": 0.02, "prior": a_lims},
    "impact_b": {"guess": b_guess, "wid": 0.01, "prior": b_lims},
    "inc_deg": {"guess": inc_guess, "wid": 0.02, "prior": inc_lims},
    "sqrt_e_cosw": {"guess": sqrt_e_cosw_guess, "wid": 0.05, "prior": sqrt_e_cosw_lims},
    "sqrt_e_sinw": {"guess": sqrt_e_sinw_guess, "wid": 0.05, "prior": sqrt_e_sinw_lims},
    "vsini": {"guess": vsini_guess, "wid": 0.2, "prior": vsini_lims},
    "lambda": {"guess": lambda_guess, "wid": 1.0, "prior": lambda_lims},
    "K": {"guess": K_guess, "wid": 0.05, "prior": K_lims},
}

for i, u in enumerate(ld_u):
    param_config[f"ld_u{i+1}"] = {"guess": u, "wid": 0.05, "prior": list(ld_u_lims[i])}


def build_param_setup():
    names = ["t0_bjd"]
    if fit_period:
        names.append("period_d")
    names += ["rp_over_rstar", "a_over_rstar"]
    if fit_b:
        names.append("impact_b")
    else:
        names.append("inc_deg")
    if fit_e:
        names += ["sqrt_e_cosw", "sqrt_e_sinw"]
    names += ["vsini", "lambda"]
    if fit_K:
        names.append("K")
    if fit_ld:
        names += [f"ld_u{i+1}" for i in range(len(ld_u))]

    p0 = [param_config[n]["guess"] for n in names]
    wid = [param_config[n]["wid"] for n in names]
    parinfo = [
        {"fixed": False, "limits": param_config[n]["prior"], "limited": [True, True]}
        for n in names
    ]
    return names, p0, wid, parinfo


# %%
# ------------------------------------------------------------
# IO setup
# ------------------------------------------------------------
run_dir = _find_run_dir(Path(__file__).resolve().parent)
lc_csvfiles = [
    run_dir / "data" / planet_name / "TOI-4137b_lightcurve_hlsp_tess-spoc_tess_phot_0000000417646390-s0019.csv",
    run_dir / "data" / planet_name / "TOI-4137b_lightcurve_hlsp_tess-spoc_tess_phot_0000000417646390-s0026.csv",
    run_dir / "data" / planet_name / "TOI-4137b_lightcurve_hlsp_tess-spoc_tess_phot_0000000417646390-s0052.csv",
    run_dir / "data" / planet_name / "TOI-4137b_lightcurve_hlsp_tess-spoc_tess_phot_0000000417646390-s0053.csv",
    run_dir / "data" / planet_name / "TOI-4137b_lightcurve_hlsp_tess-spoc_tess_phot_0000000417646390-s0059.csv",
    run_dir / "data" / planet_name / "TOI-4137b_lightcurve_hlsp_tess-spoc_tess_phot_0000000417646390-s0073.csv",
]
lc_csvfiles = [
    run_dir / "data" / planet_name / "TOI-4137b_tess_lightcurve.csv",
]
rv_csvfile = run_dir / "data" / planet_name / "2026Jan14_TIC417646390.csv"
output_dir = run_dir / "edmcmc_output" / planet_name
output_dir.mkdir(parents=True, exist_ok=True)
chains_file = output_dir / f"{planet_name}_{model_name}_chains.npz"

print(f"run_dir: {run_dir}")
print(f"lc_csvfiles: {lc_csvfiles}")
print(f"rv_csvfile: {rv_csvfile}")
print(f"output_dir: {output_dir}")


# %%
# ------------------------------------------------------------
# Load light-curve data (stacked)
# ------------------------------------------------------------
required_cols = ("time", "flux", "flux_err")
all_time = []
all_flux = []
all_flux_err = []

for csvfile in lc_csvfiles:
    df = pd.read_csv(csvfile)
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV missing required column: {col} in {csvfile}")

    mask = np.isfinite(df["time"]) & np.isfinite(df["flux"]) & np.isfinite(df["flux_err"])
    if "quality" in df.columns:
        mask &= df["quality"].to_numpy() == 0

    time = df.loc[mask, "time"].to_numpy(dtype=float)  # BJD - 2457000
    flux_raw = df.loc[mask, "flux"].to_numpy(dtype=float)
    flux_err_raw = df.loc[mask, "flux_err"].to_numpy(dtype=float)

    flux_med = np.nanmedian(flux_raw)
    flux = flux_raw / flux_med
    flux_err = flux_err_raw / flux_med

    all_time.append(time)
    all_flux.append(flux)
    all_flux_err.append(flux_err)

lc_time = np.concatenate(all_time)
lc_flux = np.concatenate(all_flux)
lc_flux_err = np.concatenate(all_flux_err)


# %%
# ------------------------------------------------------------
# Load RV data (BJD)
# ------------------------------------------------------------
rv_df = pd.read_csv(rv_csvfile, comment="#")
for col in ("ccfjdsum", "ccfrvmod", "dvrms"):
    if col not in rv_df.columns:
        raise ValueError(f"RV CSV missing required column: {col}")

rv_time = rv_df["ccfjdsum"].values.astype(float)
rv_data = rv_df["ccfrvmod"].values.astype(float)
rv_err = rv_df["dvrms"].values.astype(float)

# Derived guesses/limits
if auto_K_from_data and fit_K:
    K_guess = 0.5 * (np.nanmax(rv_data) - np.nanmin(rv_data))
    K_lims = [0.0, max(1.0, 2.0 * K_guess)]
    param_config["K"]["guess"] = K_guess
    param_config["K"]["prior"] = K_lims

# Precompute weighted systemic offset using initial guess
i_rad = np.radians(incl_fixed)
rsum = (r_1_fixed + r_2_fixed)
val = rsum / max(1e-12, np.sin(i_rad))
if val >= 1.0:
    transit_duration_days = 0.2
else:
    transit_duration_days = period_fixed / np.pi * val

transit_half_phase = (transit_duration_days / 2.0) / period_fixed
phases_for_mask = ((rv_time - t0_bjd_fixed) / period_fixed + 0.5) % 1.0 - 0.5
in_transit_mask = np.abs(phases_for_mask) < transit_half_phase
out_of_transit_mask = ~in_transit_mask

if out_of_transit_mask.sum() < 3:
    out_of_transit_mask = np.ones_like(out_of_transit_mask, dtype=bool)
weights = 1.0 / (rv_err**2)
gamma_weighted = np.sum(weights[out_of_transit_mask] * rv_data[out_of_transit_mask]) / np.sum(weights[out_of_transit_mask])
rv_data = rv_data - gamma_weighted


# %%
# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
def batman_flux(t0_bjd, per, rp_over_rstar, a_over_rstar, inc_deg, ecc_val, omega_val, ld_coeffs, time_axis):
    t0_lc = t0_bjd - 2457000.0

    bat_params = batman.TransitParams()
    bat_params.t0 = t0_lc
    bat_params.per = per
    bat_params.rp = rp_over_rstar
    bat_params.a = a_over_rstar
    bat_params.inc = inc_deg
    bat_params.ecc = ecc_val
    bat_params.w = omega_val
    bat_params.u = ld_coeffs
    bat_params.limb_dark = ld_model

    try:
        model = batman.TransitModel(bat_params, time_axis)
        return model.light_curve(bat_params)
    except Exception:
        return np.full_like(time_axis, np.nan)


def rv_model(t0_bjd, per, rp_over_rstar, a_over_rstar, inc_deg, ecc_val, omega_val, vsini, lambda_deg, K, time_axis):
    r_1 = 1.0 / a_over_rstar
    r_2 = rp_over_rstar * r_1

    rv_call = ellc.rv(
        time_axis,
        t_zero=t0_bjd,
        period=per,
        lambda_1=lambda_deg,
        radius_1=r_1,
        radius_2=r_2,
        incl=inc_deg,
        a=a_fixed,
        f_c=np.sqrt(ecc_val) * np.cos(np.deg2rad(omega_val)),
        f_s=np.sqrt(ecc_val) * np.sin(np.deg2rad(omega_val)),
        q=q_fixed,
        shape_1="sphere",
        shape_2="sphere",
        vsini_1=vsini,
        flux_weighted=True,
        sbratio=0.0,
        verbose=0,
    )
    if isinstance(rv_call, (tuple, list)):
        base = np.asarray(rv_call[0])
    else:
        base = np.asarray(rv_call)

    if K is None:
        return base

    K0 = 0.5 * (np.nanmax(base) - np.nanmin(base))
    if K0 <= 0.0:
        return np.full_like(base, np.nan)
    return base * (K / K0)


def rho_star_from_a_over_rstar(per_days, a_over_rstar):
    G_cgs = 6.6743e-8
    per_sec = per_days * 86400.0
    return (3.0 * np.pi / (G_cgs * per_sec**2)) * (a_over_rstar**3)

def unpack_params(p):
    idx = 0
    t0_bjd = p[idx]
    idx += 1
    if fit_period:
        per = p[idx]
        idx += 1
    else:
        per = period_fixed
    rp_over_rstar = p[idx]
    idx += 1
    a_over_rstar = p[idx]
    idx += 1

    if fit_b:
        b_val = p[idx]
        idx += 1
    else:
        inc_deg = p[idx]
        idx += 1

    if fit_e:
        sqrt_e_cosw = p[idx]
        sqrt_e_sinw = p[idx + 1]
        idx += 2
        ecc_val = sqrt_e_cosw**2 + sqrt_e_sinw**2
        if ecc_val > 1.0:
            return None
        omega_val = np.degrees(np.arctan2(sqrt_e_sinw, sqrt_e_cosw))
    else:
        ecc_val = ecc
        omega_val = omega

    vsini = p[idx]
    idx += 1
    lambda_deg = p[idx]
    idx += 1

    if fit_K:
        K_val = p[idx]
        idx += 1
    else:
        K_val = None

    if fit_ld:
        ld_coeffs = list(p[idx:idx + len(ld_u)])
        idx += len(ld_u)
    else:
        ld_coeffs = ld_u

    if fit_b:
        fac = (1.0 - ecc_val**2) / (1.0 + ecc_val * np.sin(np.deg2rad(omega_val)))
        if fac <= 0.0:
            return None
        cosi = b_val / (a_over_rstar * fac)
        if cosi < 0.0 or cosi > 1.0:
            return None
        inc_deg = np.degrees(np.arccos(cosi))
    else:
        if not (0.0 < inc_deg <= 90.0):
            return None

    return t0_bjd, per, rp_over_rstar, a_over_rstar, inc_deg, ecc_val, omega_val, vsini, lambda_deg, K_val, ld_coeffs

def loglikelihood_joint(p):
    unpacked = unpack_params(p)
    if unpacked is None:
        return -np.inf

    t0_bjd, per, rp_over_rstar, a_over_rstar, inc_deg, ecc_val, omega_val, vsini, lambda_deg, K_val, ld_coeffs = unpacked

    if per <= 0.0:
        return -np.inf
    if not (0.0 < rp_over_rstar < 1.0):
        return -np.inf
    if a_over_rstar <= 0.0:
        return -np.inf

    lc_model = batman_flux(t0_bjd, per, rp_over_rstar, a_over_rstar, inc_deg, ecc_val, omega_val, ld_coeffs, lc_time)
    if not np.all(np.isfinite(lc_model)):
        return -np.inf
    rv_model_vals = rv_model(t0_bjd, per, rp_over_rstar, a_over_rstar, inc_deg, ecc_val, omega_val, vsini, lambda_deg, K_val, rv_time)
    if not np.all(np.isfinite(rv_model_vals)):
        return -np.inf

    chi2_lc = np.sum((lc_flux - lc_model) ** 2 / lc_flux_err**2)
    chi2_rv = np.sum((rv_data - rv_model_vals) ** 2 / rv_err**2)
    logp = -0.5 * (chi2_lc + chi2_rv)

    if use_rho_star_prior:
        rho_star = rho_star_from_a_over_rstar(per, a_over_rstar)
        logp += -0.5 * ((rho_star - rho_star_mu) / rho_star_sigma) ** 2

    return logp

# %%
# ------------------------------------------------------------
# MCMC
# ------------------------------------------------------------
labels, p0, wid, parinfo = build_param_setup()

pos_in = None
if extend_chains and chains_file.exists():
    data = np.load(chains_file)
    if "lastpos" not in data:
        raise ValueError(f"Chains file missing 'lastpos': {chains_file}")
    pos_in = data["lastpos"]

out = edm.edmcmc(
    loglikelihood_joint,
    p0,
    wid,
    parinfo=parinfo,
    nwalkers=100,
    nlink=nlink,
    nburnin=int(nlink/100),
    ncores=ncores,
    pos_in=pos_in,
    quiet=True,
)

if write_chains:
    thinflatchains = out.get_chains(nthin=thin_n, nburnin=thin_burnin, flat=True)
    if extend_chains and chains_file.exists():
        prev = np.load(chains_file)
        if "thinflatchains" in prev:
            thinflatchains = np.vstack([prev["thinflatchains"], thinflatchains])
    np.savez(
        chains_file,
        thinflatchains=thinflatchains,
        lastpos=out.lastpos,
        nwalkers=out.nwalkers,
        npar=out.npar,
        nburnin=out.nburnin,
        nlink=out.nlink,
        labels=np.array(labels, dtype=object),
    )
    print(f"chains saved: {chains_file}")


# %%
# ------------------------------------------------------------
# Results.txt Function
# ------------------------------------------------------------
def write_results_txt(path, planet_name, model_name, labels, samples, out, lc_files, rv_file, ncores, fit_flags):
    header = [
        "***************************************",
        "#######################################",
        "######                           ######",
        "###### TOI-4137b Retrieval Output ######",
        "######                           ######",
        "#######################################",
        "***************************************",
        "",
        "#################################",
        f"PLANET: {planet_name}",
        f"Model: {model_name}",
        "#################################",
        "",
        "Datasets:",
    ]

    header += [f"-> LC: {Path(f).name}" for f in lc_files]
    header += [f"-> RV: {Path(rv_file).name}"]

    header += [
        "",
        "#################################",
        "Algorithm = EDMCMC",
        f"N_params = {len(labels)}",
        f"N_walkers = {out.nwalkers}",
        f"N_link = {out.nlink}",
        f"N_burnin = {out.nburnin}",
        f"N_cores = {ncores}",
        "",
        "Model flags:",
        f"-> fit_b = {fit_flags['fit_b']}",
        f"-> fit_e = {fit_flags['fit_e']}",
        f"-> fit_K = {fit_flags['fit_K']}",
        f"-> fit_ld = {fit_flags['fit_ld']}",
        "",
        "#################################",
        "Gelman-Rubin statistics:",
    ]

    gr = out.gelmanrubin()
    for i in range(len(gr)):
        header.append(f"Parameter {i+1} ({labels[i]}) has a Gelman-Rubin statistic of {gr[i]}")

    def write_sigma_block(fh, title, p_lo, p_hi):
        fh.write("\n******************************************\n")
        fh.write(f"{title} constraints\n")
        fh.write("******************************************\n")
        for i, name in enumerate(labels):
            s = samples[:, i]
            med = float(np.nanmedian(s))
            lo = float(np.nanpercentile(s, p_lo))
            hi = float(np.nanpercentile(s, p_hi))
            fh.write(f"{name:<15} = {med: .6g} (+{hi - med:.6g}) (-{med - lo:.6g})\n")

    with open(path, "w") as f:
        f.write("\n".join(header))
        write_sigma_block(f, "1 σ", 15.865, 84.135)
        write_sigma_block(f, "2 σ", 2.5, 97.5)
        write_sigma_block(f, "3 σ", 0.135, 99.865)
        write_sigma_block(f, "5 σ", 0.000057, 99.999943)

        f.write("\n******************************************\n")
        f.write("Best-fitting parameters\n")
        f.write("******************************************\n")
        best = getattr(out, 'bestpar', np.nanmedian(samples, axis=0))
        for i, name in enumerate(labels):
            f.write(f"{name:<15} = {best[i]: .6g}\n")
# %%
# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------
fig1, axes1 = plt.subplots(len(p0), figsize=(10, 1 + 2 * len(p0)), sharex=True)
for i in range(len(p0)):
    ax = axes1[i]
    ax.plot(out.whichlink, out.flatchains[:, i], ".", rasterized=True)
    ax.set_ylabel(labels[i])
axes1[-1].set_xlabel("Link number")
fig1_name = output_dir / f"{planet_name}_{model_name}_trace.pdf"
fig1.savefig(fig1_name)
plt.close(fig1)

fig2 = corner.corner(out.flatchains, labels=labels)
fig2_name = output_dir / f"{planet_name}_{model_name}_corner.pdf"
fig2.savefig(fig2_name)
plt.close(fig2)

med = np.median(out.flatchains, axis=0)
std = np.std(out.flatchains, axis=0)
bestfit_df = pd.DataFrame(
    {
        "parameter": labels,
        "median": med,
        "std": std,
    }
)
csv_name = output_dir / f"{planet_name}_{model_name}_bestfit.csv"
bestfit_df.to_csv(csv_name, index=False)
print(f"best-fit csv: {csv_name}")

results_path = output_dir / f"{planet_name}_{model_name}_results.txt"
write_results_txt(
    results_path,
    planet_name,
    model_name,
    labels,
    out.flatchains,
    out,
    lc_csvfiles,
    rv_csvfile,
    ncores,
    {"fit_b": fit_b, "fit_e": fit_e, "fit_K": fit_K, "fit_ld": fit_ld},
)
print(f"results written: {results_path}")


# %%
# ------------------------------------------------------------
# Full Fit Plot
# ------------------------------------------------------------
med_unpacked = unpack_params(med)
if med_unpacked is None:
    raise ValueError("Median parameters are invalid; cannot build plots.")

t0_bjd_med, per_med, rp_med, a_over_med, inc_med, ecc_med, omega_med, vsini_med, lambda_med, K_med, ld_med = med_unpacked

best_lc = batman_flux(t0_bjd_med, per_med, rp_med, a_over_med, inc_med, ecc_med, omega_med, ld_med, lc_time)
best_rv = rv_model(t0_bjd_med, per_med, rp_med, a_over_med, inc_med, ecc_med, omega_med, vsini_med, lambda_med, K_med, rv_time)

lc_phase = ((lc_time - (t0_bjd_med - 2457000.0)) / per_med + 0.5) % 1.0 - 0.5
rv_phase = ((rv_time - t0_bjd_med) / per_med + 0.5) % 1.0 - 0.5

lc_sort = np.argsort(lc_phase)
rv_sort = np.argsort(rv_phase)

# Posterior bands
all_samples = out.flatchains
nsamples_total = all_samples.shape[0]
nsamp = min(1000, nsamples_total)
rng = np.random.default_rng(12345)
sel_idx = rng.choice(nsamples_total, size=nsamp, replace=False)

# LC posterior models
lc_models = np.zeros((nsamp, len(lc_time)))
for j, idx in enumerate(sel_idx):
    unpacked = unpack_params(all_samples[idx, :])
    if unpacked is None:
        lc_models[j, :] = np.nan
        continue
    t0_bjd_s, per_s, rp_s, a_over_s, inc_s, ecc_s, omega_s, _, _, _, ld_s = unpacked
    lc_models[j, :] = batman_flux(t0_bjd_s, per_s, rp_s, a_over_s, inc_s, ecc_s, omega_s, ld_s, lc_time)

lc_median = np.nanmedian(lc_models, axis=0)
lc_p16 = np.nanpercentile(lc_models, 16.0, axis=0)
lc_p84 = np.nanpercentile(lc_models, 84.0, axis=0)
lc_p025 = np.nanpercentile(lc_models, 2.5, axis=0)
lc_p975 = np.nanpercentile(lc_models, 97.5, axis=0)

# RV posterior models
ntime = len(rv_time)
models = np.zeros((nsamp, ntime))
for j, idx in enumerate(sel_idx):
    unpacked = unpack_params(all_samples[idx, :])
    if unpacked is None:
        models[j, :] = np.nan
        continue
    t0_bjd_s, per_s, rp_s, a_over_s, inc_s, ecc_s, omega_s, vsini_s, lambda_s, K_s, _ = unpacked
    models[j, :] = rv_model(t0_bjd_s, per_s, rp_s, a_over_s, inc_s, ecc_s, omega_s, vsini_s, lambda_s, K_s, rv_time)

median_model = np.nanmedian(models, axis=0)
p16 = np.nanpercentile(models, 16.0, axis=0)
p84 = np.nanpercentile(models, 84.0, axis=0)
p025 = np.nanpercentile(models, 2.5, axis=0)
p975 = np.nanpercentile(models, 97.5, axis=0)

font_choice = 'serif'    # change to 'Times New Roman'
label_fontsize = 14      # axis label fontsize
tick_fontsize = 12       # tick label fontsize
legend_fontsize = 12     # legend fontsize
marker_size = 5          # data marker size
model_linewidth = 1.5    # model line width
band_alpha_1sig = 0.34   # alpha for 1-sigma band
band_alpha_2sig = 0.16   # alpha for 2-sigma band

fig3 = plt.figure(figsize=(14, 6))
gs = fig3.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)

# Left panel: LC with residuals
sub_lc = gs[0, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
ax_lc_top = fig3.add_subplot(sub_lc[0, 0])
ax_lc_bot = fig3.add_subplot(sub_lc[1, 0], sharex=ax_lc_top)

if show_lc_sigma:
    ax_lc_top.fill_between(
        lc_phase[lc_sort],
        lc_p025[lc_sort],
        lc_p975[lc_sort],
        color='red',
        alpha=band_alpha_2sig,
        linewidth=0.0
    )
    ax_lc_top.fill_between(
        lc_phase[lc_sort],
        lc_p16[lc_sort],
        lc_p84[lc_sort],
        color='red',
        alpha=band_alpha_1sig,
        linewidth=0.0
    )
ax_lc_top.errorbar(
    lc_phase,
    lc_flux,
    yerr=lc_flux_err,
    fmt=".",
    ms=marker_size,
    alpha=0.6,
    color="k",
    label="Data",
)
ax_lc_top.plot(lc_phase[lc_sort], lc_median[lc_sort], color="red", lw=model_linewidth, label="Median Model")
ax_lc_top.set_xlim(-0.05, 0.05)
ax_lc_top.tick_params(axis='both', labelsize=tick_fontsize)
ax_lc_top.legend(prop={'size': legend_fontsize, 'family': font_choice}, loc='best')

lc_residuals = lc_flux - lc_median
ax_lc_bot.errorbar(
    lc_phase,
    lc_residuals,
    yerr=lc_flux_err,
    fmt=".",
    ms=marker_size,
    alpha=0.6,
    color="k",
)
ax_lc_bot.axhline(0.0, color='red', linestyle='-', alpha=0.7)
ax_lc_bot.set_xlabel("Phase", fontsize=label_fontsize, fontname=font_choice)
ax_lc_bot.set_xlim(-0.05, 0.05)
ax_lc_bot.tick_params(axis='both', labelsize=tick_fontsize)

for ax in (ax_lc_top, ax_lc_bot):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(font_choice)

# Right panel: RV with residuals
sub = gs[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
ax_top = fig3.add_subplot(sub[0, 0])
ax_bot = fig3.add_subplot(sub[1, 0], sharex=ax_top)

# 2-sigma and 1-sigma bands
ax_top.fill_between(rv_phase[rv_sort], p025[rv_sort] * 1e3, p975[rv_sort] * 1e3,
                    color='red', alpha=band_alpha_2sig, linewidth=0.0)
ax_top.fill_between(rv_phase[rv_sort], p16[rv_sort] * 1e3, p84[rv_sort] * 1e3,
                    color='red', alpha=band_alpha_1sig, linewidth=0.0)

# Data and median model
ax_top.errorbar(
    rv_phase,
    rv_data * 1e3,
    yerr=rv_err * 1e3,
    fmt='o',
    ms=marker_size,
    c='k',
    label='Data',
    zorder=5
)
ax_top.plot(
    rv_phase[rv_sort],
    median_model[rv_sort] * 1e3,
    '-',
    lw=model_linewidth,
    alpha=1.0,
    c='red',
    label='Median Model'
)
ax_top.tick_params(axis='both', labelsize=tick_fontsize)
ax_top.legend(prop={'size': legend_fontsize, 'family': font_choice}, loc='best')

# Residuals
residuals_ms = (rv_data - median_model) * 1e3
ax_bot.errorbar(
    rv_phase,
    residuals_ms,
    yerr=rv_err * 1e3,
    fmt='o',
    ms=marker_size,
    c='k',
    label='residuals'
)
ax_bot.axhline(0.0, color='red', linestyle='-', alpha=0.7)
ax_bot.set_xlabel('Phase', fontsize=label_fontsize, fontname=font_choice)
ax_bot.tick_params(axis='both', labelsize=tick_fontsize)

for ax in (ax_top, ax_bot):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(font_choice)

fig3.text(0.05, 0.5, 'Normalized Flux', va='center', rotation='vertical',
          fontsize=label_fontsize, fontname=font_choice)
fig3.text(0.5, 0.5, 'Radial velocity (m/s)', va='center', rotation='vertical',
          fontsize=label_fontsize, fontname=font_choice)


fig3.tight_layout(rect=[0.02, 0.02, 1, 0.98])
fig3_name = output_dir / f"{planet_name}_{model_name}_phase_lc_rv_side_by_side.pdf"
fig3.savefig(fig3_name)
plt.close(fig3)
