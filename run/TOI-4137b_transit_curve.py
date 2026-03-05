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


# ------------------------------------------------------------
# TOI-4137b parameters from run/Planets/TOI-4137b/TOI-4137b_retrieval.py
# ------------------------------------------------------------
planet_name = "TOI-4137b"
model_name = "batman_transit"

r_1 = 0.1280615074825186  # R_star / a
r_2 = 0.01107927677378  # R_planet / a
incl = 85.7  # deg
period = 3.8016122  # days
t0_bjd = 2461054.76  # BJD
ecc = 0.0
omega = 90.0  # deg
ld_u = [0.1, 0.3]
ld_model = "quadratic"

# Batman uses:
# - rp = R_planet / R_star
# - a  = a / R_star
# - t0 in the same units/frame as the light-curve time axis (here: BJD-2457000)
rp_guess = r_2 / r_1
a_over_rstar_guess = 1.0 / r_1
t0_guess = t0_bjd - 2457000.0
period_guess = period
inc_guess = incl


# ------------------------------------------------------------
# IO setup (matches retrieval output location convention)
# ------------------------------------------------------------
run_dir = _find_run_dir(Path(__file__).resolve().parent)
csvfile = run_dir / "data" / planet_name / "TOI-4137b_tess_lightcurve.csv"
output_dir = run_dir / "edmcmc_output" / planet_name
output_dir.mkdir(parents=True, exist_ok=True)

print(f"run_dir: {run_dir}")
print(f"csvfile: {csvfile}")
print(f"output_dir: {output_dir}")


# ------------------------------------------------------------
# Data load
# ------------------------------------------------------------
df = pd.read_csv(csvfile)
required_cols = ("time", "flux", "flux_err")
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Input CSV missing required column: {col}")

mask = np.isfinite(df["time"]) & np.isfinite(df["flux"]) & np.isfinite(df["flux_err"])
if "quality" in df.columns:
    mask &= df["quality"].to_numpy() == 0

time = df.loc[mask, "time"].to_numpy(dtype=float)  # BJD - 2457000
flux_raw = df.loc[mask, "flux"].to_numpy(dtype=float)
flux_err_raw = df.loc[mask, "flux_err"].to_numpy(dtype=float)

flux_med = np.nanmedian(flux_raw)
flux = flux_raw / flux_med
flux_err = flux_err_raw / flux_med


# ------------------------------------------------------------
# Batman model + likelihood
# ------------------------------------------------------------
def build_batman_flux(params_vec, time_axis):
    t0_mod, per_mod, rp_mod, a_over_r_mod, inc_mod = params_vec

    bat_params = batman.TransitParams()
    bat_params.t0 = t0_mod
    bat_params.per = per_mod
    bat_params.rp = rp_mod
    bat_params.a = a_over_r_mod
    bat_params.inc = inc_mod
    bat_params.ecc = ecc
    bat_params.w = omega
    bat_params.u = ld_u
    bat_params.limb_dark = ld_model

    model = batman.TransitModel(bat_params, time_axis)
    return model.light_curve(bat_params)


def loglikelihood(p, x, y, e):
    t0_mod, per_mod, rp_mod, a_over_r_mod, inc_mod = p

    # Basic physical/practical bounds
    if not (t0_guess - 0.5 <= t0_mod <= t0_guess + 0.5):
        return -np.inf
    if per_mod <= 0.0:
        return -np.inf
    if not (0.0 < rp_mod < 1.0):
        return -np.inf
    if a_over_r_mod <= 0.0:
        return -np.inf
    if not (0.0 < inc_mod <= 90.0):
        return -np.inf

    model_flux = build_batman_flux(p, x)
    chi2 = np.sum((y - model_flux) ** 2 / e**2)
    return -0.5 * chi2


labels = ["t0 (BJD-2457000)", "period (d)", "rp/rstar", "a/rstar", "inc (deg)"]
p0 = [t0_guess, period_guess, rp_guess, a_over_rstar_guess, inc_guess]
wid = [0.01, 0.001, 0.001, 0.02, 0.02]
parinfo = [
    {"fixed": False, "limits": [t0_guess - 0.5, t0_guess + 0.5], "limited": [True, True]},
    {"fixed": False, "limits": [period_guess - 0.1, period_guess + 0.1], "limited": [True, True]},
    {"fixed": False, "limits": [0.001, 0.3], "limited": [True, True]},
    {"fixed": False, "limits": [1.0, 50.0], "limited": [True, True]},
    {"fixed": False, "limits": [70.0, 90.0], "limited": [True, True]},
]
ndim = len(p0)


# ------------------------------------------------------------
# MCMC
# ------------------------------------------------------------
out = edm.edmcmc(
    loglikelihood,
    p0,
    wid,
    args=(time, flux, flux_err),
    parinfo=parinfo,
    nwalkers=200,
    nlink=10000,
    nburnin=500,
    ncores=12,
    quiet=True,
)


# ------------------------------------------------------------
# Diagnostics: trace + corner
# ------------------------------------------------------------
fig1, axes1 = plt.subplots(ndim, figsize=(10, 1 + 2 * ndim), sharex=True)
for i in range(ndim):
    ax = axes1[i]
    ax.plot(out.whichlink, out.flatchains[:, i], ".")
    ax.set_ylabel(labels[i])
axes1[-1].set_xlabel("Link number")
fig1_name = output_dir / f"{planet_name}_{model_name}_trace.pdf"
fig1.savefig(fig1_name)
plt.close(fig1)
print(f"trace plot: {fig1_name}")

fig2 = corner.corner(out.flatchains, labels=labels)
fig2_name = output_dir / f"{planet_name}_{model_name}_corner.pdf"
fig2.savefig(fig2_name)
plt.close(fig2)
print(f"corner plot: {fig2_name}")


# ------------------------------------------------------------
# Best-fit summary + model plot
# ------------------------------------------------------------
med = np.median(out.flatchains, axis=0)
std = np.std(out.flatchains, axis=0)

bestfit_df = pd.DataFrame(
    {
        "parameter": ["t0_bjd_minus_2457000", "period_days", "rp_over_rstar", "a_over_rstar", "incl_deg"],
        "median": med,
        "std": std,
    }
)
csv_name = output_dir / f"{planet_name}_{model_name}_bestfit.csv"
bestfit_df.to_csv(csv_name, index=False)
print(f"best-fit csv: {csv_name}")

best_flux = build_batman_flux(med, time)
phase = ((time - med[0]) / med[1] + 0.5) % 1.0 - 0.5
sort_idx = np.argsort(phase)

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.errorbar(
    phase,
    flux,
    yerr=flux_err,
    fmt=".",
    ms=4,
    alpha=1.0,
    color="k",
    label="TESS data",
)
ax3.plot(phase[sort_idx], best_flux[sort_idx], color="red", lw=1.5, label="median model")
ax3.set_xlabel("Phase")
ax3.set_ylabel("Normalized Flux")
ax3.set_xlim(-0.05, 0.05)
ax3.legend(loc="best")
# ax3.grid(alpha=0.25)
fig3.tight_layout()

fig3_name = output_dir / f"{planet_name}_{model_name}_transit_model.pdf"
fig3.savefig(fig3_name)
plt.close(fig3)
print(f"transit model plot: {fig3_name}")


print(
    "Medians +/- std:\n"
    f"t0 (BJD-2457000): {med[0]:.8f} +/- {std[0]:.8f}\n"
    f"period (d):       {med[1]:.8f} +/- {std[1]:.8f}\n"
    f"rp/rstar:         {med[2]:.8f} +/- {std[2]:.8f}\n"
    f"a/rstar:          {med[3]:.8f} +/- {std[3]:.8f}\n"
    f"inc (deg):        {med[4]:.8f} +/- {std[4]:.8f}"
)
