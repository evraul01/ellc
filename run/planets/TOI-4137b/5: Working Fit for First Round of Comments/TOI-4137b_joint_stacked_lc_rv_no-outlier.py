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
from matplotlib.contour import QuadContourSet
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


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
model_name = "joint_stacked_lc_rv_no-outlier"

r_1_fixed = 0.1280615074825186  # R_star / a (initial)
r_2_fixed = 0.01107927677378  # R_planet / a (initial)
incl_fixed = 85.7  # deg (initial)
a_fixed = 11.228979169999995236  # solar radii
stellar_radius_fixed = a_fixed * r_1_fixed  # solar radii
q_fixed = 0.0010469210379437
period_fixed = 3.8016122  # days
t0_bjd_fixed = 2461054.7469345736  # BJD
ecc = 0.0
omega = -10.0  # deg
ld_u = [0.1, 0.3]
ld_model = "quadratic"

rp_guess = r_2_fixed / r_1_fixed
a_over_rstar_guess = 1.0 / r_1_fixed
t0_guess = t0_bjd_fixed
period_guess = period_fixed
inc_guess = incl_fixed
vsini_guess = 15.0
lambda_guess = 30.0

fit_b = True
fit_e = True
fit_K = True
fit_ld = True
fit_period = True


report_K_in_m_per_s = True
include_derived_rho_star = True
include_derived_inclination = True
include_derived_planet_radius = True
include_derived_semi_major_axis = True

R_SUN_TO_R_EARTH = 109.076
R_SUN_TO_AU = 0.00465047

corner_label_fontsize = 18
show_lc_sigma = True
show_lc_binned_points = True
lc_phase_bin_width = 0.0025
lc_raw_alpha = 0.025
lc_raw_label = "Phase-Folded Data"
lc_binned_label = "Average Binned Data"
lc_model_label = "Median Model"
rv_oversample_model = True
rv_oversample_model = False
rv_oversample_nsub = 23
omit_rv_sorted_indices = [0]  # chronological RV-point indices to omit from the fit
omitted_rv_alpha = 0.25  # opacity of ommitted RV outliers
write_chains = True
extend_chains = False
save_full_chains = False
thin_n = 5
thin_burnin = None

nlink = 50000
# nlink = 0  # Remake '_combined' outputs
nburnin = None  # defaults to total combined nlink/10 if None
ncores = 4

# %%
# ------------------------------------------------------------
# Prior/step configuration
# ------------------------------------------------------------
auto_K_from_data = False
K_guess = 0.125
K_lims = [0.0, 0.500]

t0_lims = [t0_guess - 0.001, t0_guess + 0.001]
period_lims = [period_guess - 0.1, period_guess + 0.1]  # ignored
rp_lims = [0.0, 1.0]
a_lims = [2.5, 20.0]  # ignored

ecc_lims = [0.0, 1.0]
sqrt_e_max = np.sqrt(ecc_lims[1])
sqrt_e_cosw_guess = np.sqrt(ecc) * np.cos(np.deg2rad(omega))
sqrt_e_sinw_guess = np.sqrt(ecc) * np.sin(np.deg2rad(omega))
sqrt_e_cosw_lims = [-sqrt_e_max, sqrt_e_max]
sqrt_e_sinw_lims = [-sqrt_e_max, sqrt_e_max]

b_guess = (a_over_rstar_guess * np.cos(np.deg2rad(inc_guess)) *
           (1.0 - ecc**2) / (1.0 + ecc * np.sin(np.deg2rad(omega))))
b_lims = [0.0, 1.0]

vsini_lims = [max(0.0, vsini_guess - 15.0), vsini_guess + 15.0]
lambda_lims = [lambda_guess - 210.0, lambda_guess + 180.0]
inc_lims = [70.0, 90.0]  # ignored
ld_u_lims = [(0.0, 1.0) for _ in ld_u]

use_rho_star_prior = True
rho_star_mu = 0.621
rho_star_sigma = 0.1
rho_star_prior_type = "gaussian"
if rho_star_prior_type.lower() == "gaussian":
    rho_star_prior = [rho_star_mu, rho_star_sigma]
else:
    rho_star_prior = [rho_star_mu - rho_star_sigma, rho_star_mu + rho_star_sigma]

param_config = {

    "t0_bjd": {"guess": t0_guess, "wid": 0.00005, "prior": t0_lims},
    "period_d": {"guess": period_guess, "wid": 0.00001, "prior_type": "none", "prior": period_lims},
    "rp_over_rstar": {"guess": rp_guess, "wid": 0.001, "prior": rp_lims},
    "a_over_rstar": {"guess": a_over_rstar_guess, "wid": 0.02, "prior_type": "none", "prior": a_lims},
    "impact_b": {"guess": b_guess, "wid": 0.01, "prior": b_lims},
    "inc_deg": {"guess": inc_guess, "wid": 0.02, "prior_type": "none", "prior": inc_lims},
    "sqrt_e_cosw": {"guess": sqrt_e_cosw_guess, "wid": 0.05, "prior": sqrt_e_cosw_lims},
    "sqrt_e_sinw": {"guess": sqrt_e_sinw_guess, "wid": 0.05, "prior": sqrt_e_sinw_lims},
    "vsini": {"guess": vsini_guess, "wid": 0.2, "prior": vsini_lims},
    "lambda": {"guess": lambda_guess, "wid": 1.0, "prior": lambda_lims},
    "K": {"guess": K_guess, "wid": 0.05, "prior": K_lims},
}

corner_label_map = {
    "t0_bjd": r"$t_0$ (BJD)",
    "period_d": r"$P$ (d)",
    "rp_over_rstar": r"$R_{\rm p}/R_\star$",
    "a_over_rstar": r"$a/R_\star$",
    "impact_b": r"$b$",
    "inc_deg": r"$i$ ($^\circ$)",
    "sqrt_e_cosw": r"$\sqrt{e}\cos\omega_\star$",
    "sqrt_e_sinw": r"$\sqrt{e}\sin\omega_\star$",
    "vsini": r"$v \sin i_\star$ (km s$^{-1}$)",
    "lambda": r"$\lambda$ ($^\circ$)",
    "K": r"$K$ (km s$^{-1}$)",
    "K_m_s": r"$K$ (m s$^{-1}$)",
    "ld_u1": r"$u_1$",
    "ld_u2": r"$u_2$",
    "rho_star_g_cm3": r"$\rho_\star$ (g cm$^{-3}$)",
    "incl_deg_derived": r"$i$ ($^\circ$)",
    "planet_radius_rearth": r"$R_{\rm p}$ ($R_\oplus$)",
    "semi_major_axis_au": r"$a$ (au)",
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
    parinfo = []
    for n in names:
        prior_type = param_config[n].get("prior_type", "uniform").lower()
        if prior_type == "uniform":
            parinfo.append({"fixed": False, "limits": param_config[n]["prior"], "limited": [True, True]})
        elif prior_type == "gaussian":
            parinfo.append({"fixed": False, "limits": [0.0, 0.0], "limited": [False, False]})
        elif prior_type == "none":
            parinfo.append({"fixed": False, "limits": [0.0, 0.0], "limited": [False, False]})
        else:
            raise ValueError(f"Unsupported prior_type for {n}: {prior_type}")
    return names, p0, wid, parinfo


def evaluate_prior_value(value, prior_values, prior_type="uniform"):
    prior_type = prior_type.lower()
    if prior_type == "none":
        return 0.0
    if len(prior_values) != 2:
        raise ValueError(f"Prior specification must have exactly two values for {prior_type} priors.")
    if prior_type == "uniform":
        lower, upper = prior_values
        if value < lower or value > upper:
            return -np.inf
        return 0.0
    if prior_type == "gaussian":
        mu, sigma = prior_values
        if sigma <= 0.0:
            raise ValueError("Gaussian prior sigma must be positive.")
        return -0.5 * ((value - mu) / sigma) ** 2
    raise ValueError(f"Unsupported prior_type: {prior_type}")


def evaluate_sampled_log_prior(p, labels):
    logp = 0.0
    for i, name in enumerate(labels):
        cfg = param_config[name]
        this_logp = evaluate_prior_value(p[i], cfg["prior"], cfg.get("prior_type", "uniform"))
        if not np.isfinite(this_logp):
            return -np.inf
        logp += this_logp
    return logp


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
# lc_csvfiles = [
#     run_dir / "data" / planet_name / "TOI-4137b_tess_lightcurve.csv",
# ]
rv_csvfile = run_dir / "data" / planet_name / "2026Jan14_TIC417646390.csv"
output_dir = run_dir / "edmcmc_output" / planet_name
output_dir.mkdir(parents=True, exist_ok=True)
chains_file = output_dir / f"{planet_name}_{model_name}_chains.npz"
full_chains_file = output_dir / f"{planet_name}_{model_name}_chains_full.npz"

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

rv_df = rv_df.sort_values("ccfjdsum").reset_index(drop=True)
rv_time_all = rv_df["ccfjdsum"].values.astype(float)
rv_data_all = rv_df["ccfrvmod"].values.astype(float)
rv_err_all = rv_df["dvrms"].values.astype(float)
if "exptime" in rv_df.columns:
    rv_exptime_s_all = rv_df["exptime"].values.astype(float)
else:
    rv_exptime_s_all = np.zeros_like(rv_time_all)

omit_rv_mask_all = np.zeros_like(rv_time_all, dtype=bool)
for idx in omit_rv_sorted_indices:
    if 0 <= idx < len(omit_rv_mask_all):
        omit_rv_mask_all[idx] = True

rv_time = rv_time_all[~omit_rv_mask_all]
rv_data = rv_data_all[~omit_rv_mask_all]
rv_err = rv_err_all[~omit_rv_mask_all]
rv_exptime_s = rv_exptime_s_all[~omit_rv_mask_all]
rv_time_omitted = rv_time_all[omit_rv_mask_all]
rv_data_omitted = rv_data_all[omit_rv_mask_all]
rv_err_omitted = rv_err_all[omit_rv_mask_all]
rv_exptime_s_omitted = rv_exptime_s_all[omit_rv_mask_all]

# Derived guesses/limits
if auto_K_from_data and fit_K:
    K_guess = 0.5 * (np.nanmax(rv_data) - np.nanmin(rv_data))
    K_lims = [0.0, max(1.0, 2.0 * K_guess)]
    param_config["K"]["guess"] = K_guess
    k_prior_type = param_config["K"].get("prior_type", "uniform").lower()
    if k_prior_type == "gaussian":
        param_config["K"]["prior"] = [K_guess, param_config["K"]["prior"][1]]
    elif k_prior_type == "uniform":
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
rv_data_all = rv_data_all - gamma_weighted
rv_data_omitted = rv_data_omitted - gamma_weighted


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


def semi_major_axis_from_a_over_rstar(a_over_rstar):
    return a_over_rstar * stellar_radius_fixed


def rv_model(t0_bjd, per, rp_over_rstar, a_over_rstar, inc_deg, ecc_val, omega_val, vsini, lambda_deg, K, time_axis, exptime_axis_s=None):
    r_1 = 1.0 / a_over_rstar
    r_2 = rp_over_rstar * r_1
    a_orbit = semi_major_axis_from_a_over_rstar(a_over_rstar)
    eval_time_axis = np.asarray(time_axis, dtype=float)
    reshape_shape = None
    if rv_oversample_model and exptime_axis_s is not None and rv_oversample_nsub > 1:
        exptime_days = np.asarray(exptime_axis_s, dtype=float) / 86400.0
        exptime_days = np.where(np.isfinite(exptime_days) & (exptime_days > 0.0), exptime_days, 0.0)
        sub_offsets = np.linspace(-0.5, 0.5, rv_oversample_nsub)
        eval_time_axis = eval_time_axis[:, None] + exptime_days[:, None] * sub_offsets[None, :]
        reshape_shape = eval_time_axis.shape
        eval_time_axis = eval_time_axis.reshape(-1)

    rv_call = ellc.rv(
        eval_time_axis,
        t_zero=t0_bjd,
        period=per,
        lambda_1=lambda_deg,
        radius_1=r_1,
        radius_2=r_2,
        incl=inc_deg,
        a=a_orbit,
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
    if reshape_shape is not None:
        base = np.nanmean(base.reshape(reshape_shape), axis=1)

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


def build_derived_posterior_samples(base_samples, include_constant=False):
    derived_labels = []
    if include_derived_rho_star:
        derived_labels.append("rho_star_g_cm3")
    if include_derived_inclination:
        derived_labels.append("incl_deg_derived")
    if include_derived_planet_radius:
        derived_labels.append("planet_radius_rearth")
    if include_derived_semi_major_axis:
        derived_labels.append("semi_major_axis_au")

    if not derived_labels:
        return np.empty((base_samples.shape[0], 0)), []

    derived_samples = np.full((base_samples.shape[0], len(derived_labels)), np.nan, dtype=float)

    for i in range(base_samples.shape[0]):
        unpacked = unpack_params(base_samples[i, :])
        if unpacked is None:
            continue
        _, per, rp_over_rstar, a_over_rstar, inc_deg, _, _, _, _, _, _ = unpacked
        col = 0
        if include_derived_rho_star:
            derived_samples[i, col] = rho_star_from_a_over_rstar(per, a_over_rstar)
            col += 1
        if include_derived_inclination:
            derived_samples[i, col] = inc_deg
            col += 1
        if include_derived_planet_radius:
            derived_samples[i, col] = rp_over_rstar * stellar_radius_fixed * R_SUN_TO_R_EARTH
            col += 1
        if include_derived_semi_major_axis:
            derived_samples[i, col] = semi_major_axis_from_a_over_rstar(a_over_rstar) * R_SUN_TO_AU

    finite_mask = np.all(np.isfinite(derived_samples), axis=1)
    if not np.any(finite_mask):
        return np.empty((0, 0)), []

    derived_samples = derived_samples[finite_mask]
    keep = np.ones(len(derived_labels), dtype=bool)
    if not include_constant:
        keep = np.nanstd(derived_samples, axis=0) > 0.0
    return derived_samples[:, keep], [derived_labels[j] for j in range(len(derived_labels)) if keep[j]]


def transform_output_samples(samples, labels):
    transformed_samples = np.array(samples, copy=True)
    transformed_labels = list(labels)
    for i, name in enumerate(transformed_labels):
        if name == "K" and report_K_in_m_per_s:
            transformed_samples[:, i] = transformed_samples[:, i] * 1e3
            transformed_labels[i] = "K_m_s"
    return transformed_samples, transformed_labels


def get_plot_display_labels(labels):
    return [corner_label_map.get(label, label) for label in labels]


def _make_corner_scalar_formatter():
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    formatter.set_useOffset(True)
    return formatter


def _merge_offset_into_label(label_text, offset_text):
    if not offset_text:
        return label_text
    if " (" in label_text and label_text.endswith(")"):
        main_text, unit_text = label_text.rsplit(" (", 1)
        unit_text = unit_text[:-1].strip()
        if "BJD" in unit_text:
            return f"{main_text}\n({unit_text}{offset_text})"
        if unit_text:
            return f"{main_text} ({unit_text}{offset_text})"
        return f"{main_text} ({offset_text})"
    return f"{label_text}\n[{offset_text}]"


def apply_corner_scalar_formatting(fig, display_labels, label_fontsize):
    ndim = len(display_labels)
    if ndim == 0:
        return

    axes = np.array(fig.axes).reshape((ndim, ndim))
    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]
            ax.xaxis.set_major_formatter(_make_corner_scalar_formatter())
            if row > col:
                ax.yaxis.set_major_formatter(_make_corner_scalar_formatter())

    fig.canvas.draw()

    for col in range(ndim):
        ax = axes[-1, col]
        x_offset = ax.xaxis.get_offset_text().get_text().strip()
        if x_offset:
            ax.set_xlabel(_merge_offset_into_label(display_labels[col], x_offset), fontsize=label_fontsize, labelpad=12)
            ax.xaxis.get_offset_text().set_visible(False)

    for row in range(1, ndim):
        ax = axes[row, 0]
        y_offset = ax.yaxis.get_offset_text().get_text().strip()
        if y_offset:
            ax.set_ylabel(_merge_offset_into_label(display_labels[row], y_offset), fontsize=label_fontsize, labelpad=12)
            ax.yaxis.get_offset_text().set_visible(False)


def combine_posterior_outputs(base_samples, base_labels, include_constant=False):
    derived_samples, derived_labels = build_derived_posterior_samples(base_samples, include_constant=include_constant)
    combined_samples = base_samples
    combined_labels = list(base_labels)
    if derived_labels:
        combined_samples = np.hstack([base_samples, derived_samples])
        combined_labels += derived_labels
    return transform_output_samples(combined_samples, combined_labels)


def filter_constant_plot_columns(samples, labels, tol=1e-15):
    if samples.size == 0:
        return samples, labels
    keep = np.zeros(samples.shape[1], dtype=bool)
    for i in range(samples.shape[1]):
        col = samples[:, i]
        finite = np.isfinite(col)
        if not np.any(finite):
            continue
        col_finite = col[finite]
        span = np.nanmax(col_finite) - np.nanmin(col_finite)
        scale = max(1.0, np.nanmax(np.abs(col_finite)))
        keep[i] = span > tol * scale
    if np.any(keep):
        return samples[:, keep], [labels[i] for i in range(len(labels)) if keep[i]]
    raise ValueError("No parameters with non-zero dynamic range are available for the corner plot.")


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
    sampled_logp = evaluate_sampled_log_prior(p, labels)
    if not np.isfinite(sampled_logp):
        return -np.inf

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
    rv_model_vals = rv_model(t0_bjd, per, rp_over_rstar, a_over_rstar, inc_deg, ecc_val, omega_val, vsini, lambda_deg, K_val, rv_time, rv_exptime_s)
    if not np.all(np.isfinite(rv_model_vals)):
        return -np.inf

    chi2_lc = np.sum((lc_flux - lc_model) ** 2 / lc_flux_err**2)
    chi2_rv = np.sum((rv_data - rv_model_vals) ** 2 / rv_err**2)
    logp = sampled_logp - 0.5 * (chi2_lc + chi2_rv)

    if use_rho_star_prior:
        rho_star = rho_star_from_a_over_rstar(per, a_over_rstar)
        rho_logp = evaluate_prior_value(rho_star, rho_star_prior, rho_star_prior_type)
        if not np.isfinite(rho_logp):
            return -np.inf
        logp += rho_logp

    return logp

# %%
# ------------------------------------------------------------
# MCMC
# ------------------------------------------------------------
labels, p0, wid, parinfo = build_param_setup()

def flatten_fullchains_for_posterior(fullchains, burnin, nthin):
    nwalkers_saved, nlink_saved, npar_saved = fullchains.shape
    if burnin >= nlink_saved:
        return np.empty((0, npar_saved))
    indices = burnin + np.arange(np.floor((nlink_saved - burnin) / nthin)) * nthin
    indices = indices.astype(int)
    return fullchains[:, indices, :].reshape(-1, npar_saved)

def approximate_thinflatchains_for_posterior(thinflatchains, saved_nlink, saved_posterior_burnin, requested_posterior_burnin, saved_thin_n):
    if requested_posterior_burnin <= saved_posterior_burnin:
        return thinflatchains
    saved_kept_perwalker = int(np.floor((saved_nlink - saved_posterior_burnin) / saved_thin_n))
    if saved_kept_perwalker <= 0:
        return np.empty((0, thinflatchains.shape[1]))
    drop_perwalker = int(np.ceil((requested_posterior_burnin - saved_posterior_burnin) / saved_thin_n))
    drop_perwalker = max(0, min(drop_perwalker, saved_kept_perwalker))
    drop_fraction = drop_perwalker / float(saved_kept_perwalker)
    n_drop = int(np.floor(thinflatchains.shape[0] * drop_fraction))
    return thinflatchains[n_drop:, :]

def compute_gelmanrubin_from_fullchains(fullchains, burnin, out):
    if fullchains is None:
        return None, "Combined Gelman-Rubin unavailable because previous full chains are not available."
    cutchains = fullchains[:, burnin:, :]
    if cutchains.shape[1] < 2:
        return None, f"Combined Gelman-Rubin unavailable because only {cutchains.shape[1]} post-burn-in links remain per walker."
    grstats = np.zeros(cutchains.shape[2])
    for i in range(cutchains.shape[2]):
        grstats[i] = out.onegelmanrubin(cutchains[:, :, i])
    return grstats, None


class SavedChainReplay:
    def __init__(self, thin_samples, nwalkers, npar, nlink, nburnin, fullchains=None, fullneglogl=None):
        self.flatchains = thin_samples
        self.nwalkers = nwalkers
        self.npar = npar
        self.nlink = nlink
        self.nburnin = nburnin
        self.fullchains = fullchains
        self.fullneglogl = fullneglogl
        if fullchains is not None:
            post_burn = fullchains[:, nburnin:nlink, :]
            npost = post_burn.shape[1]
            self.flatchains = post_burn.reshape(nwalkers * npost, npar)
            self.whichlink = np.tile(np.arange(nburnin, nlink), nwalkers)
        else:
            self.whichlink = np.arange(self.flatchains.shape[0])

    def onegelmanrubin(self, chain):
        ssq = np.var(chain, axis=1, ddof=1)
        W = np.mean(ssq, axis=0)
        thetab = np.mean(chain, axis=1)
        thetabb = np.mean(thetab, axis=0)
        m = chain.shape[0]
        n = chain.shape[1]
        B = n / (m - 1) * np.sum((thetabb - thetab)**2, axis=0)
        var_theta = (n - 1) / n * W + 1 / n * B
        return np.sqrt(var_theta / W)

prev_thin = None
pos_in = None
prev_total_nlink = 0
prev_fullchains = None
prev_fullneglogl = None
if extend_chains and chains_file.exists():
    data = np.load(chains_file, allow_pickle=True)
    if "labels" not in data:
        raise ValueError(f"Chains file missing 'labels': {chains_file}")
    if list(data["labels"]) != list(labels):
        raise ValueError("Chains file labels do not match current parameter labels; refusing to extend.")
    if "lastpos" not in data:
        raise ValueError(f"Chains file missing 'lastpos': {chains_file}")
    pos_in = data["lastpos"]
    prev_total_nlink = int(data["total_combined_nlink"]) if "total_combined_nlink" in data else int(data["nlink"])

combined_total_nlink = prev_total_nlink + nlink
effective_total_nburnin = int(combined_total_nlink / 10) if nburnin is None else nburnin
effective_total_posterior_burnin = int(combined_total_nlink / 10) if thin_burnin is None else thin_burnin
run_nburnin = min(nlink, max(0, effective_total_nburnin - prev_total_nlink))
run_posterior_burnin = min(nlink, max(0, effective_total_posterior_burnin - prev_total_nlink))

if extend_chains and chains_file.exists():
    if full_chains_file.exists():
        full_data = np.load(full_chains_file, allow_pickle=True)
        if "labels" not in full_data:
            raise ValueError(f"Full chain file missing 'labels': {full_chains_file}")
        if list(full_data["labels"]) != list(labels):
            raise ValueError("Full chain file labels do not match current parameter labels; refusing to extend.")
        if "fullchains" not in full_data:
            raise ValueError(f"Full chain file missing 'fullchains': {full_chains_file}")
        full_total_nlink = int(full_data["total_combined_nlink"]) if "total_combined_nlink" in full_data else full_data["fullchains"].shape[1]
        if full_total_nlink == prev_total_nlink:
            prev_fullchains = full_data["fullchains"]
            if "fullneglogl" in full_data:
                prev_fullneglogl = full_data["fullneglogl"]
            prev_thin = flatten_fullchains_for_posterior(prev_fullchains, effective_total_posterior_burnin, thin_n)
    if prev_thin is None:
        if "thinflatchains" not in data:
            raise ValueError(f"Chains file missing 'thinflatchains': {chains_file}")
        saved_posterior_burnin = int(data["posterior_burnin"]) if "posterior_burnin" in data else int(data["nburnin"])
        saved_thin_n = int(data["thin_n"]) if "thin_n" in data else thin_n
        saved_nlink = prev_total_nlink
        prev_thin = approximate_thinflatchains_for_posterior(
            data["thinflatchains"],
            saved_nlink,
            saved_posterior_burnin,
            effective_total_posterior_burnin,
            saved_thin_n,
        )
    if prev_thin.size == 0:
        raise ValueError(
            f"Requested burn-in ({effective_total_posterior_burnin}) removes all saved samples from the previous run."
        )
use_combined = extend_chains and prev_thin is not None
if nlink == 0:
    if not chains_file.exists():
        raise FileNotFoundError(f"Cannot re-plot with nlink=0 because no saved chain file exists: {chains_file}")
    saved = np.load(chains_file, allow_pickle=True)
    if "labels" not in saved:
        raise ValueError(f"Chains file missing 'labels': {chains_file}")
    if list(saved["labels"]) != list(labels):
        raise ValueError("Saved chain labels do not match current parameter labels; refusing to re-plot.")
    if "thinflatchains" not in saved:
        raise ValueError(f"Chains file missing 'thinflatchains': {chains_file}")

    saved_total_nlink = int(saved["total_combined_nlink"]) if "total_combined_nlink" in saved else int(saved["nlink"])
    saved_posterior_burnin = int(saved["posterior_burnin"]) if "posterior_burnin" in saved else int(saved["nburnin"])
    saved_thin_n = int(saved["thin_n"]) if "thin_n" in saved else thin_n

    combined_total_nlink = saved_total_nlink
    effective_total_nburnin = int(combined_total_nlink / 10) if nburnin is None else nburnin
    effective_total_posterior_burnin = int(combined_total_nlink / 10) if thin_burnin is None else thin_burnin

    combined_fullchains = None
    combined_fullneglogl = None
    if full_chains_file.exists():
        full_saved = np.load(full_chains_file, allow_pickle=True)
        if "labels" not in full_saved:
            raise ValueError(f"Full chain file missing 'labels': {full_chains_file}")
        if list(full_saved["labels"]) != list(labels):
            raise ValueError("Saved full-chain labels do not match current parameter labels; refusing to re-plot.")
        if "fullchains" not in full_saved:
            raise ValueError(f"Full chain file missing 'fullchains': {full_chains_file}")
        combined_fullchains = full_saved["fullchains"]
        if "fullneglogl" in full_saved:
            combined_fullneglogl = full_saved["fullneglogl"]

    if combined_fullchains is not None:
        samples_for_outputs = flatten_fullchains_for_posterior(
            combined_fullchains,
            effective_total_posterior_burnin,
            thin_n,
        )
    else:
        samples_for_outputs = approximate_thinflatchains_for_posterior(
            saved["thinflatchains"],
            saved_total_nlink,
            saved_posterior_burnin,
            effective_total_posterior_burnin,
            saved_thin_n,
        )
    if samples_for_outputs.size == 0:
        raise ValueError("Requested posterior burn-in removes all saved samples; nothing left to re-plot.")

    out = SavedChainReplay(
        thin_samples=samples_for_outputs,
        nwalkers=int(saved["nwalkers"]),
        npar=int(saved["npar"]),
        nlink=combined_total_nlink,
        nburnin=effective_total_nburnin,
        fullchains=combined_fullchains,
        fullneglogl=combined_fullneglogl,
    )
else:
    out = edm.edmcmc(
        loglikelihood_joint,
        p0,
        wid,
        parinfo=parinfo,
        nwalkers=100,
        nlink=nlink,
        nburnin=run_nburnin,
        ncores=ncores,
        pos_in=pos_in,
        quiet=True,
    )

    new_thin = out.get_chains(nthin=thin_n, nburnin=run_posterior_burnin, flat=True)
    combined_thin = new_thin
    if use_combined:
        combined_thin = np.vstack([prev_thin, new_thin])
    samples_for_outputs = combined_thin if use_combined else new_thin

    if write_chains:
        thinflatchains = combined_thin if use_combined else new_thin
        np.savez(
            chains_file,
            thinflatchains=thinflatchains,
            lastpos=out.lastpos,
            nwalkers=out.nwalkers,
            npar=out.npar,
            nburnin=out.nburnin,
            posterior_burnin=effective_total_posterior_burnin,
            thin_n=thin_n,
            nlink=out.nlink,
            total_combined_nlink=combined_total_nlink,
            labels=np.array(labels, dtype=object),
        )
        print(f"chains saved: {chains_file}")
        if save_full_chains:
            fullchains_to_save = out.fullchains
            fullneglogl_to_save = out.fullneglogl
            if extend_chains and prev_fullchains is not None and prev_fullneglogl is not None:
                if prev_fullchains.shape[0] == out.fullchains.shape[0] and prev_fullchains.shape[2] == out.fullchains.shape[2]:
                    fullchains_to_save = np.concatenate([prev_fullchains, out.fullchains], axis=1)
                    fullneglogl_to_save = np.concatenate([prev_fullneglogl, out.fullneglogl], axis=1)
            np.savez(
                full_chains_file,
                fullchains=fullchains_to_save,
                fullneglogl=fullneglogl_to_save,
                lastpos=out.lastpos,
                nwalkers=out.nwalkers,
                npar=out.npar,
                nburnin=out.nburnin,
                posterior_burnin=effective_total_posterior_burnin,
                thin_n=thin_n,
                nlink=out.nlink,
                total_combined_nlink=fullchains_to_save.shape[1],
                labels=np.array(labels, dtype=object),
            )
            print(f"full chains saved: {full_chains_file}")

    combined_fullchains = out.fullchains
    if extend_chains and prev_fullchains is not None:
        if prev_fullchains.shape[0] == out.fullchains.shape[0] and prev_fullchains.shape[2] == out.fullchains.shape[2]:
            combined_fullchains = np.concatenate([prev_fullchains, out.fullchains], axis=1)

gr_metrics, gr_warning = compute_gelmanrubin_from_fullchains(combined_fullchains, effective_total_nburnin, out)
output_suffix = "_combined" if use_combined else ""


# %%
# ------------------------------------------------------------
# Results.txt Function
# ------------------------------------------------------------
def write_results_txt(path, planet_name, model_name, labels, samples, out, lc_files, rv_file, ncores, n_fit_params, fit_flags, gr_metrics=None, gr_warning=None):
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
        f"N_params = {n_fit_params}",
        f"N_reported = {len(labels)}",
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

    if gr_metrics is not None:
        for i in range(len(gr_metrics)):
            header.append(f"Parameter {i+1} ({labels[i]}) has a Gelman-Rubin statistic of {gr_metrics[i]}")
    elif gr_warning is not None:
        header.append(f"WARNING: {gr_warning}")

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
        best = np.nanmedian(samples, axis=0)
        for i, name in enumerate(labels):
            f.write(f"{name:<15} = {best[i]: .6g}\n")
# %%
# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------
plot_samples, plot_labels = filter_constant_plot_columns(*combine_posterior_outputs(samples_for_outputs, labels, include_constant=False))
summary_samples, summary_labels = combine_posterior_outputs(samples_for_outputs, labels, include_constant=True)

sampled_med = np.median(samples_for_outputs, axis=0)
summary_med = np.median(summary_samples, axis=0)
summary_p16 = np.nanpercentile(summary_samples, 15.865, axis=0)
summary_p84 = np.nanpercentile(summary_samples, 84.135, axis=0)
summary_minus_1sigma = summary_med - summary_p16
summary_plus_1sigma = summary_p84 - summary_med
bestfit_df = pd.DataFrame(
    {
        "parameter": summary_labels,
        "median": summary_med,
        "minus_1sigma": summary_minus_1sigma,
        "plus_1sigma": summary_plus_1sigma,
    }
)
csv_name = output_dir / f"{planet_name}_{model_name}_bestfit{output_suffix}.csv"
bestfit_df.to_csv(csv_name, index=False)
print(f"best-fit csv: {csv_name}")

results_path = output_dir / f"{planet_name}_{model_name}_results{output_suffix}.txt"
if use_combined and gr_metrics is None:
    print(f"warning: {gr_warning}")
    print(f"results skipped: {results_path}")
else:
    write_results_txt(
        results_path,
        planet_name,
        model_name,
        summary_labels,
        summary_samples,
        out,
        lc_csvfiles,
        rv_csvfile,
        ncores,
        len(p0),
        {"fit_b": fit_b, "fit_e": fit_e, "fit_K": fit_K, "fit_ld": fit_ld},
        gr_metrics=gr_metrics,
        gr_warning=gr_warning,
    )
    print(f"results written: {results_path}")

trace_x = out.whichlink
trace_samples = out.flatchains
if use_combined:
    trace_samples = samples_for_outputs
    trace_x = np.arange(trace_samples.shape[0])

fig1, axes1 = plt.subplots(len(p0), figsize=(10, 1 + 2 * len(p0)), sharex=True)
for i in range(len(p0)):
    ax = axes1[i]
    ax.plot(trace_x, trace_samples[:, i], ".", alpha=0.2, rasterized=True)
    ax.set_ylabel(labels[i])
    if all(parinfo[i]["limited"]):
        ylim_pad = 0.05 * (parinfo[i]["limits"][1] - parinfo[i]["limits"][0])
        ax.set_ylim(parinfo[i]["limits"][0] - ylim_pad, parinfo[i]["limits"][1] + ylim_pad)
    else:
        finite_trace = trace_samples[np.isfinite(trace_samples[:, i]), i]
        if finite_trace.size > 0:
            trace_min = np.nanmin(finite_trace)
            trace_max = np.nanmax(finite_trace)
            trace_span = trace_max - trace_min
            if trace_span <= 0.0:
                ylim_pad = max(1e-6, 0.05 * max(1.0, abs(trace_min)))
            else:
                ylim_pad = 0.05 * trace_span
            ax.set_ylim(trace_min - ylim_pad, trace_max + ylim_pad)
axes1[-1].set_xlabel("Link number")
fig1_name = output_dir / f"{planet_name}_{model_name}_trace{output_suffix}.pdf"
fig1.savefig(fig1_name)
plt.close(fig1)

plot_samples, plot_labels = filter_constant_plot_columns(plot_samples, plot_labels)
corner_display_labels = get_plot_display_labels(plot_labels)
fig2 = corner.corner(plot_samples, labels=corner_display_labels, label_kwargs={"fontsize": corner_label_fontsize})
apply_corner_scalar_formatting(fig2, corner_display_labels, corner_label_fontsize)
for ax in fig2.axes:
    for artist in list(ax.collections) + list(ax.images) + list(ax.patches):
        if not isinstance(artist, QuadContourSet):
            artist.set_rasterized(True)
fig2_name = output_dir / f"{planet_name}_{model_name}_corner{output_suffix}.pdf"
fig2.savefig(fig2_name, bbox_inches='tight', pad_inches=0.3)
plt.close(fig2)


# %%
# ------------------------------------------------------------
# Full Fit Plot
# ------------------------------------------------------------
med_unpacked = unpack_params(sampled_med)
if med_unpacked is None:
    raise ValueError("Median parameters are invalid; cannot build plots.")

t0_bjd_med, per_med, rp_med, a_over_med, inc_med, ecc_med, omega_med, vsini_med, lambda_med, K_med, ld_med = med_unpacked

best_lc = batman_flux(t0_bjd_med, per_med, rp_med, a_over_med, inc_med, ecc_med, omega_med, ld_med, lc_time)
best_rv = rv_model(t0_bjd_med, per_med, rp_med, a_over_med, inc_med, ecc_med, omega_med, vsini_med, lambda_med, K_med, rv_time_all, rv_exptime_s_all)

lc_phase = ((lc_time - (t0_bjd_med - 2457000.0)) / per_med + 0.5) % 1.0 - 0.5
rv_phase_all = ((rv_time_all - t0_bjd_med) / per_med + 0.5) % 1.0 - 0.5
rv_phase = rv_phase_all[~omit_rv_mask_all]
rv_phase_omitted = rv_phase_all[omit_rv_mask_all]

lc_sort = np.argsort(lc_phase)
rv_sort = np.argsort(rv_phase_all)

# Posterior bands
all_samples = samples_for_outputs
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

lc_residuals = lc_flux - lc_median
lc_plot_window = 0.05

lc_binned_phase = np.array([])
lc_binned_flux = np.array([])
lc_binned_flux_err = np.array([])
lc_binned_residual = np.array([])
lc_binned_residual_err = np.array([])
if show_lc_binned_points:
    lc_bin_mask = np.abs(lc_phase) <= lc_plot_window
    lc_phase_bins = np.arange(-lc_plot_window, lc_plot_window + lc_phase_bin_width, lc_phase_bin_width)
    if lc_phase_bins[-1] < lc_plot_window:
        lc_phase_bins = np.append(lc_phase_bins, lc_plot_window)
    lc_bin_index = np.digitize(lc_phase[lc_bin_mask], lc_phase_bins) - 1

    binned_phase = []
    binned_flux = []
    binned_flux_err = []
    binned_residual = []
    binned_residual_err = []
    phase_in_window = lc_phase[lc_bin_mask]
    flux_in_window = lc_flux[lc_bin_mask]
    flux_err_in_window = lc_flux_err[lc_bin_mask]
    residual_in_window = lc_residuals[lc_bin_mask]
    for i in range(len(lc_phase_bins) - 1):
        in_bin = lc_bin_index == i
        if not np.any(in_bin):
            continue
        weights = 1.0 / np.maximum(flux_err_in_window[in_bin], 1e-12) ** 2
        weight_sum = np.sum(weights)
        binned_phase.append(np.sum(weights * phase_in_window[in_bin]) / weight_sum)
        binned_flux.append(np.sum(weights * flux_in_window[in_bin]) / weight_sum)
        binned_flux_err.append(np.sqrt(1.0 / weight_sum))
        binned_residual.append(np.sum(weights * residual_in_window[in_bin]) / weight_sum)
        binned_residual_err.append(np.sqrt(1.0 / weight_sum))

    lc_binned_phase = np.asarray(binned_phase)
    lc_binned_flux = np.asarray(binned_flux)
    lc_binned_flux_err = np.asarray(binned_flux_err)
    lc_binned_residual = np.asarray(binned_residual)
    lc_binned_residual_err = np.asarray(binned_residual_err)

# RV posterior models
ntime = len(rv_time_all)
models = np.zeros((nsamp, ntime))
for j, idx in enumerate(sel_idx):
    unpacked = unpack_params(all_samples[idx, :])
    if unpacked is None:
        models[j, :] = np.nan
        continue
    t0_bjd_s, per_s, rp_s, a_over_s, inc_s, ecc_s, omega_s, vsini_s, lambda_s, K_s, _ = unpacked
    models[j, :] = rv_model(t0_bjd_s, per_s, rp_s, a_over_s, inc_s, ecc_s, omega_s, vsini_s, lambda_s, K_s, rv_time_all, rv_exptime_s_all)

median_model = np.nanmedian(models, axis=0)
p16 = np.nanpercentile(models, 16.0, axis=0)
p84 = np.nanpercentile(models, 84.0, axis=0)
p025 = np.nanpercentile(models, 2.5, axis=0)
p975 = np.nanpercentile(models, 97.5, axis=0)

font_choice = 'serif'    # maybe change to 'Times New Roman?'
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
    alpha=lc_raw_alpha,
    color="k",
    label=lc_raw_label,
    zorder=2,
)
ax_lc_top.plot(
    lc_phase[lc_sort],
    lc_median[lc_sort],
    color="red",
    lw=model_linewidth,
    label=lc_model_label,
    zorder=4,
)
if show_lc_binned_points and lc_binned_phase.size > 0:
    ax_lc_top.errorbar(
        lc_binned_phase,
        lc_binned_flux,
        yerr=lc_binned_flux_err,
        fmt='o',
        ms=marker_size,
        mfc='k',
        mec='k',
        ecolor='k',
        alpha=1.0,
        label=lc_binned_label,
        zorder=5,
    )
ax_lc_top.set_xlim(-lc_plot_window, lc_plot_window)
ax_lc_top.set_ylim(top=1.01)
ax_lc_top.tick_params(axis='both', labelsize=tick_fontsize)
ax_lc_top.tick_params(axis='x', labelbottom=False)
lc_legend_handles = [
    Line2D([], [], linestyle='None', marker='.', color='k', alpha=lc_raw_alpha, markersize=marker_size, label=lc_raw_label),
    Line2D([], [], color='red', lw=model_linewidth, label=lc_model_label),
]
if show_lc_binned_points and lc_binned_phase.size > 0:
    lc_legend_handles.append(
        Line2D([], [], linestyle='None', marker='o', color='k', markerfacecolor='k', markeredgecolor='k',
               markersize=marker_size, label=lc_binned_label)
    )
ax_lc_top.legend(handles=lc_legend_handles, prop={'size': legend_fontsize, 'family': font_choice}, loc='upper right')

ax_lc_bot.errorbar(
    lc_phase,
    lc_residuals,
    yerr=lc_flux_err,
    fmt=".",
    ms=marker_size,
    alpha=lc_raw_alpha,
    color="k",
    zorder=2,
)
ax_lc_bot.axhline(0.0, color='red', linestyle='-', alpha=0.7, zorder=4)
if show_lc_binned_points and lc_binned_phase.size > 0:
    ax_lc_bot.errorbar(
        lc_binned_phase,
        lc_binned_residual,
        yerr=lc_binned_residual_err,
        fmt='o',
        ms=marker_size,
        mfc='k',
        mec='k',
        ecolor='k',
        alpha=1.0,
        zorder=5,
    )
ax_lc_bot.set_xlabel("Phase", fontsize=label_fontsize, fontname=font_choice)
ax_lc_bot.set_xlim(-lc_plot_window, lc_plot_window)
ax_lc_bot.tick_params(axis='both', labelsize=tick_fontsize)

for ax in (ax_lc_top, ax_lc_bot):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(font_choice)

# Right panel: RV with residuals
sub = gs[0, 1].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
ax_top = fig3.add_subplot(sub[0, 0])
ax_bot = fig3.add_subplot(sub[1, 0], sharex=ax_top)

# 2-sigma and 1-sigma bands
ax_top.fill_between(rv_phase_all[rv_sort], p025[rv_sort] * 1e3, p975[rv_sort] * 1e3,
                    color='red', alpha=band_alpha_2sig, linewidth=0.0)
ax_top.fill_between(rv_phase_all[rv_sort], p16[rv_sort] * 1e3, p84[rv_sort] * 1e3,
                    color='red', alpha=band_alpha_1sig, linewidth=0.0)

# Data and median model
if rv_phase_omitted.size > 0:
    ax_top.errorbar(
        rv_phase_omitted,
        rv_data_omitted * 1e3,
        yerr=rv_err_omitted * 1e3,
        fmt='o',
        ms=marker_size,
        c='k',
        alpha=omitted_rv_alpha,
        zorder=4
    )
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
    rv_phase_all[rv_sort],
    median_model[rv_sort] * 1e3,
    '-',
    lw=model_linewidth,
    alpha=1.0,
    c='red',
    label='Median Model'
)
ax_top.tick_params(axis='both', labelsize=tick_fontsize)
ax_top.tick_params(axis='x', labelbottom=False)
ax_top.legend(prop={'size': legend_fontsize, 'family': font_choice}, loc='best')

# Residuals
residuals_ms_all = (rv_data_all - median_model) * 1e3
residuals_ms = residuals_ms_all[~omit_rv_mask_all]
residuals_ms_omitted = residuals_ms_all[omit_rv_mask_all]
if rv_phase_omitted.size > 0:
    ax_bot.errorbar(
        rv_phase_omitted,
        residuals_ms_omitted,
        yerr=rv_err_omitted * 1e3,
        fmt='o',
        ms=marker_size,
        c='k',
        alpha=omitted_rv_alpha,
    )
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
fig3_name = output_dir / f"{planet_name}_{model_name}_phase_lc_rv_side_by_side{output_suffix}.pdf"
fig3.savefig(fig3_name)
plt.close(fig3)
