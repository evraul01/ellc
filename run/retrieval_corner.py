"""Regenerate TOI-4137b corner plots from saved EDMCMC chains."""

from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.contour import QuadContourSet
from matplotlib.ticker import ScalarFormatter


R_SUN_TO_R_EARTH = 109.076
R_SUN_TO_AU = 0.00465047
TOI_4137_STELLAR_RADIUS_RSUN = 11.228979169999995236 * 0.1280615074825186

CORNER_LABEL_MAP = {
    "t0_bjd": r"$t_0$ (BJD)",
    "period_d": r"$P$ (d)",
    "rp_over_rstar": r"$R_{\rm p}/R_\star$",
    "a_over_rstar": r"$a/R_\star$",
    "impact_b": r"$b$",
    "inc_deg": r"$i$ ($^\circ$)",
    "sqrt_e_cosw": r"$\sqrt{e}\cos\omega_\star$",
    "sqrt_e_sinw": r"$\sqrt{e}\sin\omega_\star$",
    "eccentricity": r"$e$",
    "omega_deg": r"$\omega_\star$ ($^\circ$)",
    "vsini": r"$v \sin i_\star$ (km s$^{-1}$)",
    "lambda": r"$\lambda$ ($^\circ$)",
    "K": r"$K$ (km s$^{-1}$)",
    "K_m_s": r"$K$ (m s$^{-1}$)",
    "gamma_rv": r"$\gamma$ (km s$^{-1}$)",
    "rv_slope_mps_day": r"$\dot{\gamma}$ (m s$^{-1}$ d$^{-1}$)",
    "log_rv_jitter_mps": r"$\ln\sigma_{\rm RV}$",
    "rv_jitter_mps": r"$\sigma_{\rm RV}$ (m s$^{-1}$)",
    "ld_u1": r"$u_1$",
    "ld_u2": r"$u_2$",
    "ground_ld_u1": r"$u_{1,i'}$",
    "ground_ld_u2": r"$u_{2,i'}$",
    "ground_flux_baseline": r"$F_{0,i'}$",
    "ground_airmass_coeff": r"$C_{\rm air,i'}$",
    "ground_width_coeff": r"$C_{\rm width,i'}$",
    "rho_star_g_cm3": r"$\rho_\star$ (g cm$^{-3}$)",
    "incl_deg_derived": r"$i$ ($^\circ$)",
    "planet_radius_rearth": r"$R_{\rm p}$ ($R_\oplus$)",
    "semi_major_axis_au": r"$a$ (au)",
}


def _scalar_formatter():
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


def _format_corner_axes(fig, display_labels, label_fontsize):
    ndim = len(display_labels)
    axes = np.asarray(fig.axes).reshape((ndim, ndim))
    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]
            ax.xaxis.set_major_formatter(_scalar_formatter())
            if row > col:
                ax.yaxis.set_major_formatter(_scalar_formatter())

    fig.canvas.draw()
    for col in range(ndim):
        ax = axes[-1, col]
        offset = ax.xaxis.get_offset_text().get_text().strip()
        if offset:
            ax.set_xlabel(
                _merge_offset_into_label(display_labels[col], offset),
                fontsize=label_fontsize,
                labelpad=12,
            )
            ax.xaxis.get_offset_text().set_visible(False)
    for row in range(1, ndim):
        ax = axes[row, 0]
        offset = ax.yaxis.get_offset_text().get_text().strip()
        if offset:
            ax.set_ylabel(
                _merge_offset_into_label(display_labels[row], offset),
                fontsize=label_fontsize,
                labelpad=12,
            )
            ax.yaxis.get_offset_text().set_visible(False)


def _load_saved_samples(output_dir, planet_name, model_name, prefer_full_chains=False):
    output_dir = Path(output_dir)
    thin_path = output_dir / f"{planet_name}_{model_name}_chains.npz"
    full_path = output_dir / f"{planet_name}_{model_name}_chains_full.npz"

    if prefer_full_chains and full_path.exists():
        with np.load(full_path, allow_pickle=True) as saved:
            labels = [str(label) for label in saved["labels"]]
            fullchains = np.asarray(saved["fullchains"], dtype=float)
            burnin = int(saved["posterior_burnin"]) if "posterior_burnin" in saved else int(saved["nburnin"])
            thin_n = int(saved["thin_n"]) if "thin_n" in saved else 1
        indices = np.arange(burnin, fullchains.shape[1], thin_n, dtype=int)
        samples = fullchains[:, indices, :].reshape(-1, fullchains.shape[2])
        source_path = full_path
    else:
        if not thin_path.exists():
            raise FileNotFoundError(f"Saved chain file not found: {thin_path}")
        with np.load(thin_path, allow_pickle=True) as saved:
            if "thinflatchains" not in saved or "labels" not in saved:
                raise ValueError(f"Saved chain file is missing samples or labels: {thin_path}")
            samples = np.asarray(saved["thinflatchains"], dtype=float)
            labels = [str(label) for label in saved["labels"]]
        source_path = thin_path

    if samples.ndim != 2 or samples.shape[1] != len(labels):
        raise ValueError(f"Saved chain dimensions do not match labels in {source_path}")
    return samples, labels, source_path


def _add_derived_parameters(samples, labels, stellar_radius_rsun):
    labels = list(labels)
    columns = {name: samples[:, i] for i, name in enumerate(labels)}
    derived = []
    derived_labels = []

    if "sqrt_e_cosw" in columns and "sqrt_e_sinw" in columns:
        x = columns["sqrt_e_cosw"]
        y = columns["sqrt_e_sinw"]
        eccentricity = x**2 + y**2
        omega_deg = np.degrees(np.arctan2(y, x))
        derived.extend([eccentricity, omega_deg])
        derived_labels.extend(["eccentricity", "omega_deg"])
    else:
        eccentricity = np.zeros(samples.shape[0])
        omega_deg = np.zeros(samples.shape[0])

    if "period_d" in columns and "a_over_rstar" in columns:
        period_seconds = columns["period_d"] * 86400.0
        rho_star = 3.0 * np.pi * columns["a_over_rstar"] ** 3 / (6.6743e-8 * period_seconds**2)
        derived.append(rho_star)
        derived_labels.append("rho_star_g_cm3")

    if "impact_b" in columns and "a_over_rstar" in columns:
        fac = (1.0 - eccentricity**2) / (
            1.0 + eccentricity * np.sin(np.deg2rad(omega_deg))
        )
        cosi = columns["impact_b"] / (columns["a_over_rstar"] * fac)
        inclination = np.degrees(np.arccos(np.clip(cosi, 0.0, 1.0)))
        derived.append(inclination)
        derived_labels.append("incl_deg_derived")

    if "rp_over_rstar" in columns:
        planet_radius = columns["rp_over_rstar"] * stellar_radius_rsun * R_SUN_TO_R_EARTH
        derived.append(planet_radius)
        derived_labels.append("planet_radius_rearth")

    if "a_over_rstar" in columns:
        semi_major_axis = columns["a_over_rstar"] * stellar_radius_rsun * R_SUN_TO_AU
        derived.append(semi_major_axis)
        derived_labels.append("semi_major_axis_au")

    if derived:
        samples = np.column_stack([samples, *derived])
        labels.extend(derived_labels)
    return samples, labels


def _transform_reported_parameters(samples, labels):
    samples = np.array(samples, copy=True)
    labels = list(labels)
    for i, name in enumerate(labels):
        if name == "K":
            samples[:, i] *= 1e3
            labels[i] = "K_m_s"
        elif name == "log_rv_jitter_mps":
            samples[:, i] = np.exp(samples[:, i])
            labels[i] = "rv_jitter_mps"
    return samples, labels


def _drop_invalid_or_constant_columns(samples, labels, tolerance=1e-15):
    finite_rows = np.all(np.isfinite(samples), axis=1)
    samples = samples[finite_rows]
    if samples.size == 0:
        raise ValueError("No finite posterior samples are available for the corner plot.")

    keep = np.zeros(samples.shape[1], dtype=bool)
    for i in range(samples.shape[1]):
        span = np.ptp(samples[:, i])
        scale = max(1.0, np.max(np.abs(samples[:, i])))
        keep[i] = span > tolerance * scale
    if not np.any(keep):
        raise ValueError("No non-constant posterior parameters are available for plotting.")
    return samples[:, keep], [labels[i] for i in range(len(labels)) if keep[i]]


def regenerate_corner_plot(
    output_dir,
    model_name,
    planet_name="TOI-4137b",
    max_samples=50000,
    plot_datapoints=False,
    random_seed=12345,
    label_fontsize=18,
    include_derived=True,
    prefer_full_chains=False,
    stellar_radius_rsun=TOI_4137_STELLAR_RADIUS_RSUN,
    output_path=None,
):
    """Load a completed retrieval and regenerate only its corner PDF."""
    samples, labels, source_path = _load_saved_samples(
        output_dir,
        planet_name,
        model_name,
        prefer_full_chains=prefer_full_chains,
    )
    total_samples = samples.shape[0]

    if max_samples is not None and total_samples > max_samples:
        rng = np.random.default_rng(random_seed)
        indices = rng.choice(total_samples, size=max_samples, replace=False)
        samples = samples[indices]

    if include_derived:
        samples, labels = _add_derived_parameters(samples, labels, stellar_radius_rsun)
    samples, labels = _transform_reported_parameters(samples, labels)
    samples, labels = _drop_invalid_or_constant_columns(samples, labels)

    display_labels = [CORNER_LABEL_MAP.get(name, name) for name in labels]
    fig = corner.corner(
        samples,
        labels=display_labels,
        label_kwargs={"fontsize": label_fontsize},
        plot_datapoints=plot_datapoints,
        pcolor_kwargs={"rasterized": True},
    )
    _format_corner_axes(fig, display_labels, label_fontsize)
    for ax in fig.axes:
        for artist in list(ax.collections) + list(ax.images) + list(ax.patches):
            if not isinstance(artist, QuadContourSet):
                artist.set_rasterized(True)

    if output_path is None:
        output_path = Path(output_dir) / f"{planet_name}_{model_name}_corner.pdf"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)

    print(f"Loaded {total_samples} posterior samples from {source_path}")
    print(f"Corner plot used {samples.shape[0]} samples and {samples.shape[1]} parameters")
    print(f"Corner plot written: {output_path}")
    return output_path

