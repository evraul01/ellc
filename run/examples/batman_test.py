# import statements
import batman
import edmcmc as edm
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii
import corner
import ellc

settings = np.seterr(over="ignore")

params = {
    "axes.labelsize": 18,
    "axes.labelpad": 9,
    "axes.titlesize": 20,
    "axes.linewidth": 2,
    "axes.labelweight": 3,
    "axes.titleweight": 3,
    "font.size": 15,
    "legend.fontsize": 15,
    "lines.linewidth": 2,
    "xtick.major.width": 2,
    "xtick.minor.width": 1,
    "xtick.major.size": 8,
    "xtick.minor.size": 5,
    "xtick.major.pad": 5,
    "xtick.labelsize": 15,
    "xtick.minor.visible": True,
    "xtick.direction": "in",
    "xtick.top": True,
    "ytick.major.width": 2,
    "ytick.minor.width": 1,
    "ytick.major.size": 8,
    "ytick.minor.size": 5,
    "ytick.major.pad": 7,
    "ytick.labelsize": 15,
    "ytick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.right": True,
    "legend.frameon": True,
    "legend.loc": "upper right",
    #'text.usetex': True,
    #'text.latex.preamble': '\usepackage{helvet}\usepackage[T1]{fontenc}\usepackage{sfmath}',
    #'font.sans-serif': "Helvetica",
    #'font.family': "sans-serif",
    "ps.usedistiller": "xpdf",
    "savefig.dpi": 300,
    "figure.figsize": [7, 7],
}

plt.rcParams.update(params)

# |%%--%%| <|tHZSC2CPSh>
file = ascii.read(
    "/home/nadja/Documents/UWMadison/Research/TOI5082/tic437011608flattened-2min.csv"
)

time = file["Time (BJD-2457000)"]
flux = file["Flux"]
flat_flux = file["Flattened Flux"]

# events vs time (non-phase folded)
plt.scatter(time, flat_flux)
plt.xlabel("Time")
plt.ylabel("Flux")
plt.show()


# Radius of planet
rad_planet = (10 ** (-3) * 0.93**2) ** 0.5  # stellar radius units
print(rad_planet)

a_over_R = 11
r_1 = 1 / a_over_R
r_2 = rad_planet * r_1

print(r_1, r_2)

orbital_period = 4.2403567  # orbital_period

# time is an array in the data file (BJD-2457000), flat_flux is also an array
phase = ((time - 2508.82) / orbital_period) - np.round(
    (time - 2508.82) / orbital_period
)

# if you want to try batman these are the parameters
params = batman.TransitParams()
params.t0 = 2508.82  # time of inferior conjunction
params.per = orbital_period  # orbital period
params.rp = rad_planet  # planet radius (in units of stellar radii)
params.a = 11  # semi-major axis (in units of stellar radii)
params.inc = 87.0  # orbital inclination (in degrees)
params.ecc = 0.0  # eccentricity
params.w = 90.0  # longitude of periastron (in degrees)
params.u = [0.1, 0.3]  # limb darkening coefficients [u1, u2]
params.limb_dark = "quadratic"  # limb darkening model

m = batman.TransitModel(params, time)
bat_flux = m.light_curve(params)


# batman and ellc are dumb and you have to sort the array or else its gonna bungle everything up
phase4 = np.copy(phase)
sort_index = np.argsort(phase4)  # this returns the indices that would sort the array

phase4 = phase4[sort_index]
bat_flux_phase = bat_flux[sort_index]

#ellc version of prior batman implementation
ellc_flux = ellc.lc(
    t_obs=time,
    radius_1=r_1,  # units of semi-major axis
    radius_2=r_2,  # units of semi-major axis
    sbratio=0,
    incl=87.0,
    t_zero=2508.82,
    period=orbital_period,
    a=11,  # semi-major axis (in units of stellar radii)
    ldc_1=[0.1, 0.3],
    f_c=0.0,
    f_s=0.0,
    ld_1="quad",
    shape_1="sphere",
    shape_2="sphere",
    verbose=False,
)

ellc_flux_phase = ellc_flux[sort_index]




# Using edmcmc for fitting the graph
# yerr calculation: std of points out of transit curve
# idx where val phase < 0.02

#adding noise to transit
def std_dev(arr_time):
    t_or_f = arr_time < -0.25
    arr_mask_1 = arr_time[t_or_f]
    masking2 = arr_time > 0.175
    arr_mask_2 = arr_time[masking2]

    # finding the values where the noise starts
    val1 = arr_mask_1[-1]
    val2 = arr_mask_2[0]

    # index where it occurs
    idx1 = np.where(arr_time == val1)
    idx2 = np.where(arr_time == val2)
    return idx1, idx2


i, j = std_dev(phase)

idx1 = i[0][0]
idx2 = j[0][0]

# print(idx1)
# print(flat_flux[idx2:])
noise_arr = np.concatenate((flat_flux[:idx1], flat_flux[idx2:]), axis=0)
error = np.nanstd(noise_arr)
yerr = np.full_like(flat_flux, error)
# print(yerr)

fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.set_ylabel("Flux")
ax1.set_xlabel("Phase")
ax1.set_xlim(-0.05, 0.05)

plt.scatter(phase, flat_flux)
plt.errorbar(phase, flat_flux, yerr=yerr, xerr=None, alpha=0.3, fmt="None")
plt.plot(phase4, bat_flux_phase, color="red")
plt.plot(phase4, ellc_flux_phase, color="black", linestyle="dashed")

plt.show()

###edmcmc
print(time)


# Commented here is the edmcmc I made for batman. Dunno if this helps 

"""def loglikelihood(p, x, y, e):
    # here p is an array of the parameters: let's define p[0] = slope, and p[1] = intercept
    params = batman.TransitParams()
    params.t0 = p[0]  # time of inferior conjunction
    if p[0] > 2508.82 + 1 or p[0] < 2508.82 - 1:
        return -np.inf
    params.per = p[1]  # orbital period
    if p[1] < 0:
        return -np.inf
    params.rp = p[2]  # planet radius (in units of stellar radii)
    if p[2] < 0:
        return -np.inf
    elif p[2] > 1:
        return -np.inf
    params.a = p[3]  # semi-major axis (in units of stellar radii)
    params.inc = p[4]  # orbital inclination (in degrees)
    if p[4] < 0:
        return -np.inf
    elif p[4] > 90:
        return -np.inf
    params.ecc = 0.0  # eccentricity
    params.w = 90  # longitude of periastron (in degrees)
    params.u = [0.1, 0.3]  # limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"  # limb darkening model

    m = batman.TransitModel(params, x)
    model = m.light_curve(params)
    chisq = np.sum((y - model) ** 2 / e**2)
    loglikelihood = -0.5 * chisq
    return loglikelihood"""


# fit one parameter first, debug from there
out = edm.edmcmc(
    loglikelihood,
    [2508.82, orbital_period, rad_planet, 11, 87],
    [0.01, 0.001, 0.001, 0.01, 0.01],
    args=(time, flat_flux, yerr),
    nwalkers=200,
    nlink=2000,
    nburnin=750,
)
# prior function (if function)
corner.corner(
    out.flatchains, labels=["Time", "period", "radius", "axis", "orb inclination"]
)
plt.show()

print(np.median(out.flatchains[:, 0]), "+/-", np.std(out.flatchains[:, 0]))
print(np.median(out.flatchains[:, 1]), "+/-", np.std(out.flatchains[:, 1]))
print(np.median(out.flatchains[:, 2]), "+/-", np.std(out.flatchains[:, 2]))
print(np.median(out.flatchains[:, 3]), "+/-", np.std(out.flatchains[:, 3]))
print(np.median(out.flatchains[:, 4]), "+/-", np.std(out.flatchains[:, 4]))

# making the model with whatever edmcmc spits out
params1 = batman.TransitParams()
params1.t0 = np.median(out.flatchains[:, 0])  # time of inferior conjunction
params1.per = np.median(out.flatchains[:, 1])  # orbital period
params1.rp = np.median(
    out.flatchains[:, 2]
)  # planet radius (in units of stellar radii)
params1.a = 11  # semi-major axis (in units of stellar radii)
params1.inc = 87.0  # orbital inclination (in degrees)
params1.ecc = 0.0  # eccentricity
params1.w = 90.0  # longitude of periastron (in degrees)
params1.u = [0.1, 0.3]  # limb darkening coefficients [u1, u2]
params1.limb_dark = "quadratic"  # limb darkening model


model = batman.TransitModel(params1, time)
unsorted_flux = model.light_curve(params1)

bat_flux_sorted = unsorted_flux[sort_index]


#Plotting said figure
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.set_ylabel("Flux")
ax2.set_xlabel("Phase")
ax2.set_xlim(-0.05, 0.05)

plt.scatter(phase, flat_flux)
plt.errorbar(phase, flat_flux, yerr=yerr, xerr=None, alpha=0.3, fmt="None")
plt.plot(phase4, bat_flux_sorted, color="red")


plt.show()


plt.plot(out.chains[:, :, 1])
plt.show()


print(out.chains.shape)