import h5py, csv, math

H5 = "./GWTC-1_sample_release/GW150914_GWTC-1.hdf5"
OUT = "gw150914_samples.csv"

H0 = 67.9
Om = 0.3065
OL = 1.0 - Om
C = 299792.458

def Ez(z):
    return math.sqrt(Om*(1+z)**3 + OL)

ZMAX = 0.05
N = 1000
DZ = ZMAX / N
TAB = [0.0]
dc = 0.0
for i in range(1, N+1):
    z0 = DZ*(i-1); z1 = DZ*i
    dc += (C/H0) * DZ * 0.5 * (1.0/Ez(z0) + 1.0/Ez(z1))
    TAB.append((1.0+z1)*dc)

def z_of_dl(dl):
    if dl <= 0: return 0.0
    if dl > TAB[-1]: return min((H0/C)*dl, 1.0)
    lo, hi = 0.0, ZMAX
    for _ in range(50):
        mid = 0.5*(lo+hi)
        idx = mid/DZ
        i0 = int(idx); i1 = min(i0+1, N); f = idx - i0
        v = TAB[i0]*(1-f) + TAB[i1]*f
        if v < dl: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

with h5py.File(H5, "r") as f:
    d = f["IMRPhenomPv2NRT_lowSpin_posterior"][:]
    m1d = d["m1_detector_frame_Msun"]
    m2d = d["m2_detector_frame_Msun"]
    chi1z = d["chi1z"]
    chi2z = d["chi2z"]
    dl  = d["luminosity_distance_Mpc"]

    n = len(m1d)
    with open(OUT, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["mass_1","mass_2","chi1z","chi2z","luminosity_distance_Mpc","redshift"])
        for i in range(n):
            z = z_of_dl(float(dl[i]))
            m1 = float(m1d[i]) / (1.0 + z)
            m2 = float(m2d[i]) / (1.0 + z)
            w.writerow([m1, m2, float(chi1z[i]), float(chi2z[i]), float(dl[i]), z])

print("csv generated, samples =", n)
