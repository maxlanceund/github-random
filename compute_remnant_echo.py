import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt  # 云端CI禁用绘图

# --------------------------
# Constants
# --------------------------
Msun = 1.98847e30       # kg
G = 6.67430e-11         # m^3 kg^-1 s^-2
c = 2.99792458e8        # m/s
lp = 1.616255e-35       # Planck length [m]

def healy_couto_remnant(m1, m2, chi1z, chi2z):
    """
    Healy & Couto (2022) BBH remnant fit (practical implementation)
    Input: component masses m1,m2 [solar mass]; component spin z‑projection chi1z, chi2z
    Return: M_rem [Msun], a_rem (dimensionless Kerr spin 0<=a<1)
    """
    q = m2 / m1
    eta = q / (1.0 + q)**2
    m_tot = m1 + m2

    # effective aligned spin
    chi_eff = (m1 * chi1z + m2 * chi2z) / (m1 + m2)

    # remnant mass fit (dimensionless radiated‑mass fraction)
    delta = 0.0441 * eta - 0.0676 * eta**2 + 0.0838 * eta * chi_eff
    M_rem = m_tot * (1.0 - delta)

    # remnant dimensionless spin
    a_rem = (0.686 * np.sqrt(eta) + 0.15 * chi_eff
             - 0.04 * chi_eff**2 + 0.02 * chi_eff**3)
    a_rem = np.clip(a_rem, 0.0, 0.998)
    return M_rem, a_rem


def kerr_echo_delay(M_rem_msun, a_rem, d=lp):
    """
    Compute Kerr corrected echo round‑trip delay Δt [seconds]
    Δt ≈ (4 G M_rem / c**3) * (1 + 1 / sqrt(1‑a_rem**2)) * ln(4 G M_rem/(c**2 * d))
    M_rem_msun: remnant mass in solar masses
    a_rem: dimensionless kerr spin
    d: proper distance of reflective wall [m], default Planck length
    """
    M_rem_kg = M_rem_msun * Msun
    term1 = (4.0 * G * M_rem_kg) / (c**3)
    spin_factor = 1.0 + 1.0 / np.sqrt(1.0 - a_rem**2)
    arg_log = (4.0 * G * M_rem_kg) / (c**2 * d)
    log_term = np.log(arg_log)
    dt_s = term1 * spin_factor * log_term
    return dt_s


def ringdown_freq_hz(M_rem_msun,a_rem):
    """QNM ring‑down frequency for l=m=2, approximate LVK formula, return Hz"""
    M_kg = M_rem_msun * Msun
    f_0 = (c**3/(2*np.pi*G*M_kg))
    omega = 1.0 - 0.63*(1-a_rem)**0.3
    f_hz = f_0 * omega
    return f_hz


# --------------------------
# Load posterior samples
# --------------------------
df = pd.read_csv("posterior_samples_GW150914.csv")

Mrem_list = []
arem_list = []
delay_list = []
ring_freq_list = []

for idx, row in df.iterrows():
    m1 = row["m1"]
    m2 = row["m2"]
    chi1z = row["chi1z"]
    chi2z = row["chi2z"]

    Mr, ar = healy_couto_remnant(m1, m2, chi1z, chi2z)
    dt = kerr_echo_delay(Mr, ar)
    fr = ringdown_freq_hz(Mr, ar)

    Mrem_list.append(Mr)
    arem_list.append(ar)
    delay_list.append(dt)
    ring_freq_list.append(fr)

df["M_rem"] = Mrem_list
df["a_rem"] = arem_list
df["echo_delay_s"] = delay_list
df["f_ringdown_Hz"] = ring_freq_list

df.to_csv("posterior_with_remnant.csv", index=False)
print("[OK] Saved posterior_with_remnant.csv")

# --------------------------
# Quick statistics for checking
# --------------------------
med_dt = np.median(df['echo_delay_s'])
p5_dt  = np.quantile(df['echo_delay_s'],0.05)
p95_dt = np.quantile(df['echo_delay_s'],0.95)
med_fd = np.median(df['f_ringdown_Hz'])

print(f"Echo delay median [s]: {med_dt:.4f}")
print(f"Echo delay 5%‑95% quantile [s]: {p5_dt:.4f} ~ {p95_dt:.4f}")
print(f"Ringdown frequency median [Hz]: {med_fd:.2f}")

# ============ 下面绘图代码全部注释，CI云端不运行绘图 ============
# plt.figure(figsize=(8,5))
# plt.hist(df["echo_delay_s"], bins=80, density=True, alpha=0.7)
# plt.xlabel("Echo delay (s)")
# plt.ylabel("Posterior density")
# plt.title("GW150914 echo delay, Kerr‑spin + NR‑fit")
# plt.tight_layout()
# plt.savefig("bh_echo_delay_new.png", dpi=200)
# plt.close()
# print("[OK] Saved bh_echo_delay_new.png")

# plt.figure(figsize=(8,5))
# plt.hist(df["f_ringdown_Hz"],bins=80,density=True,alpha=0.7,color="orange")
# plt.xlabel("Ringdown frequency [Hz]")
# plt.ylabel("Posterior density")
# plt.title("GW150914 Ringdown Frequency Posterior")
# plt.tight_layout()
# plt.savefig("ringdown_posterior.png",dpi=200)
# plt.close()
# print("[OK] Saved ringdown_posterior.png")
