import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp, norm

# Pre-computed TOV tables
m_hc = np.array([0.14,0.31,0.52,0.74,0.94,1.11,1.26,1.38,1.48,1.56,1.61,1.64,1.65])
lam_hc = np.array([1820,970,540,310,185,115,78,58,44,34,27,22,20])

m_sly = np.array([0.20,0.40,0.65,0.90,1.10,1.30,1.45,1.60,1.75,1.90,2.00,2.05])
lam_sly = np.array([2200,1180,660,380,240,160,110,78,54,36,26,21])

m_apr = np.array([0.22,0.45,0.70,0.95,1.15,1.35,1.50,1.65,1.80,1.95,2.10,2.20])
lam_apr = np.array([2050,1090,610,350,220,145,100,70,48,32,21,16])

m_dd2 = np.array([0.25,0.50,0.75,1.00,1.20,1.40,1.60,1.80,2.00,2.20,2.35,2.42])
lam_dd2 = np.array([3400,1820,1030,600,390,260,175,118,78,48,28,18])

def make_interp(m_arr, val_arr):
    return interp1d(m_arr, val_arr, kind="linear", bounds_error=False, fill_value=np.nan)

interp_lam_hc = make_interp(m_hc, lam_hc)
interp_lam_sly = make_interp(m_sly, lam_sly)
interp_lam_apr = make_interp(m_apr, lam_apr)
interp_lam_dd2 = make_interp(m_dd2, lam_dd2)

r14_hc=10.5
r14_sly=11.7
r14_apr=11.4
r14_dd2=13.2

# 复用仓库已有gw170817_lambda.csv
df_ns = pd.read_csv("./gw170817_lambda.csv")
m1_obs = df_ns["mass_1_source"].values
m2_obs = df_ns["mass_2_source"].values
lam_obs = df_ns["lambda_tilde"].values

def compute_lambda_tilde(m1,m2,lam1,lam2):
    mtot = m1+m2
    eta = (m1*m2)/(mtot**2)
    term1 = (1+7*eta-31*eta**2)*(lam1+lam2)
    term2 = (1-9*eta)*(lam1-lam2)
    return (8.0/13.0)*(term1+term2)

def predict_lambda_tilde_vec(m1_arr,m2_arr, interp_lam):
    l1 = interp_lam(m1_arr)
    l2 = interp_lam(m2_arr)
    mtot = m1_arr + m2_arr
    eta = (m1_arr*m2_arr)/(mtot**2)
    term1 = (1+7*eta-31*eta**2)*(l1+l2)
    term2 = (1-9*eta)*(l1-l2)
    lt = (8.0/13.0)*(term1+term2)
    return lt

lam_tilde_hc_pred = predict_lambda_tilde_vec(m1_obs,m2_obs, interp_lam_hc)
lam_tilde_sly_pred = predict_lambda_tilde_vec(m1_obs,m2_obs, interp_lam_sly)
lam_tilde_apr_pred = predict_lambda_tilde_vec(m1_obs,m2_obs, interp_lam_apr)
lam_tilde_dd2_pred = predict_lambda_tilde_vec(m1_obs,m2_obs, interp_lam_dd2)

# 输出npz给Termux绘图
np.savez("ppd_arrays.npz",
         lam_obs=lam_obs,
         lam_tilde_hc_pred=lam_tilde_hc_pred,
         lam_tilde_sly_pred=lam_tilde_sly_pred,
         lam_tilde_apr_pred=lam_tilde_apr_pred,
         lam_tilde_dd2_pred=lam_tilde_dd2_pred)

R14_mu = 12.75
R14_sigma = 0.65
nicer_like_hc = norm.pdf(r14_hc, loc=R14_mu, scale=R14_sigma)
nicer_like_sly = norm.pdf(r14_sly, loc=R14_mu, scale=R14_sigma)
nicer_like_apr = norm.pdf(r14_apr, loc=R14_mu, scale=R14_sigma)
nicer_like_dd2 = norm.pdf(r14_dd2, loc=R14_mu, scale=R14_sigma)

def ks_compare(obs, pred):
    ok = np.isfinite(pred)
    o = obs[ok]
    p = pred[ok]
    if len(o)<10 or len(p)<10:
        return np.nan, np.nan, len(p), np.sum(ok)/len(pred)
    stat, pval = ks_2samp(o, p)
    return stat, pval, len(p), np.sum(ok)/len(pred)

ks_hc, p_hc, n_ok_hc, frac_ok_hc = ks_compare(lam_obs, lam_tilde_hc_pred)
ks_sly, p_sly, n_ok_sly, frac_ok_sly = ks_compare(lam_obs, lam_tilde_sly_pred)
ks_apr, p_apr, n_ok_apr, frac_ok_apr = ks_compare(lam_obs, lam_tilde_apr_pred)
ks_dd2, p_dd2, n_ok_dd2, frac_ok_dd2 = ks_compare(lam_obs, lam_tilde_dd2_pred)

out_lines = []
out_lines.append("="*80)
out_lines.append("PPD‑KS test summary (obs=GW170817 tilde‑Lambda posterior)")
row_fmt="{:<12} {:<12.4f} {:<12.4f} {:<14d} {:<12.3f} {:<12.2f} {:<12.4e}"
header = f"{'EOS':<12} {'KS-stat':<12} {'p-value':<12} {'valid_samples':<14} {'frac_valid':<12} {'NICER_R14':<12} {'NICER_like':<12}"
out_lines.append(header)
out_lines.append(row_fmt.format("Hard-core", ks_hc, p_hc, int(n_ok_hc), frac_ok_hc, r14_hc, nicer_like_hc))
out_lines.append(row_fmt.format("SLy", ks_sly, p_sly, int(n_ok_sly), frac_ok_sly, r14_sly, nicer_like_sly))
out_lines.append(row_fmt.format("APR", ks_apr, p_apr, int(n_ok_apr), frac_ok_apr, r14_apr, nicer_like_apr))
out_lines.append(row_fmt.format("DD2", ks_dd2, p_dd2, int(n_ok_dd2), frac_ok_dd2, r14_dd2, nicer_like_dd2))
out_lines.append("="*80)

for line in out_lines:
    print(line)

with open("ppd_result.txt","w",encoding="utf-8") as f:
    f.write("\n".join(out_lines))
