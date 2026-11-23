import h5py
import pandas as pd
import numpy as np

# 解压后路径，和你 inspect_h5.py 看到的结构一致
h5_path = "GWTC-1_sample_release/GW150914/GW150914_posterior.h5"
out_csv = "posterior_samples_GW150914.csv"

with h5py.File(h5_path, "r") as f:
    # 对应 IMRPhenomPv2_lowSpin_posterior 或者 highSpin，和你GW170817对齐
    grp = f["IMRPhenomPv2_lowSpin_posterior"]
    m1 = grp["mass_1"][:]
    m2 = grp["mass_2"][:]
    chi1z = grp["spin_1z"][:]
    chi2z = grp["spin_2z"][:]

df = pd.DataFrame({
    "m1": m1,
    "m2": m2,
    "chi1z": chi1z,
    "chi2z": chi2z
})
df = df.sample(n=5000, random_state=42, ignore_index=True)
df.to_csv(out_csv, index=False)
print(f"saved {out_csv}, samples = {len(df)}")
