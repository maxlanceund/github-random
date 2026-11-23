import h5py
import pandas as pd

h5_path = "GW150914_posterior.h5"
out_csv = "posterior_samples_GW150914.csv"

with h5py.File(h5_path, "r") as f:
    # 和GW170817一致的分组名
    grp = f["IMRPhenomPv2NRT_lowSpin_posterior"]
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
print("✅ GW150914 csv 生成完成")
