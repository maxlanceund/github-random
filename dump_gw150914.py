import h5py
import pandas as pd
import os, glob

# 解压之后目录
base_dir = "GWTC-1_posteriors"
h5_list = glob.glob(os.path.join(base_dir, "GW150914", "*.h5"))
print("找到h5：", h5_list)
h5_path = h5_list[0]

with h5py.File(h5_path, "r") as f:
    grp = f["IMRPhenomPv2NRT_lowSpin_posterior"]
    m1 = grp["mass_1"][:]
    m2 = grp["mass_2"][:]
    chi1z = grp["spin_1z"][:]
    chi2z = grp["spin_2z"][:]

df = pd.DataFrame({"m1":m1,"m2":m2,"chi1z":chi1z,"chi2z":chi2z})
df = df.sample(n=5000, random_state=42, ignore_index=True)
df.to_csv("posterior_samples_GW150914.csv", index=False)
print("✅ 导出 posterior_samples_GW150914.csv")
