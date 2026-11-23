import h5py
import pandas as pd
import numpy as np
import subprocess
import os

local_h5 = "GW150914-v3_posterior.h5"
out_csv = "posterior_samples_GW150914.csv"

url = "https://zenodo.org/api/files/440a1f04-3c75-4875-bb85-44830842006c/GW150914-v3_posterior.h5"

# wget带上UA头绕过zenodo防盗链
cmd = [
    "wget",
    "--user-agent=Mozilla/5.0",
    "-O", local_h5,
    url
]

print("Downloading GW150914 posterior h5 via wget ...")
ret = subprocess.run(cmd, capture_output=True)
print("wget return code:", ret.returncode)

if not os.path.exists(local_h5):
    raise SystemExit("ERROR: file did not download")

# 检查文件大小，如果很小说明下载失败
sz = os.path.getsize(local_h5)
print(f"Downloaded file size bytes: {sz}")
if sz < 50000:
    raise SystemExit("ERROR: file too small, likely html instead of h5")

with h5py.File(local_h5, "r") as f:
    m1 = f["posterior_samples"]["mass_1"][:]
    m2 = f["posterior_samples"]["mass_2"][:]
    chi1z = f["posterior_samples"]["spin_1z"][:]
    chi2z = f["posterior_samples"]["spin_2z"][:]

df = pd.DataFrame({
    "m1": m1,
    "m2": m2,
    "chi1z": chi1z,
    "chi2z": chi2z
})
df = df.sample(n=5000, random_state=42, ignore_index=True)
df.to_csv(out_csv, index=False)
print(f"Exported {out_csv}, total {len(df)} samples")
