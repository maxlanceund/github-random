import h5py
import pandas as pd
import numpy as np
import requests

url = "https://zenodo.org/api/files/440a1f04-3c75-4875-bb85-44830842006c/GW150914-v3_posterior.h5"
local_h5 = "GW150914-v3_posterior.h5"
out_csv = "posterior_samples_GW150914.csv"

print(f"Downloading GW150914 posterior h5 ...")
r = requests.get(url, stream=True)
with open(local_h5, "wb") as f:
    for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)
print(f"Download finished: {local_h5}")

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
