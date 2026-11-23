import h5py
import pandas as pd
import numpy as np

local_h5 = "GW150914-v3_posterior.h5"
out_csv = "posterior_samples_GW150914.csv"

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
print(f"saved {out_csv}")
