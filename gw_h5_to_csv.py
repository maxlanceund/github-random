import h5py
import pandas as pd

IN_H5 = "GW170817_GWTC-1.hdf5"
OUT_CSV = "gw170817_posterior.csv"

with h5py.File(IN_H5, "r") as f:
    samples = f["posterior_samples"]
    df = pd.DataFrame({
        "mass_1": samples["mass_1"][:],
        "mass_2": samples["mass_2"][:],
        "lambda_tilde": samples["lambda_tilde"][:]
    })

df.to_csv(OUT_CSV, index=False)
print(f"Done. Saved {OUT_CSV}, samples = {len(df)}")
