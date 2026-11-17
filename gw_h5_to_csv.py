import h5py
import pandas as pd

IN_H5 = "GW170817_lowspin.hdf5"
OUT_CSV = "gw170817_posterior.csv"

with h5py.File(IN_H5, "r") as f:
    # 这个lowspin文件的顶层group就是 'posterior_samples'
    grp = f["posterior_samples"]
    df = pd.DataFrame({
        "mass_1": grp["mass_1"][:],
        "mass_2": grp["mass_2"][:],
        "lambda_tilde": grp["lambda_tilde"][:]
    })

df.to_csv(OUT_CSV, index=False)
print(f"Extracted {len(df)} samples, saved to {OUT_CSV}")
