import h5py
import pandas as pd

IN_H5 = "GWTC-1_sample_release/GW170817_GWTC-1.hdf5"
OUT_CSV = "gw170817_posterior.csv"

with h5py.File(IN_H5, "r") as f:
    # 这个GWTC-1打包文件里面低自旋组：IMRPhenomPv2NRT_lowSpin_posterior
    grp = f["IMRPhenomPv2NRT_lowSpin_posterior"]
    df = pd.DataFrame({
        "mass_1": grp["mass_1"][:],
        "mass_2": grp["mass_2"][:],
        "lambda_tilde": grp["lambda_tilde"][:]
    })

df.to_csv(OUT_CSV, index=False)
print(f"OK, extracted {len(df)} samples to {OUT_CSV}")
