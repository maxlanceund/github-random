import pandas as pd
from gwpy.timeseries import TimeSeries
from gwpy.io.gwosc import get_event
from gwpy.results import read_samples

# 读取GW170817 GWTC-1后验样本
samples = read_samples("GW170817", catalog="GWTC-1")

df = pd.DataFrame({
    "mass_1": samples["mass_1"],
    "mass_2": samples["mass_2"],
    "lambda_tilde": samples["lambda_tilde"]
})

df.to_csv("gw170817_posterior.csv", index=False)
print(f"Success! Saved {len(df)} samples to gw170817_posterior.csv")
