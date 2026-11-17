import pandas as pd
from pesummary.io import read

# pesummary 直接拉 GWTC-1 GW170817 后验
samples = read("GW170817", package="GWTC-1")
post = samples.samples_dict["combined"]

df = pd.DataFrame({
    "mass_1": post["mass_1"],
    "mass_2": post["mass_2"],
    "lambda_tilde": post["lambda_tilde"]
})

df.to_csv("gw170817_posterior.csv", index=False)
print(f"OK, saved {len(df)} samples to gw170817_posterior.csv")
