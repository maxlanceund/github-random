import h5py
import csv

h5_path = "./GWTC-1_sample_release/GW170817_GWTC-1.hdf5"
out_name = "gw170817_lambda.csv"

with h5py.File(h5_path, 'r') as f:
    grp = f['IMRPhenomPv2NRT_lowSpin_posterior']['samples']
    m1 = grp['mass_1']
    m2 = grp['mass_2']
    lamt = grp['lambda_tilde']

with open(out_name, 'w', newline='') as fw:
    w = csv.writer(fw)
    w.writerow(["mass_1", "mass_2", "lambda_tilde"])
    for a,b,c in zip(m1,m2,lamt):
        w.writerow([float(a), float(b), float(c)])
print("csv generated successfully")
