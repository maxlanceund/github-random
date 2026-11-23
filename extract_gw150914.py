import h5py
import csv

h5_path = "./GWTC-1_sample_release/GW150914_GWTC-1.hdf5"
out_csv = "./gw150914.csv"

with h5py.File(h5_path, "r") as f:
    d = f["IMRPhenomPv2_posterior"]
    keys = list(d.keys())
    with open(out_csv, "w", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(keys)
        for row in zip(*[d[k][:] for k in keys]):
            writer.writerow(row)
