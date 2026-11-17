import h5py

h5_path = "./GWTC-1_sample_release/GW170817_GWTC-1.hdf5"

with h5py.File(h5_path, 'r') as f:
    ds = f["IMRPhenomPv2NRT_lowSpin_posterior"]
    print("=== dataset dtype 全部字段名称 ===")
    print(ds.dtype.names)
