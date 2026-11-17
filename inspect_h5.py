import h5py

IN_H5 = "GWTC-1_sample_release/GW170817_GWTC-1.hdf5"

def print_name(name):
    print(name)

with h5py.File(IN_H5, "r") as f:
    print("===== 顶层所有key =====")
    f.visit(print_name)
