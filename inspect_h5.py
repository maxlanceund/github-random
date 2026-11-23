import h5py
h5_path = "./GWTC-1_sample_release/GW150914_GWTC-1.hdf5"
with h5py.File(h5_path, "r") as f:
    print("===== GW150914 顶层所有key =====")
    for k in list(f.keys()):
        print(k)
