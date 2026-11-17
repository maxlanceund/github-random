import h5py

h5_path = "./GWTC-1_sample_release/GW170817_GWTC-1.hdf5"

def print_h5(name, obj):
    print(name, " | ", type(obj))
    if isinstance(obj, h5py.Dataset):
        print("   dtype:", obj.dtype)
        print("   shape:", obj.shape)

with h5py.File(h5_path, 'r') as f:
    f.visititems(print_h5)
