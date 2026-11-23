import h5py

h5_path = "./GWTC-1_sample_release/GW150914_GWTC-1.hdf5"
with h5py.File(h5_path, "r") as f:
    print("==== GW150914 HDF5 顶层全部 keys ====")
    top_keys = list(f.keys())
    for k in top_keys:
        print(f"- {k}")
    print("\n==== 每个key内部结构预览 ====")
    for k in top_keys:
        try:
            subkeys = list(f[k].keys())
            print(f"【{k}】子项：{subkeys[:10]}")
        except Exception:
            print(f"【{k}】不是分组，是数据集")
