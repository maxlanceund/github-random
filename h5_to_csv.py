import h5py
import numpy as np
import urllib.request
import os
import csv

file_name = "GW150914_posterior.h5"
if not os.path.exists(file_name):
    print("正在云端下载 129M 真实黑洞数据...")
    url = "https://zenodo.org/api/files/ecf41927-9275-47da-8b37-e299693fe5cb/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
    urllib.request.urlretrieve(url, file_name)
    print("下载完成！")

print("正在提取黑洞质量和自旋数据...")
with h5py.File(file_name, 'r') as f:
    group_name = list(f.keys())[0]
    g = f[group_name]
    masses = g['final_mass_source'][:10000]
    spins = g['final_spin'][:10000]

csv_file = "blackhole_data.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Mass_Msun", "Spin"])
    for m, s in zip(masses, spins):
        writer.writerow([f"{m:.4f}", f"{s:.4f}"])

print(f"✅ 转换成功！已生成 {csv_file}！")
