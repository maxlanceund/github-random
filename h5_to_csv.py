import h5py
import numpy as np
import urllib.request
import os
import csv

file_name = "GW150914_posterior.h5"
if not os.path.exists(file_name):
    print("本地无缓存，正在下载 129M 真实黑洞数据...")
    url = "https://zenodo.org/api/files/ecf41927-9275-47da-8b37-e299693fe5cb/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
    urllib.request.urlretrieve(url, file_name)
    print("下载完成！")
else:
    print("✅ 命中云端缓存，跳过下载！")

print("正在精准提取真实的黑洞数据...")
m1_data, m2_data, spin1z, spin2z = None, None, None, None

with h5py.File(file_name, 'r') as f:
    def find_data(name, obj):
        global m1_data, m2_data, spin1z, spin2z
        if isinstance(obj, h5py.Dataset):
            if name.endswith('samples/mass_1_source') and m1_data is None:
                m1_data = obj[:10000]
            elif name.endswith('samples/mass_2_source') and m2_data is None:
                m2_data = obj[:10000]
            elif name.endswith('samples/spin_1z') and spin1z is None:
                spin1z = obj[:10000]
            elif name.endswith('samples/spin_2z') and spin2z is None:
                spin2z = obj[:10000]
    f.visititems(find_data)

if m1_data is not None and m2_data is not None:
    final_mass_est = (m1_data + m2_data) * 0.95
    final_spin_est = (m1_data * spin1z + m2_data * spin2z) / (m1_data + m2_data) 
    
    csv_file = "blackhole_data.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["M1_Msun", "M2_Msun", "Est_Final_Mass", "Est_Final_Spin"])
        for m1, m2, mf, sf in zip(m1_data, m2_data, final_mass_est, final_spin_est):
            writer.writerow([f"{m1:.4f}", f"{m2:.4f}", f"{mf:.4f}", f"{sf:.4f}"])
    print(f"✅ 提取成功！已生成 {csv_file}！")
else:
    print("❌ 失败！未找到质量参数！")
