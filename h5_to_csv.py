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

print("正在深度搜索真实黑洞参数...")
masses = None
spins = None

with h5py.File(file_name, 'r') as f:
    # 定义一个搜索函数，遍历文件里的所有对象
    def find_data(name, obj):
        global masses, spins
        if isinstance(obj, h5py.Dataset):
            # 只要找到名字匹配的数据集，就抓取
            if name.endswith('final_mass_source') and masses is None:
                print(f"找到质量数据路径: {name}")
                masses = obj[:10000]
            if name.endswith('final_spin') and spins is None:
                print(f"找到自旋数据路径: {name}")
                spins = obj[:10000]

    # 启动深度搜索
    f.visititems(find_data)

if masses is None or spins is None:
    print("❌ 没找到数据！")
    exit(1)

# 3. 转换成极小极小的 CSV 文件
csv_file = "blackhole_data.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Mass_Msun", "Spin"])
    for m, s in zip(masses, spins):
        writer.writerow([f"{m:.4f}", f"{s:.4f}"])

print(f"✅ 转换成功！已生成 {csv_file}！")
