import h5py
import numpy as np
import urllib.request
import os

file_name = "GW150914_posterior.h5"
if not os.path.exists(file_name):
    print("正在云端下载 129M 真实黑洞数据...")
    url = "https://zenodo.org/api/files/ecf41927-9275-47da-8b37-e299693fe5cb/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_cosmo.h5"
    urllib.request.urlretrieve(url, file_name)
    print("下载完成！")

print("\n====== 开始侦察文件结构 ======")
with h5py.File(file_name, 'r') as f:
    # 定义侦察函数
    def print_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"发现数据集: {name} (形状: {obj.shape})")

    # 遍历所有对象
    f.visititems(print_datasets)
print("====== 侦察结束 ======")
