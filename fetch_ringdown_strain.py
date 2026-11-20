import h5py
import numpy as np
import urllib.request
import os
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import argrelextrema

event_name = "GW190521"
det = "H1"
gps_peak = 1242442967.44
pre_win = 0.02
post_win = 0.06

h5_name = f"{event_name}_{det}_strain.h5"
strain_url = "https://gwosc.org/eventapi/json/GW190521/H1/strain/"

if not os.path.exists(h5_name):
    print(f"本地无 {h5_name}，开始下载 strain h5")
    urllib.request.urlretrieve(strain_url, h5_name)
    print("下载完成")
else:
    print(f"✅ 命中缓存，跳过下载 {h5_name}")

t_start = gps_peak - pre_win
t_end = gps_peak + post_win

with h5py.File(h5_name, "r") as f:
    strain = f[det]["strain"][:]
    times = f[det]["times"][:]

df_raw = pd.DataFrame({"time": times, "strain": strain})
df_ring = df_raw[(df_raw["time"] >= t_start) & (df_raw["time"] <= t_end)]
ring_csv = f"{event_name}_{det}_ringdown.csv"
df_ring.to_csv(ring_csv, index=False)
print(f"✅ 铃宕波形保存 {ring_csv}")

dt = df_ring["time"].iloc[1] - df_ring["time"].iloc[0]
N = len(df_ring)
freq = fftfreq(N, d=dt)
amp = np.abs(fft(df_ring["strain"]))
spec_df = pd.DataFrame({"freq": freq, "amp": amp})
spec_df = spec_df[spec_df["freq"] > 0]
spec_df = spec_df[(spec_df["freq"] > 50) & (spec_df["freq"] < 2000)]
spec_df = spec_df.sort_values("freq").reset_index(drop=True)
min_pos = argrelextrema(spec_df["amp"].values, np.less)[0]
valley_df = spec_df.iloc[min_pos]

spec_csv = f"{event_name}_{det}_spec.csv"
valley_csv = f"{event_name}_{det}_valley.csv"
spec_df.to_csv(spec_csv, index=False)
valley_df.to_csv(valley_csv, index=False)
print(f"✅ 频谱 {spec_csv}，V谷结果 {valley_csv}")
print(f"检测到潜在V谷数量：{len(valley_df)}")
