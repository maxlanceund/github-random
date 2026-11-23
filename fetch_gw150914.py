import numpy as np
import pandas as pd
import gwosc
from gwosc.datasets import event_gps
import h5py
import requests

event = "GW150914"
gps = event_gps(event)
detector = "H1"
t_start = gps - 0.2
t_end = gps + 0.2

url = f"https://www.gw-openscience.org/eventapi/json/GWTC-1-confirmed/{event}"
meta = requests.get(url).json()
strain_url = None
for f in meta["events"][event]["files"]:
    if f["detector"] == detector and f["description"] == "strain":
        strain_url = f["url"]
        break

r = requests.get(strain_url)
with open("gw150914_strain.hdf5","wb") as f:
    f.write(r.content)

with h5py.File("gw150914_strain.hdf5","r") as f:
    group = f[detector]["strain"]
    h = group["strain"][:]
    t0 = group.attrs["GPSstart"]
    dt = group.attrs["dt"]
    t = t0 + np.arange(len(h)) * dt

mask = (t >= t_start) & (t <= t_end)
t = t[mask]
h = h[mask]

from scipy.signal import welch
fs = 1.0/dt
freq_psd, psd = welch(h, fs=fs, nperseg=len(h)//2)
h_fft = np.fft.rfft(h)
freq = np.fft.rfftfreq(len(h), d=dt)
psd_interp = np.interp(freq, freq_psd, psd)
h_white_fft = h_fft / np.sqrt(psd_interp)
h_white = np.fft.irfft(h_white_fft, n=len(h))
amp_white = np.abs(h_white_fft)
spec_df = pd.DataFrame({"freq":freq,"amp":amp_white})

win = 15
valley_list = []
for i in range(win, len(freq)-win):
    local = amp_white[i-win:i+win+1]
    mu = np.mean(local)
    sigma = np.std(local)
    center_amp = amp_white[i]
    is_min = center_amp == np.min(local)
    if is_min and center_amp <= mu - sigma:
        valley_list.append({"freq":freq[i],"amp":center_amp})
valley_df = pd.DataFrame(valley_list)

ring_white_df = pd.DataFrame({"time":t,"strain":h_white})
ring_white_df.to_csv("GW150914_H1_ringdown_white.csv",index=False)
spec_df.to_csv("GW150914_H1_spec_white.csv",index=False)
valley_df.to_csv("GW150914_H1_valley_white.csv",index=False)

print("GW150914处理完成")
print(f"识别V谷数量 {len(valley_df)}")
