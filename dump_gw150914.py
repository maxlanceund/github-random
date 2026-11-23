import pandas as pd
from gwosc.datasets import event_posterior_samples

# GWTC-1 对应的事件版本 GW150914-v3
samples = event_posterior_samples("GW150914-v3")

# 打印所有字段，用来排错，防止键名不对直接崩溃
print("===== 全部可用字段列表 =====")
print(list(samples.keys()))

# 注意：gwosc 的后验有两套质量：探测器系 mass_1 / 源系 mass_1_source
# 你之前GW170817用的是源框架质量，这里保持对齐
df = pd.DataFrame({
    "m1": samples["mass_1_source"],
    "m2": samples["mass_2_source"],
    "chi1z": samples["spin_1z"],
    "chi2z": samples["spin_2z"]
})

# 和GW170817一样，固定随机种子采样5000条
df = df.sample(n=5000, random_state=42, ignore_index=True)
df.to_csv("posterior_samples_GW150914.csv", index=False)
print("✅ GW150914 CSV导出完成")
