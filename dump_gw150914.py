import os
print("==== 当前目录 ====")
print(os.listdir("."))
print("==== GWTC-1_sample_release ====")
if os.path.exists("GWTC-1_sample_release"):
    print(os.listdir("GWTC-1_sample_release"))
else:
    print("GWTC-1_sample_release 不存在")
gw150914 = "GWTC-1_sample_release/GW150914"
print("==== GW150914 目录 ====")
if os.path.exists(gw150914):
    print(os.listdir(gw150914))
else:
    print("GW150914 文件夹不存在")
