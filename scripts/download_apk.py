import json
import os
import requests

LAST_FILE = os.path.expanduser("~/github-random/.last_draw.json")
APK_DIR = "/sdcard/Download/apks"

if not os.path.exists(LAST_FILE):
    print("还没有抽过项目，先敲 抽APK")
    exit()

with open(LAST_FILE, "r", encoding="utf-8") as f:
    repo = json.load(f)

if "apk_url" not in repo:
    print(f"{repo['name']} 没有现成的 APK")
    exit()

filename = repo["apk_name"]
url = repo["apk_url"]
out_path = os.path.join(APK_DIR, filename)

print(f"正在下载: {filename}")
print(f"来源: {repo['name']}")
os.makedirs(APK_DIR, exist_ok=True)

headers = {"User-Agent": "apk-downloader/1.0"}
try:
    with requests.get(url, headers=headers, stream=True, timeout=300, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r进度: {pct}% ({downloaded//1024//1024}MB / {total//1024//1024}MB)", end="")
    print()
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"下载完成: {out_path}")
    print(f"文件大小: {size_mb:.1f} MB")
    print("去文件管理器 → Download → apks 里点安装")
except Exception as e:
    print(f"\n下载失败: {e}")
