import json
import random
import webbrowser
import os

FILE = os.path.expanduser("~/github-random/data/random_repos.json")

with open(FILE, "r", encoding="utf-8") as f:
    repos = json.load(f)

apk_repos = [r for r in repos if r.get("apk_url")]

if not apk_repos:
    print("数据里没有带 APK 的项目，先敲 抓 抓新数据")
    exit()

repo = random.choice(apk_repos)

print("=" * 50)
print(f"项目名: {repo.get('name', '未知')}")
print(f"语言:   {repo.get('language', '未知')}")
print(f"星星:   {repo.get('stars', 0)}")
print(f"简介:   {repo.get('description', '无描述')}")
apk_name = repo.get("apk_name", "未知")
apk_size = repo.get("apk_size", 0)
if apk_size:
    print(f"APK:    {apk_name} ({apk_size/1024/1024:.1f} MB)")
else:
    print(f"APK:    {apk_name}")
print(f"链接:   {repo.get('url', '')}")
print("=" * 50)
print(f"(从 {len(apk_repos)} 个带 APK 的项目里抽取)")

with open(os.path.expanduser("~/github-random/.last_draw.json"), "w", encoding="utf-8") as f:
    json.dump(repo, f, ensure_ascii=False, indent=2)

try:
    webbrowser.open(repo['url'])
except Exception:
    pass
