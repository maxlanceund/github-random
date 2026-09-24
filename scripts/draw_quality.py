import json
import random
import webbrowser
import os

FILE = os.path.expanduser("~/github-random/data/random_repos.json")

with open(FILE, "r", encoding="utf-8") as f:
    repos = json.load(f)

apk_repos = [r for r in repos if "apk_url" in r]

BAD_WORDS = ["xposed", "shizuku", "需要配合", "需配合", "需要root", "需root", "补丁", "patch", "module"]
def is_quality(r):
    if r["stars"] < 2000:
        return False
    if r["apk_size"] < 1 * 1024 * 1024:
        return False
    desc = (r.get("description") or "").lower()
    for w in BAD_WORDS:
        if w in desc:
            return False
    return True

quality = [r for r in apk_repos if is_quality(r)]

if not quality:
    print("没有符合条件的精品 APK")
    exit()

repo = random.choice(quality)
print("=" * 50)
print(f"项目名: {repo['name']}")
print(f"语言:   {repo['language']}")
print(f"星星:   {repo['stars']}")
size_mb = repo["apk_size"] / 1024 / 1024
print(f"APK:    {repo['apk_name']} ({size_mb:.1f} MB)")
print(f"简介:   {repo['description']}")
print(f"链接:   {repo['url']}")
print("=" * 50)
print(f"(从 {len(quality)} 个精品里抽取，总池子 {len(apk_repos)} 个)")

with open(os.path.expanduser("~/github-random/.last_draw.json"), "w", encoding="utf-8") as f:
    json.dump(repo, f, ensure_ascii=False, indent=2)

try:
    webbrowser.open(repo['url'])
    print("已在浏览器打开该项目")
except Exception as e:
    print(f"打开浏览器失败: {e}")
