import json
import random
import webbrowser
import os
import sys

FILE = os.path.expanduser("~/github-random/data/random_repos.json")

with open(FILE, "r", encoding="utf-8") as f:
    repos = json.load(f)

mode = sys.argv[1] if len(sys.argv) > 1 else "all"

if mode == "apk":
    repos = [r for r in repos if "apk_url" in r]

if not repos:
    print("没有匹配的项目")
    exit()

# 按星标排序，取前 30% 作为"热门"
repos_sorted = sorted(repos, key=lambda x: x["stars"], reverse=True)
top_n = max(3, len(repos_sorted) * 30 // 100)
hot = repos_sorted[:top_n]

repo = random.choice(hot)

print("=" * 50)
print(f"项目名: {repo['name']}")
print(f"语言:   {repo['language']}")
print(f"星星:   {repo['stars']}  (热门 Top {top_n})")
print(f"简介:   {repo['description']}")
if "apk_name" in repo:
    size_mb = repo["apk_size"] / 1024 / 1024
    print(f"APK:    {repo['apk_name']} ({size_mb:.1f} MB)")
print(f"链接:   {repo['url']}")
print("=" * 50)

with open(os.path.expanduser("~/github-random/.last_draw.json"), "w", encoding="utf-8") as f:
    json.dump(repo, f, ensure_ascii=False, indent=2)

try:
    webbrowser.open(repo['url'])
    print("已在浏览器打开该项目")
except Exception as e:
    print(f"打开浏览器失败: {e}")
