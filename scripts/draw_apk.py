import json
import random
import webbrowser
import os

FILE = os.path.expanduser("~/github-random/data/random_repos.json")

with open(FILE, "r", encoding="utf-8") as f:
    repos = json.load(f)

# 只留安卓相关的项目
android_repos = [r for r in repos if "android" in r["keyword"].lower()
                 or "apk" in r["keyword"].lower()
                 or "android" in r["name"].lower()
                 or "android" in r["description"].lower()]

if not android_repos:
    print("数据里没有安卓项目，先敲 抓 抓新数据")
    exit()

repo = random.choice(android_repos)
print("=" * 50)
print(f"项目名: {repo['name']}")
print(f"语言:   {repo['language']}")
print(f"星星:   {repo['stars']}")
print(f"简介:   {repo['description']}")
print(f"链接:   {repo['url']}")
print("=" * 50)

with open(os.path.expanduser("~/github-random/.last_draw.json"), "w", encoding="utf-8") as f:
    json.dump(repo, f, ensure_ascii=False, indent=2)

try:
    webbrowser.open(repo['url'])
    print("已在浏览器打开该项目")
except Exception as e:
    print(f"打开浏览器失败: {e}")
