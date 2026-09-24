import json
import random
import webbrowser
import sys

with open("data/random_repos.json", "r", encoding="utf-8") as f:
    repos = json.load(f)

keyword = sys.argv[1] if len(sys.argv) > 1 else None
if keyword:
    repos = [r for r in repos if keyword.lower() in r["keyword"].lower()
             or keyword.lower() in r["name"].lower()
             or keyword.lower() in r["description"].lower()]

if not repos:
    print("没有匹配的项目，先跑 collect.py 抓新数据")
    exit()

repo = random.choice(repos)
print("=" * 40)
print(f"项目名: {repo['name']}")
print(f"语言:   {repo['language']}")
print(f"星星:   {repo['stars']}")
print(f"简介:   {repo['description']}")
print(f"链接:   {repo['url']}")
print("=" * 40)

try:
    webbrowser.open(repo['url'])
    print("已在浏览器打开该项目")
except Exception as e:
    print(f"打开浏览器失败: {e}")

# 保存最近一次抽到的项目，供收藏使用
with open(".last_draw.json", "w", encoding="utf-8") as f:
    json.dump(repo, f, ensure_ascii=False, indent=2)
