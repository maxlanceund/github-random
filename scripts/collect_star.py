import json
import os

LAST_FILE = os.path.expanduser("~/github-random/.last_draw.json")
FAV_FILE = os.path.expanduser("~/github-random/data/favorites.json")

if not os.path.exists(LAST_FILE):
    print("还没有抽过项目，先敲 抽")
    exit()

with open(LAST_FILE, "r", encoding="utf-8") as f:
    repo = json.load(f)

if os.path.exists(FAV_FILE):
    with open(FAV_FILE, "r", encoding="utf-8") as f:
        favs = json.load(f)
else:
    favs = []

if any(r["name"] == repo["name"] for r in favs):
    print(f"已经收藏过了: {repo['name']}")
    exit()

favs.append(repo)
with open(FAV_FILE, "w", encoding="utf-8") as f:
    json.dump(favs, f, ensure_ascii=False, indent=2)

print(f"已收藏: {repo['name']}")
print(f"当前收藏总数: {len(favs)}")
