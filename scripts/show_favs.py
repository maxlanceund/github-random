import json
import os

FAV_FILE = os.path.expanduser("~/github-random/data/favorites.json")

if not os.path.exists(FAV_FILE):
    print("收藏夹还是空的，先敲 抽 抽到好东西再敲 收藏")
    exit()

with open(FAV_FILE, "r", encoding="utf-8") as f:
    favs = json.load(f)

if not favs:
    print("收藏夹还是空的")
    exit()

print(f"共 {len(favs)} 个收藏项目")
print("=" * 50)

for i, r in enumerate(favs, 1):
    print(f"{i}. {r['name']}  ⭐{r['stars']}")
    print(f"   {r['description']}")
    print(f"   {r['url']}")
    print("-" * 50)
