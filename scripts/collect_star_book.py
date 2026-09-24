import json
import os

LAST_FILE = os.path.expanduser("~/github-random/.last_book.json")
FAV_FILE = os.path.expanduser("~/github-random/data/favorite_books.json")

if not os.path.exists(LAST_FILE):
    print("还没有抽过书，先敲 抽书")
    exit()

with open(LAST_FILE, "r", encoding="utf-8") as f:
    book = json.load(f)

if os.path.exists(FAV_FILE):
    with open(FAV_FILE, "r", encoding="utf-8") as f:
        favs = json.load(f)
else:
    favs = []

if any(r["url"] == book["url"] for r in favs):
    print(f"已经收藏过了: {book['title']}")
    exit()

favs.append(book)
with open(FAV_FILE, "w", encoding="utf-8") as f:
    json.dump(favs, f, ensure_ascii=False, indent=2)

print(f"已收藏书: {book['title']}")
print(f"当前书籍收藏总数: {len(favs)}")
