import json
import os

PROJ_FILE = os.path.expanduser("~/github-random/data/favorites.json")
BOOK_FILE = os.path.expanduser("~/github-random/data/favorite_books.json")

print("=" * 55)

# 项目收藏
if os.path.exists(PROJ_FILE):
    with open(PROJ_FILE, "r", encoding="utf-8") as f:
        projs = json.load(f)
else:
    projs = []

print(f"📦 项目收藏 ({len(projs)} 个)")
print("-" * 55)
if projs:
    for i, r in enumerate(projs, 1):
        if "apk_url" in r:
            size_mb = r["apk_size"] / 1024 / 1024
            print(f"{i}. {r['name']}  ⭐{r['stars']}  [📱 APK {size_mb:.1f}MB]")
        else:
            print(f"{i}. {r['name']}  ⭐{r['stars']}")
        print(f"   {r['description'][:80]}")
        print(f"   {r['url']}")
        print()
else:
    print("(空)")
    print()

# 书籍收藏
if os.path.exists(BOOK_FILE):
    with open(BOOK_FILE, "r", encoding="utf-8") as f:
        books = json.load(f)
else:
    books = []

print(f"📚 书籍收藏 ({len(books)} 本)")
print("-" * 55)
if books:
    for i, b in enumerate(books, 1):
        print(f"{i}. {b['title']}  [{b.get('language', '?')}]")
        print(f"   {b.get('category', '')}")
        print(f"   {b['url']}")
        print()
else:
    print("(空)")

print("=" * 55)
