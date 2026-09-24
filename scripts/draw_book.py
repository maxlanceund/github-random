import json
import random
import webbrowser
import os
import sys

BOOK_FILE = os.path.expanduser("~/github-random/data/books.json")

if not os.path.exists(BOOK_FILE):
    print("还没有书籍数据，先敲 抓书")
    exit()

with open(BOOK_FILE, "r", encoding="utf-8") as f:
    books = json.load(f)

if not books:
    print("书库是空的，先敲 抓书")
    exit()

# 参数1：语言筛选（zh / en）
# 参数2：分类关键词（可选）
lang = None
cat = None
for arg in sys.argv[1:]:
    if arg in ("zh", "en"):
        lang = arg
    else:
        cat = arg

if lang:
    books = [b for b in books if b.get("language") == lang]
if cat:
    books = [b for b in books if cat in b.get("category", "")]

if not books:
    print("没有匹配的书")
    exit()

book = random.choice(books)

print("=" * 50)
print(f"书名:   {book['title']}")
print(f"分类:   {book['category']}")
print(f"语言:   {book.get('language', '未知')}")
print(f"链接:   {book['url']}")
print("=" * 50)

with open(os.path.expanduser("~/github-random/.last_book.json"), "w", encoding="utf-8") as f:
    json.dump(book, f, ensure_ascii=False, indent=2)

try:
    webbrowser.open(book['url'])
    print("已在浏览器打开书籍页面")
except Exception as e:
    print(f"打开浏览器失败: {e}")
