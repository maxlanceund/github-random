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

# 支持按语言筛选，比如：抽书 zh
lang = sys.argv[1] if len(sys.argv) > 1 else None
if lang:
    books = [b for b in books if lang in b["languages"]]

if not books:
    print(f"没有 {lang} 语言的书")
    exit()

book = random.choice(books)

print("=" * 50)
print(f"书名:   {book['title']}")
print(f"作者:   {book['author']}")
print(f"语言:   {', '.join(book['languages'])}")
print(f"下载量: {book['download_count']}")
print(f"阅读:   {book['read_url']}")
print(f"详情:   {book['gutenberg_url']}")
print("=" * 50)

# 保存最近抽到的书，供收藏用
with open(os.path.expanduser("~/github-random/.last_book.json"), "w", encoding="utf-8") as f:
    json.dump(book, f, ensure_ascii=False, indent=2)

try:
    webbrowser.open(book['gutenberg_url'])
    print("已在浏览器打开书籍页面")
except Exception as e:
    print(f"打开浏览器失败: {e}")
