import requests
import json
import os
import random

# 古腾堡计划：7万+本公共领域书籍，全部免费合法
API = "https://gutendex.com/books"

def fetch_books(page=1, language=None):
    params = {"page": page}
    if language:
        params["languages"] = language
    headers = {"User-Agent": "book-random"}
    resp = requests.get(API, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

def main():
    # 随机抓几页，混合多语言
    all_books = []
    pages = random.sample(range(1, 50), 5)  # 随机抽5页
    for p in pages:
        try:
            data = fetch_books(page=p)
            for book in data.get("results", []):
                # 优先选有中文或英文的
                langs = book.get("languages", [])
                if not any(l in ["en", "zh", "fr", "de", "es"] for l in langs):
                    continue
                # 找可读的格式链接
                formats = book.get("formats", {})
                read_url = (formats.get("text/html") or
                            formats.get("application/epub+zip") or
                            formats.get("text/plain; charset=utf-8") or
                            formats.get("text/plain") or "")
                if not read_url:
                    continue
                authors = book.get("authors", [])
                author_name = authors[0]["name"] if authors else "未知作者"
                all_books.append({
                    "title": book.get("title", "无标题"),
                    "author": author_name,
                    "languages": langs,
                    "download_count": book.get("download_count", 0),
                    "read_url": read_url,
                    "gutenberg_url": f"https://www.gutenberg.org/ebooks/{book['id']}"
                })
        except Exception as e:
            print(f"第{p}页抓取失败: {e}")
    # 去重
    seen = set()
    unique = []
    for b in all_books:
        key = b["gutenberg_url"]
        if key not in seen:
            seen.add(key)
            unique.append(b)
    os.makedirs("data", exist_ok=True)
    with open("data/books.json", "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(unique)} 本书")

if __name__ == "__main__":
    main()
