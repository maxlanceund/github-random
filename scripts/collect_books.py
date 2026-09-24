import requests
import json
import os

def fetch_category(api, category, limit=50):
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": limit,
        "cmnamespace": "0",   # 关键：只要主命名空间（真正的作品页）
        "format": "json"
    }
    headers = {"User-Agent": "book-random/1.0"}
    resp = requests.get(api, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

ZH_API = "https://zh.wikisource.org/w/api.php"
ZH_CATS = ["四庫全書", "中國古典小說", "唐詩", "宋詞",
           "論語", "道德經", "莊子", "史記", "資治通鑑"]

EN_API = "https://en.wikisource.org/w/api.php"
EN_CATS = ["Novels", "Poems", "Essays", "Plays",
           "Philosophy", "Science", "History", "Short stories"]

def collect(api, cats, lang, base_url):
    books = []
    for cat in cats:
        try:
            data = fetch_category(api, cat, limit=20)
            for item in data.get("query", {}).get("categorymembers", []):
                title = item.get("title", "")
                if not title:
                    continue
                page_url = f"{base_url}/wiki/{title.replace(' ', '_')}"
                books.append({
                    "title": title,
                    "category": cat,
                    "language": lang,
                    "url": page_url
                })
        except Exception as e:
            print(f"抓取 {lang}/{cat} 失败: {e}")
    return books

def main():
    all_books = []
    all_books += collect(ZH_API, ZH_CATS, "zh", "https://zh.wikisource.org")
    all_books += collect(EN_API, EN_CATS, "en", "https://en.wikisource.org")

    seen = set()
    unique = []
    for b in all_books:
        if b["url"] not in seen:
            seen.add(b["url"])
            unique.append(b)

    os.makedirs("data", exist_ok=True)
    with open("data/books.json", "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    zh_count = len([b for b in unique if b["language"] == "zh"])
    en_count = len([b for b in unique if b["language"] == "en"])
    print(f"已保存 {len(unique)} 本书 (中文 {zh_count} 本，英文 {en_count} 本)")

if __name__ == "__main__":
    main()
