import json
import os
import re
import requests

LAST_FILE = os.path.expanduser("~/github-random/.last_book.json")
BOOK_DIR = "/sdcard/Download/books"

if not os.path.exists(LAST_FILE):
    print("还没有抽过书，先敲 抽书")
    exit()

with open(LAST_FILE, "r", encoding="utf-8") as f:
    book = json.load(f)

title = book["title"]
lang = book.get("language", "zh")
url = book["url"]

if lang == "en":
    api = "https://en.wikisource.org/w/api.php"
else:
    api = "https://zh.wikisource.org/w/api.php"

# 用 parse API 拿 HTML
params = {
    "action": "parse",
    "page": title,
    "prop": "text",
    "format": "json",
    "disablelimitreport": "1",
    "disableeditsection": "1",
}
headers = {"User-Agent": "book-downloader/1.0"}

print(f"正在下载: {title}")
try:
    resp = requests.get(api, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    html = data.get("parse", {}).get("text", {}).get("*", "")

    if not html:
        print("没有拿到页面内容")
        exit()

    # 清洗 HTML：去掉标签、去掉引用角标、压缩空行
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    # 去掉编辑链接和角标
    text = re.sub(r"<sup[^>]*>.*?</sup>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    # 清理 HTML 实体
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    # 压缩空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    safe_title = title.replace("/", "_").replace("\\", "_").replace(":", "_")
    os.makedirs(BOOK_DIR, exist_ok=True)
    out_path = os.path.join(BOOK_DIR, f"{safe_title}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"书名: {title}\n")
        f.write(f"来源: {url}\n")
        f.write("=" * 40 + "\n\n")
        f.write(text)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"下载完成: {out_path}")
    print(f"文件大小: {size_kb:.1f} KB")
except Exception as e:
    print(f"下载失败: {e}")
