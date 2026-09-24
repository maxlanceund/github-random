import json
import random
import webbrowser
import os
import sys

FILE = os.path.expanduser("~/github-random/data/papers.json")

if not os.path.exists(FILE):
    print("还没有论文数据，先敲 抓论文")
    exit()

with open(FILE, "r", encoding="utf-8") as f:
    papers = json.load(f)

if not papers:
    print("论文库是空的，先敲 抓论文")
    exit()

# 支持按学科筛选：抽论文 physics
subject = sys.argv[1] if len(sys.argv) > 1 else None
if subject:
    papers = [p for p in papers if subject.lower() in p["subject"].lower()]

if not papers:
    print(f"没有找到 {subject} 相关的论文")
    exit()

paper = random.choice(papers)

print("=" * 60)
print(f"标题:   {paper['title']}")
print(f"作者:   {paper['author']}")
print(f"年份:   {paper['year']}")
print(f"学科:   {paper['subject']}")
if paper.get("pdf_url"):
    print(f"PDF:    {paper['pdf_url']}")
if paper.get("doi"):
    print(f"DOI:    {paper['doi']}")
print(f"链接:   {paper['url']}")
print("=" * 60)

with open(os.path.expanduser("~/github-random/.last_paper.json"), "w", encoding="utf-8") as f:
    json.dump(paper, f, ensure_ascii=False, indent=2)

try:
    webbrowser.open(paper['url'])
    print("已在浏览器打开论文页面")
except Exception as e:
    print(f"打开浏览器失败: {e}")
