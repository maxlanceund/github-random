import requests
import json
import os
import random
import time
import urllib3
urllib3.disable_warnings()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_FILE = os.path.join(PROJECT_DIR, "data", "digest.md")

SUBJECTS = [
    "physics", "quantum mechanics", "cosmology", "astrophysics",
    "macroeconomics", "behavioral economics", "medicine",
    "epidemiology", "neuroscience", "genetics", "ecology",
    "geology", "climate change", "archaeology", "ancient civilization",
    "anthropology", "world history", "philosophy", "ethics",
    "artificial intelligence", "machine learning", "mathematics",
    "psychology", "sociology", "linguistics", "economics",
]

def rebuild_abstract(inv):
    if not inv:
        return ""
    pairs = []
    for word, positions in inv.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort()
    return " ".join(w for _, w in pairs)

def search(query, per_page=2):
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": per_page,
        "sort": "relevance_score:desc",
        "filter": "is_oa:true",
    }
    headers = {"User-Agent": "digest/1.0 (mailto:maxlanceund@example.com)"}
    r = requests.get(url, params=params, headers=headers, timeout=30, verify=False)
    r.raise_for_status()
    return r.json().get("results", [])

def main():
    selected = random.sample(SUBJECTS, 5)
    lines = []
    lines.append("# 学术速递\n")
    lines.append("随机抽取 5 个学科，每科 2 篇。\n")

    for subj in selected:
        lines.append(f"\n## {subj}\n")
        try:
            papers = search(subj, per_page=2)
            for i, p in enumerate(papers, 1):
                title = p.get("title") or ""
                if not title:
                    continue
                year = p.get("publication_year", "")
                abstract = rebuild_abstract(p.get("abstract_inverted_index"))
                url = p.get("id", "")

                lines.append(f"### {i}. {title}")
                lines.append(f"**年份**: {year}")
                if abstract:
                    lines.append(f"**摘要**: {abstract[:250]}...")
                lines.append(f"**链接**: {url}\n")
        except Exception as e:
            lines.append(f"抓取失败: {e}\n")
        time.sleep(1)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已生成速递: {OUT_FILE}")

if __name__ == "__main__":
    main()
