import requests
import json
import os
import random
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_FILE = os.path.join(PROJECT_DIR, "data", "papers.json")

SUBJECTS = [
    # 经济与商业
    "economics", "macroeconomics", "microeconomics", "economic history",
    "finance", "behavioral economics", "game theory", "development economics",
    # 医学与健康
    "medicine", "epidemiology", "public health", "oncology", "cardiology",
    "immunology", "pharmacology", "psychiatry", "nutrition",
    # 生物
    "biology", "genetics", "neuroscience", "ecology", "evolution",
    "microbiology", "biochemistry", "molecular biology", "botany", "zoology",
    # 地理与地球科学
    "geography", "climate change", "geology", "oceanography",
    "meteorology", "earth science", "environmental science",
    # 考古与文明
    "archaeology", "ancient civilization", "anthropology",
    "cultural history", "prehistoric archaeology", "egyptology",
    "mesoamerican studies", "classical antiquity",
    # 历史
    "world history", "medieval history", "modern history",
    "military history", "history of science",
    # 物理
    "physics", "quantum mechanics", "cosmology", "particle physics",
    "astrophysics", "condensed matter", "relativity", "thermodynamics",
    "optics", "acoustics", "plasma physics", "nuclear physics",
    "biophysics", "computational physics", "mathematical physics",
    "quantum field theory", "string theory", "gravitational waves",
    "dark matter", "dark energy", "atomic physics", "molecular physics",
    "solid state physics", "statistical mechanics", "electromagnetism",
    # 化学与材料
    "chemistry", "organic chemistry", "materials science",
    "physical chemistry", "nanotechnology",
    # 哲学
    "philosophy", "ethics", "epistemology", "metaphysics",
    "philosophy of mind", "political philosophy", "logic",
    # 数学与计算机
    "mathematics", "artificial intelligence", "machine learning",
    "computer science", "cryptography", "algorithms", "statistics",
    "information theory", "data science",
    # 心理与社会
    "psychology", "cognitive science", "sociology", "political science",
    "linguistics", "education", "international relations",
    # 艺术与人文
    "art history", "musicology", "literature", "religious studies",
    "cultural studies", "media studies"
]

def search_openalex(query, per_page=5):
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": per_page,
        "sort": "relevance_score:desc",
        "filter": "is_oa:true"
    }
    headers = {"User-Agent": "paper-random/1.0 (mailto:maxlanceund@example.com)"}
    resp = requests.get(url, params=params, headers=headers, timeout=30, verify=False)
    resp.raise_for_status()
    return resp.json().get("results", [])

def main():
    all_papers = []
    seen = set()

    # 每次随机抽 15 个学科
    selected = random.sample(SUBJECTS, 15)
    print(f"本次抓取学科: {', '.join(selected)}")

    for subject in selected:
        try:
            results = search_openalex(subject, per_page=5)
            count = 0
            for work in results:
                title = work.get("title") or ""
                if not title or work["id"] in seen:
                    continue
                seen.add(work["id"])

                oa = work.get("open_access", {})
                pdf_url = oa.get("oa_url") or ""
                doi = work.get("doi") or ""

                authorships = work.get("authorships", [])
                authors = [a.get("author", {}).get("display_name", "") for a in authorships[:3]]
                author_str = ", ".join([a for a in authors if a])

                all_papers.append({
                    "title": title,
                    "author": author_str,
                    "year": work.get("publication_year"),
                    "subject": subject,
                    "doi": doi,
                    "pdf_url": pdf_url,
                    "url": work.get("id", "").replace("https://openalex.org/", "https://openalex.org/works/")
                })
                count += 1
            print(f"  {subject}: {count} 篇")
        except Exception as e:
            print(f"  搜索 {subject} 失败: {e}")
        time.sleep(1.5)  # 拉长间隔，避免限流

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)

    print(f"\n已保存 {len(all_papers)} 篇论文")

if __name__ == "__main__":
    main()
