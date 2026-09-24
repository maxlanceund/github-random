import requests
import random
import json
import os

KEYWORDS = ["AI", "Android", "Python", "Philosophy", "Economics",
            "machine-learning", "CLI", "web", "Rust", "Go",
            "TypeScript", "linux", "docker", "api", "database"]

def search_repos(keyword, per_page=20):
    url = "https://api.github.com/search/repositories"
    params = {"q": keyword, "sort": "stars", "order": "desc", "per_page": per_page}
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "gh-random"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(url, params=params, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json().get("items", [])

def main():
    all_repos = []
    selected = random.sample(KEYWORDS, 5)
    for kw in selected:
        try:
            for item in search_repos(kw):
                all_repos.append({
                    "name": item["full_name"],
                    "description": item.get("description") or "无描述",
                    "url": item["html_url"],
                    "stars": item["stargazers_count"],
                    "language": item.get("language") or "未知",
                    "keyword": kw
                })
        except Exception as e:
            print(f"搜索 {kw} 失败: {e}")
    seen = set()
    unique = []
    for r in all_repos:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique.append(r)
    os.makedirs("data", exist_ok=True)
    with open("data/random_repos.json", "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(unique)} 个项目")

if __name__ == "__main__":
    main()
