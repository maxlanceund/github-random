import requests
import json
import os
import urllib3
urllib3.disable_warnings()

TOKEN_FILE = os.path.expanduser("~/.github_token")
TOKEN = ""
if os.path.exists(TOKEN_FILE):
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        TOKEN = f.read().strip()

ANDROID_QUERIES = [
    "android app", "android client", "android apk",
    "android kotlin app", "android music player",
    "android reader", "android browser", "android note app"
]
OTHER_QUERIES = ["AI", "Python", "Rust", "CLI"]

MAX_APK = 20       # 最多抓 10 个带 APK 的
MAX_OTHER = 5     # 最多抓 20 个普通项目

def headers():
    h = {"Accept": "application/vnd.github+json", "User-Agent": "gh-random"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    return h

import random

def search_repos(query, per_page=20):
    url = "https://api.github.com/search/repositories"
    page = random.randint(1, 5)
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": per_page, "page": page}
    resp = requests.get(url, params=params, headers=headers(), timeout=30, verify=False)
    resp.raise_for_status()
    return resp.json().get("items", [])

def get_latest_apk(repo_full_name):
    url = f"https://api.github.com/repos/{repo_full_name}/releases/latest"
    try:
        resp = requests.get(url, headers=headers(), timeout=20, verify=False)
        if resp.status_code != 200:
            return None
        release = resp.json()
        for asset in release.get("assets", []):
            if asset["name"].lower().endswith(".apk"):
                return {
                    "apk_name": asset["name"],
                    "apk_url": asset["browser_download_url"],
                    "apk_size": asset["size"]
                }
        return None
    except Exception:
        return None

def save(all_repos):
    os.makedirs("data", exist_ok=True)
    path = "data/random_repos.json"
    # 读取旧数据
    old = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            old = json.load(f)
    # 合并去重（新的覆盖旧的）
    merged = {}
    for r in old + all_repos:
        merged[r["name"]] = r
    result = list(merged.values())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return len(result)

def main():
    all_repos = []
    apk_count = 0
    seen = set()

    # 抓安卓项目，凑够 MAX_APK 个就停
    for q in ANDROID_QUERIES:
        if apk_count >= MAX_APK:
            break
        try:
            for item in search_repos(q):
                if apk_count >= MAX_APK:
                    break
                if item["full_name"] in seen:
                    continue
                apk = get_latest_apk(item["full_name"])
                if apk:
                    seen.add(item["full_name"])
                    all_repos.append({
                        "name": item["full_name"],
                        "description": item.get("description") or "无描述",
                        "url": item["html_url"],
                        "stars": item["stargazers_count"],
                        "language": item.get("language") or "未知",
                        "keyword": q,
                        "apk_name": apk["apk_name"],
                        "apk_url": apk["apk_url"],
                        "apk_size": apk["apk_size"]
                    })
                    apk_count += 1
                    print(f"[{apk_count}/{MAX_APK}] ✅ {item['full_name']} -> {apk['apk_name']}")
                    save(all_repos)   # 边抓边存
        except Exception as e:
            print(f"搜索 {q} 失败: {e}")

    # 抓普通项目，凑够 MAX_OTHER 个就停
    other_count = 0
    for q in OTHER_QUERIES:
        if other_count >= MAX_OTHER:
            break
        try:
            for item in search_repos(q):
                if other_count >= MAX_OTHER:
                    break
                if item["full_name"] in seen:
                    continue
                seen.add(item["full_name"])
                all_repos.append({
                    "name": item["full_name"],
                    "description": item.get("description") or "无描述",
                    "url": item["html_url"],
                    "stars": item["stargazers_count"],
                    "language": item.get("language") or "未知",
                    "keyword": q
                })
                other_count += 1
        except Exception as e:
            print(f"搜索 {q} 失败: {e}")

    total = save(all_repos)
    print(f"\n本次新抓 {len(all_repos)} 个，文件里现在共有 {total} 个项目")

if __name__ == "__main__":
    main()
