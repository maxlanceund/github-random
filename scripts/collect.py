import requests, json, os, time, random, urllib3
urllib3.disable_warnings()

TOKEN = ""
tf = os.path.expanduser("~/.github_token")
if os.path.exists(tf):
    TOKEN = open(tf).read().strip()

UA = "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36"

def h():
    x = {"Accept": "application/vnd.github+json", "User-Agent": UA}
    if TOKEN: x["Authorization"] = f"Bearer {TOKEN}"
    return x

def req(url, params=None, timeout=10, retries=3):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=h(), timeout=timeout)
            if r.status_code == 200: return r
            if r.status_code == 403: time.sleep(10)
            else: return r
        except Exception:
            if i < retries-1: time.sleep(2)
    return None

def search(q):
    url = "https://api.github.com/search/repositories"
    p = {"q": q, "sort": "stars", "order": "desc", "per_page": 20, "page": random.randint(1,3)}
    r = req(url, params=p)
    if not r: return []
    try: return r.json().get("items", [])
    except: return []

def get_apk(name):
    r = req(f"https://api.github.com/repos/{name}/releases/latest", timeout=8, retries=2)
    if not r: return None
    try:
        for a in r.json().get("assets", []):
            if a["name"].lower().endswith(".apk"):
                return {"apk_name": a["name"], "apk_url": a["browser_download_url"], "apk_size": a["size"]}
    except: pass
    return None

def save(new):
    os.makedirs("data", exist_ok=True)
    path = "data/random_repos.json"
    old = []
    if os.path.exists(path):
        try: old = json.load(open(path))
        except: old = []
    m = {}
    for r in old + new: m[r["name"]] = r
    result = list(m.values())
    json.dump(result, open(path, "w"), ensure_ascii=False, indent=2)
    return len(result)

def main():
    AQ = ["android app", "android client", "android apk", "android kotlin app",
          "android music player", "android reader", "android browser", "android note app"]
    OQ = ["AI", "Python", "Rust", "CLI"]
    repos, seen, cnt = [], set(), 0
    for q in AQ:
        if cnt >= 20: break
        try:
            for it in search(q):
                if cnt >= 20: break
                if it["full_name"] in seen: continue
                apk = get_apk(it["full_name"])
                if apk:
                    seen.add(it["full_name"])
                    repos.append({"name": it["full_name"], "description": it.get("description") or "无描述",
                                  "url": it["html_url"], "stars": it["stargazers_count"],
                                  "language": it.get("language") or "未知", "keyword": q,
                                  "apk_name": apk["apk_name"], "apk_url": apk["apk_url"], "apk_size": apk["apk_size"]})
                    cnt += 1
                    print(f"[{cnt}/20] OK {it['full_name']} -> {apk['apk_name']}")
            save(repos)
        except Exception as e:
            print(f"搜索 {q} 失败: {e}")
    oc = 0
    for q in OQ:
        if oc >= 5: break
        try:
            for it in search(q):
                if oc >= 5: break
                if it["full_name"] in seen: continue
                seen.add(it["full_name"])
                repos.append({"name": it["full_name"], "description": it.get("description") or "无描述",
                              "url": it["html_url"], "stars": it["stargazers_count"],
                              "language": it.get("language") or "未知", "keyword": q})
                oc += 1
        except Exception as e:
            print(f"搜索 {q} 失败: {e}")
    total = save(repos)
    print(f"\n本次新抓 {len(repos)} 个，文件里现在共有 {total} 个项目")

if __name__ == "__main__":
    main()
