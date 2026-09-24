import json
import os

path = os.path.expanduser("~/github-random/data/random_repos.json")
with open(path, "r", encoding="utf-8") as f:
    repos = json.load(f)

apk_repos = [r for r in repos if "apk_url" in r]
apk_repos.sort(key=lambda x: x["stars"], reverse=True)

print("=" * 65)
print(f"📱 共有 {len(apk_repos)} 个带 APK 的项目")
print("=" * 65)

for i, r in enumerate(apk_repos, 1):
    size_mb = r["apk_size"] / 1024 / 1024
    print(f"{i:2d}. {r['name']}  ⭐{r['stars']}  ({size_mb:.1f}MB)")
    print(f"    {r['description'][:70]}")
    print()
