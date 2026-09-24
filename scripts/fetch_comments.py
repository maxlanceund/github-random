import subprocess
import json
import os
import sys
import glob

# 让输出目录基于脚本所在位置，无论在本地还是云端都对
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_DIR, "data", "comments")

def main():
    if len(sys.argv) < 2:
        print("用法: python fetch_comments.py <YouTube视频链接>")
        sys.exit(1)

    url = sys.argv[1]
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"输出目录: {OUT_DIR}")
    print(f"正在抓取评论: {url}")
    print("这可能需要几分钟，请耐心等待...")

    cmd = [
        "yt-dlp",
        "--write-comments",
        "--skip-download",
        "--extractor-args", "youtube:comment_sort=top;max_comments=100",
        "--no-warnings",
        "-o", os.path.join(OUT_DIR, "%(id)s.%(ext)s"),
        url
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

    if result.returncode != 0:
        print("抓取失败:")
        print(result.stderr[-3000:])
        sys.exit(1)

    print(result.stdout[-2000:])

    json_files = sorted(glob.glob(os.path.join(OUT_DIR, "*.info.json")),
                        key=os.path.getmtime, reverse=True)
    if not json_files:
        print("没有找到输出文件")
        sys.exit(1)

    latest = json_files[0]
    print(f"输出文件: {latest}")

    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)

    comments = data.get("comments", [])
    title = data.get("title", "未知标题")
    print(f"共抓取 {len(comments)} 条评论")

    txt_file = latest.replace(".info.json", ".txt")
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"视频: {title}\n")
        f.write(f"链接: {url}\n")
        f.write(f"共 {len(comments)} 条评论\n")
        f.write("=" * 60 + "\n\n")
        for i, c in enumerate(comments[:100], 1):
            author = c.get("author", "匿名")
            text = c.get("text", "")
            likes = c.get("like_count", 0)
            f.write(f"[{i}] {author}  👍{likes}\n")
            f.write(f"{text}\n")
            f.write("-" * 60 + "\n")

    print(f"文本文件已保存: {txt_file}")

if __name__ == "__main__":
    main()
