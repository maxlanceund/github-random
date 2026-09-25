import os

bbm_dir = "bbm/models"

if not os.path.exists(bbm_dir):
    print(f"目录不存在: {bbm_dir}")
    exit()

# 统计 .bnet 文件数
bnet_files = []
for root, dirs, files in os.walk(bbm_dir):
    for f in files:
        if f.endswith('.bnet'):
            bnet_files.append(os.path.join(root, f))

print(f"=== .bnet 文件总数: {len(bnet_files)} ===")
print()

# 列出前 10 个文件
print("=== 前 10 个文件 ===")
for f in bnet_files[:10]:
    print(f)
print()

# 输出第一个文件的完整内容
if bnet_files:
    print("=== 第一个文件的内容 ===")
    print(f"文件: {bnet_files[0]}")
    print("-" * 60)
    with open(bnet_files[0], 'r') as fh:
        content = fh.read()
    print(content[:2000])
    print("-" * 60)
    print(f"文件大小: {len(content)} 字符")

    # 统计行数分布
    print()
    print("=== 模型规模分布（按 .bnet 文件行数）===")
    sizes = []
    for f in bnet_files:
        with open(f, 'r') as fh:
            lines = [l for l in fh if l.strip() and not l.startswith('#')]
            sizes.append(len(lines))
    sizes.sort()
    if sizes:
        print(f"最小值: {sizes[0]}")
        print(f"最大值: {sizes[-1]}")
        print(f"中位数: {sizes[len(sizes)//2]}")
        print(f"平均值: {sum(sizes)/len(sizes):.1f}")
