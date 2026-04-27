import json
import os
import random

# 你的路径
DATA_DIR = "/root/autodl-tmp/dataset/labeled_final"
TEST_FILE = "/root/autodl-tmp/dataset/test_set.json"

# 抽取 20% 测试集
test_set = []
for i in range(17, 21):
    fname = f"part_{i:02d}.json"
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            test_set.extend(json.load(f))


# 保存
with open(TEST_FILE, "w", encoding="utf-8") as f:
    json.dump(test_set, f, ensure_ascii=False, indent=2)

print(f"✅ 测试集生成完成！共 {len(test_set)} 条（20%）")
print(f"✅ 保存路径：{TEST_FILE}")