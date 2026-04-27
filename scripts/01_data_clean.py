import json
import random
import os
import re
import unicodedata

def split_json_lines_file(input_file, split_num, output_dir):
    """
    将每行一条JSON的数据文件均匀分割成 N 份，并保存到指定输出文件夹
    :param input_file: 输入文件路径（如 efad.utf8.txt）
    :param split_num: 分割份数（如 20）
    :param output_dir: 输出文件夹路径（如 annotated_parts）
    """
    # 自动创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 1. 读取所有数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 随机打乱（保证标注均匀）
    random.seed(42)
    random.shuffle(data)

    # 3. 计算每份大小
    total = len(data)
    per_part = total // split_num
    remainder = total % split_num

    print(f"📊 总数据量：{total}")
    print(f"📦 每份平均：{per_part} 条")
    print(f"📂 输出目录：{output_dir}\n")

    # 4. 分割并保存
    index = 0
    for i in range(split_num):
        end = index + per_part + (1 if i < remainder else 0)
        part = data[index:end]
        index = end

        # 文件名 part_01.json
        filename = f"part_{i+1:02d}.json"
        save_path = os.path.join(output_dir, filename)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False, indent=2)

        print(f"✅ {filename}  ——  {len(part)} 条")

    print(f"\n🎉 全部完成！共 {split_num} 份 → {output_dir}")

# 只提取每条数据的 title，生成干净的情绪文本
def extract_only_title(input_file, output_file):
    result = []
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            title = data.get("title", "").strip()
            
            if title:  # 只保留有内容的
                result.append({
                    "text": title
                })

    # 保存干净数据
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 提取完成！共 {len(result)} 条 title 数据")
    print(f"📄 文件：{output_file}")

def clean_gender_prefix(input_file, output_file):
    """
    清洗文本开头的 男/女 前缀（含空格/逗号/重复情况）
    例如：
    女 我很烦 → 我很烦
    男 男 纠结 → 纠结
    女,焦虑 → 焦虑
    女，抑郁 → 抑郁
    """
    # 读取数据
    # 1. 清理前缀：男/女 + ，/,/空格
    pattern_prefix = r'^([男女][，,\s]+)+'

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_data = []
    for item in data:
        text = item.get("text", "").strip()

        # ① 去掉性别前缀
        text = re.sub(pattern_prefix, "", text).strip()
        
        # ② 替换所有 \t 制表符为 空格
        text = text.replace("\t", " ")
        
        # ③ 多个连续空格 → 1个空格
        text = re.sub(r'\s+', ' ', text).strip()

        # 4. 【关键】清理重复标点符号
        text = re.sub(r'([，。！？…])\1+', r'\1', text)

        # 5. 【关键】全角标点统一转半角
        text = unicodedata.normalize('NFKC', text)

        # 6. 【关键】过滤乱码（替换字符�）
        if '\ufffd' in text:
            continue

        if len(text) < 5:
            continue
        if text:
            cleaned_data.append({"text": text})

    # 保存
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 终极清洗完成！")
    print(f"原始：{len(data)} 条")
    print(f"最终：{len(cleaned_data)} 条")
if __name__ == "__main__":
    # extract_only_title(
    #     input_file="efad.utf8.txt",       # 你的原始文件
    #     output_file="emotion_title_only.json"  # 最终干净数据
    # )
    # clean_gender_prefix(
    #     input_file="emotion_title_only.json",
    #     output_file="emotion_final_clean.json"  # 最终完美数据
    # )
    # 把 efad.utf8.txt 分割成20份，保存到 annotated_parts 文件夹
    split_json_lines_file(
        input_file="emotion_final_clean.json",
        split_num=20,
        output_dir="annotated_parts"
    )