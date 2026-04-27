import json
import re

# 需要检查的异常字符
DANGER_CHARS = [
    "\ufffd",       # 乱码 �
    "\u200b",       # 零宽空格
    "\u200e",       # 不可见字符
    "\x00", "\x01", "\x02", "\x03", "\x04", "\x05", "\x06", "\x07", "\x08", "\x09", "\x0a", "\x0b", "\x0c", "\x0d"
]

# 图片链接、资源路径
IMAGE_PATTERN = re.compile(r"https?://.*\.(png|jpg|jpeg|gif|bmp|webp)", re.IGNORECASE)
RESOURCE_PATTERN = re.compile(r"tos-cn-i-a9rns2rl98|byteimg|imagex", re.IGNORECASE)

def check_dirty_data(input_file, bad_file="dirty_data_list.json"):
    """
    检查脏数据：乱码、图片链接、异常字符、空文本
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    bad_items = []
    good_count = 0

    for idx, item in enumerate(data):
        text = item.get("text", "").strip()

        # 检查项目
        has_danger = any(c in text for c in DANGER_CHARS)
        has_image = IMAGE_PATTERN.search(text) is not None
        has_resource = RESOURCE_PATTERN.search(text) is not None
        is_empty = len(text) < 2

        # 只要有一项问题，就是脏数据
        is_bad = has_danger or has_image or has_resource or is_empty

        if is_bad:
            reason = []
            if has_danger: reason.append("含乱码/不可见字符")
            if has_image: reason.append("含图片链接")
            if has_resource: reason.append("含资源路径")
            if is_empty: reason.append("文本过短/空")

            bad_items.append({
                "index": idx,
                "text": text,
                "reason": " | ".join(reason)
            })
        else:
            good_count += 1

    # 输出脏数据列表
    with open(bad_file, "w", encoding="utf-8") as f:
        json.dump(bad_items, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"📊 数据检查完成")
    print(f"总数据：{total} 条")
    print(f"✅ 干净数据：{good_count} 条")
    print(f"⚠️ 脏数据：{len(bad_items)} 条")
    print(f"📄 脏数据详情已保存：{bad_file}")
    print("=" * 60)

    # 打印前10条脏数据预览
    if bad_items:
        print("\n🔍 脏数据预览（前10条）：")
        for b in bad_items[:10]:
            print(f"[{b['reason']}] -> {b['text']}")

# ==================== 执行检查 ====================
if __name__ == "__main__":
    check_dirty_data(
        input_file="emotion_final_clean.json",  # 你的最终数据
        bad_file="dirty_data_list.json"           # 脏数据会导出在这里
    )