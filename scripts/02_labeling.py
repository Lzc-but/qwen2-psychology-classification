import json
import os
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# ===================== 【你的路径】核心配置 =====================
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"  # 你的模型路径
CONFIDENCE_THRESHOLD = 0.7  # 置信度过滤
INPUT_DIR = "data/processed/annotated_parts"    # 20份数据目录
OUTPUT_DIR = "data/labeled/all"   # 输出目录
# ================================================================

# 4分类 + 严格JSON输出Prompt
PROMPT = """你是专业心理文本分类专家，严格按规则分类，只输出标准JSON，无其他内容。

分类规则：
0：正常（日常闲聊、中性提问、无负面情绪、无心理困扰）
1：焦虑（不安、多疑、纠结、担心、内耗、胡思乱想）
2：低落（难过、压抑、自卑、无助、心情差、自我否定）
3：高风险（自杀、自残、想死、厌世、极端危险想法）

文本：{}
按以下JSON格式输出，仅返回这两个字段：
{{"label":"0/1/2/3", "confidence":0.0~1.0}}
"""

# ===================== 加载本地模型 =====================
print("🔹 正在加载 Qwen2.5-7B-Instruct 模型...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

gen_config = GenerationConfig(
    max_new_tokens=64,
    do_sample=False
)
print("✅ 模型加载完成！")

# ===================== 模型推理 =====================
def infer(text):
    prompt = PROMPT.format(text)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()

# ===================== 解析JSON =====================
def extract_json(res):
    try:
        match = re.search(r"\{.*\}", res, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except:
        return None

# ===================== 标注单个文本 =====================
def get_label(text):
    res = infer(text)
    data = extract_json(res)
    if not data:
        return "0", 0.0

    label = str(data.get("label", "0"))
    conf = float(data.get("confidence", 0.0))

    if label in {"0", "1", "2", "3"} and 0 <= conf <= 1:
        return label, conf
    return "0", 0.0

# ===================== 处理单个文件 =====================
def process_file(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid = []
    for item in tqdm(data, desc=os.path.basename(in_path)):
        text = item["text"].strip()
        label, conf = get_label(text)
        if conf >= CONFIDENCE_THRESHOLD:
            valid.append({
                "text": text,
                "label": label,
                "confidence": round(conf, 3)
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, ensure_ascii=False, indent=2)
    print(f"✅ 完成：{len(data)} → 保留 {len(valid)} 条")

# ===================== 批量处理20份 =====================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i in range(2, 21):
        fname = f"part_{i:02d}.json"
        in_p = os.path.join(INPUT_DIR, fname)
        out_p = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(in_p):
            process_file(in_p, out_p)

    print("\n🎉🎉🎉 全部 20 份数据 弱监督标注 + 置信度过滤 完成！")