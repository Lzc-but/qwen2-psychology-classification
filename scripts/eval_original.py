import json
import torch
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 路径
MODEL_PATH = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"
TEST_FILE = "/root/autodl-tmp/dataset/test_set.json"

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

# 分类Prompt
PROMPT = """你是专业心理文本分类专家，只输出数字 0/1/2/3
0：正常 1：焦虑 2：低落 3：高风险
文本：{}
输出数字："""

# 推理
def predict(text):
    prompt = PROMPT.format(text)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([inputs], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(result)
    find = re.findall(r'[0123]', result)
    if find:
        return find[0]
    return "0"

# 加载测试集
with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

test_data = test_data
# 评估
correct = 0
total = len(test_data)

for item in tqdm(test_data):
    text = item["text"]
    true_label = item["label"]
    pred = predict(text)
    if pred == true_label:
        correct += 1

acc = correct / total
print(f"\n🎉 未微调模型 准确率：{acc:.2%}")
print(f"正确：{correct} / 总数：{total}")