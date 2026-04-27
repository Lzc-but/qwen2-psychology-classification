import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 路径
BASE_MODEL = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"
LORA_PATH = "./output/qwen_psy_lora/checkpoint-500"
MERGED_SAVE_PATH = "/root/autodl-tmp/qwen_psy_merged"

# 加载
print("加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 加载 LoRA
print("加载 LoRA...")
model = PeftModel.from_pretrained(model, LORA_PATH)

# 合并
print("开始合并...")
model = model.merge_and_unload()

# 保存合并后的模型
print("保存合并模型...")
model.save_pretrained(MERGED_SAVE_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_SAVE_PATH)

print(f"✅ 合并完成！模型保存到：{MERGED_SAVE_PATH}")