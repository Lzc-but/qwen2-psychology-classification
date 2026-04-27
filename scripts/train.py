from datasets import Dataset
import pandas as pd
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

# ====================== 你的 Qwen 模型路径 ======================
model_path = "/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct"

def process_func(example):
    MAX_LENGTH = 384

    # ====================== 心理情绪分类模板（完全仿照你的甄嬛格式） ======================
    prompt = f"你是专业心理文本分类专家，请对用户的文本进行情绪分类。\n用户：{example['text']}\n心理评估结果："
    response = f"{example['label_text']}"

    # 编码输入和输出
    instruction_token = tokenizer(prompt, add_special_tokens=False)
    response_token = tokenizer(response, add_special_tokens=False)

    input_ids = instruction_token["input_ids"] + response_token["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction_token["attention_mask"] + response_token["attention_mask"] + [1]
    labels = [-100] * len(instruction_token["input_ids"]) + response_token["input_ids"] + [tokenizer.eos_token_id]

    # 长度截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == "__main__":
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.enable_input_require_grads()

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ====================== 加载你的心理情绪数据集 ======================
    DATA_DIR = "/root/autodl-tmp/dataset/labeled_final"
    all_data = []
    for i in range(1, 3):
        fname = f"part_{i:02d}.json"
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                all_data += json.load(f)

    # 标签映射（数字 → 文字）
    label_map = {
        "0": "正常",
        "1": "焦虑",
        "2": "低落",
        "3": "高风险"
    }

    # 构造数据集格式
    format_data = []
    for item in all_data:
        format_data.append({
            "text": item["text"],
            "label_text": label_map[item["label"]]
        })

    ds = Dataset.from_list(format_data)
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    # LoRA 配置
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 训练参数
    args = TrainingArguments(
        output_dir="./output/qwen_psy_lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=2e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        bf16=True
    )

    # 启动训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()