# Qwen2.5 心理健康文本分类微调项目
基于 Qwen2.5-7B-Instruct 的心理健康文本情绪分类项目，包含 **数据清洗 → 打标签 → 训练集/测试集划分 → LoRA 微调 → 模型合并 → GGUF 导出** 全流程。

支持标签：正常 / 焦虑 / 低落 / 高风险

---

## 📌 项目说明
本项目使用 Qwen2.5-7B 大模型进行轻量级 LoRA 微调，实现对心理健康文本的自动情绪分类。整套流程可复现、可部署、可导出 GGUF 模型在本地运行。需要注意对数据进行清洗，提取title是在本地电脑运行，微调是在autodl的服务器上跑的。

其中原始数据来源于：https://aistudio.baidu.com/datasetdetail/37106

功能包括：
- 原始数据清洗、去重、提取标题
- 自动化/半自动化打标签
- 训练集 / 测试集自动划分
- LoRA 高效微调
- 合并模型并导出 GGUF 格式

## 📂 目录结构
```
qwen2-psychology-classification/
├── data/                     # 数据目录
│   ├── raw/                  # 原始未处理数据
│   ├── processed/            # 清洗、格式化后的数据
│   └── labeled/              # 最终标注好的数据
├── models/                   # 模型权重存放目录
├── outputs/                  # 训练输出、GGUF模型导出
├── scripts/                  # 核心执行脚本
│   ├── model_download.py     # 下载Qwen2.5模型
│   ├── 01_data_clean.py      # 数据清洗、去重、提取title
│   ├── 02_labeling.py        # 数据打标签
│   ├── split_test_set.py     # 划分测试集（20%）
│   ├── train.py              # LoRA微调训练
│   ├── merge_lora.py         # 合并LoRA与基模型
│   └── eval_original.py      # 模型效果评估
├── .gitignore
└── README.md
```

## 🚀 完整使用教程（命令行直接复制）

### 1. 安装依赖
```bash
pip install torch transformers datasets peft accelerate sentencepiece protobuf
```

### 2. 下载 Qwen2.5-7B 模型
下载前需要指定模型下载路径
```bash
python scripts/model_download.py
```

### 3. 数据预处理（清洗 + 去重 + 提取标题）
```bash
python scripts/01_data_clean.py
```

### 4. 打标签（情绪分类标注）
```bash
python scripts/02_labeling.py
```

### 5. 划分测试集（自动抽取 20% 数据）
```bash
python scripts/split_test_set.py
```

### 6. LoRA 微调训练
```bash
python scripts/train.py
```

训练完成后，模型保存在：
```
outputs/qwen_psy_lora/checkpoint-xxx
```

### 7. 合并 LoRA 与基础模型
```bash
python scripts/merge_lora.py
```

### 8. 导出 GGUF 模型（本地部署可用）
首先在autodl-tmp目录下先下载llama的zip，
```bash
wget https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.zip
unzip master.zip
mv llama.cpp-master llama.cpp
```
随后安装依赖库
```bash
cd llama.cpp
pip install -r requirements.txt
```
最后进入autodl-tmp目录下，运行下面命令：
```bash
python llama.cpp/convert_hf_to_gguf.py qwen_psy_merged --outfile qwen_psy_v1.f16.gguf --outtype f16
```
最终会得到qwen_psy_v1.f16.gguf文件，可通过du命令查看文件大小为15G，如果需要进一步压缩，量化为4bit，则需要用到qunatize，命令为：
```bash
cd llama.cpp && mkdir build && cd build
cmake ..
make -j4
```
随后回到autodl-tmp目录，使用下面命令进行量化：
```bash
./llama.cpp/build/bin/llama-quantize qwen_psy_v1.f16.gguf qwen_psy_v1.q4_0.gguf q4_0
```
最终会得到qwen_psy_v1.q4_0.gguf，大小仅为4G

### 9. 模型效果评估
```bash
python scripts/eval_original.py
```

## 📌 模型支持格式
✅ PyTorch (.bin / .safetensors)  
✅ GGUF (llama.cpp / Ollama / 本地UI)  
✅ LoRA 适配器  

## 📝 许可证
MIT License
