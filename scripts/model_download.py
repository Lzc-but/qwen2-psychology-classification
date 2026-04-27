
from modelscope import snapshot_download

# 下载 Qwen2.5-1.8B-Chat（国内稳定版）
model_dir = snapshot_download(
    'Qwen/Qwen2.5-7B-Instruct',
    cache_dir="/root/autodl-tmp/models",
    revision="master"
)

print("✅ 模型下载成功！路径：", model_dir)