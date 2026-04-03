#!/bin/bash
# DINOv2 下载修复脚本 - 百度内网专用

echo "=========================================="
echo "?? DINOv2 下载修复工具（百度内网）"
echo "=========================================="
echo ""

# 1. 删除损坏的缓存
echo "步骤 1: 清理损坏的缓存文件"
cache_dir="$HOME/.cache/huggingface/hub/models--facebook--dinov2-with-registers-base"
if [ -d "$cache_dir" ]; then
    echo "  发现缓存目录，正在删除..."
    rm -rf "$cache_dir"
    echo "  ✅ 已删除损坏的缓存"
else
    echo "  ℹ️  缓存目录不存在，跳过"
fi

echo ""

# 2. 设置正确的环境变量
echo "步骤 2: 配置百度内网环境"
export http_proxy='http://agent.baidu.com:8891'
export https_proxy='http://agent.baidu.com:8891'
export HTTP_PROXY='http://agent.baidu.com:8891'
export HTTPS_PROXY='http://agent.baidu.com:8891'

# 关键：使用原始 HuggingFace，不用镜像
export HF_ENDPOINT='https://huggingface.co'

echo "  ✅ http_proxy: $http_proxy"
echo "  ✅ https_proxy: $https_proxy"
echo "  ✅ HF_ENDPOINT: $HF_ENDPOINT"

echo ""

# 3. 验证网络连接
echo "步骤 3: 验证网络连接"
if timeout 5 curl -s -I https://huggingface.co > /dev/null 2>&1; then
    echo "  ✅ 网络连接正常"
else
    echo "  ⚠️  网络连接失败，请检查代理设置"
    echo "  提示：确保 VPN 已连接"
    exit 1
fi

echo ""

# 4. 使用 huggingface-cli 预下载（更稳定）
echo "步骤 4: 预下载 DINOv2 模型"
echo "  开始下载 facebook/dinov2-with-registers-base..."
echo "  （支持断点续传，可随时中断）"
echo ""

# 检查是否安装了 huggingface_hub
if python -c "import huggingface_hub" 2>/dev/null; then
    echo "  使用 huggingface-cli 下载..."
    huggingface-cli download facebook/dinov2-with-registers-base --resume-download
else
    echo "  使用 Python 下载..."
    python - <<'EOF'
from huggingface_hub import snapshot_download
import os

os.environ['http_proxy'] = 'http://agent.baidu.com:8891'
os.environ['https_proxy'] = 'http://agent.baidu.com:8891'
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'

print("正在下载 DINOv2 模型...")
try:
    cache_dir = snapshot_download(
        "facebook/dinov2-with-registers-base",
        resume_download=True
    )
    print(f"✅ 下载完成！缓存位置: {cache_dir}")
except Exception as e:
    print(f"❌ 下载失败: {e}")
    print("提示：可以稍后重新运行此脚本继续下载")
    exit(1)
EOF
fi

echo ""
echo "=========================================="
echo "✅ 修复完成！"
echo "=========================================="
echo ""
echo "现在可以重新运行你的脚本："
echo "  bash RAE_inference.sh"
echo ""
echo "如果下载中断，重新运行此脚本即可继续下载"