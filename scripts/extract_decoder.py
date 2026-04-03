#!/usr/bin/env python3
"""
从训练checkpoint中提取decoder权重并转换为model.pt格式

用法:
    python extract_decoder.py --input <checkpoint_path> --output <output_path> [--use_ema]

示例:
    python extract_decoder.py \
        --input results/stage1/011-RAE-bf16/checkpoints/0035000.pt \
        --output models/decoders/dinov2/wReg_base/ViTXL_n08/model_from_0035000.pt \
        --use_ema
"""

import argparse
import torch
import os


def extract_decoder_weights(checkpoint_path: str, output_path: str, use_ema: bool = True):
    """
    从checkpoint中提取decoder权重并转换格式
    
    Args:
        checkpoint_path: 输入checkpoint路径
        output_path: 输出model.pt路径
        use_ema: 是否使用EMA权重（推荐True，效果更稳定）
    """
    print("=" * 80)
    print("Decoder Weight Extractor")
    print("=" * 80)
    
    # 检查输入文件
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    file_size = os.path.getsize(checkpoint_path)
    print(f"\n📂 Input: {checkpoint_path}")
    print(f"💾 Size: {file_size:,} bytes ({file_size / (1024**3):.2f} GB)")
    
    # 加载checkpoint
    print("\n⏳ Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("✅ Checkpoint loaded")
    
    # 选择权重源
    source_key = 'ema' if use_ema and 'ema' in checkpoint else 'model'
    if source_key not in checkpoint:
        raise KeyError(f"Neither 'ema' nor 'model' found in checkpoint. Available keys: {list(checkpoint.keys())}")
    
    print(f"\n🔧 Using weights from: '{source_key}'")
    source_dict = checkpoint[source_key]
    
    # 提取decoder权重并去掉前缀
    new_state_dict = {}
    prefix = "decoder."
    
    for key, value in source_dict.items():
        if key.startswith(prefix):
            # 去掉 "decoder." 前缀
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
    
    if not new_state_dict:
        raise ValueError("No decoder weights found in checkpoint!")
    
    # 统计信息
    total_params = sum(v.numel() for v in new_state_dict.values() if isinstance(v, torch.Tensor))
    print(f"\n📊 Statistics:")
    print(f"   Extracted parameters: {len(new_state_dict)}")
    print(f"   Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n📁 Created output directory: {output_dir}")
    
    # 保存
    print(f"\n💾 Saving to: {output_path}")
    torch.save(new_state_dict, output_path)
    
    output_size = os.path.getsize(output_path)
    print(f"✅ Saved successfully! File size: {output_size:,} bytes ({output_size/1e6:.2f} MB)")
    
    # 显示前几个键作为验证
    print(f"\n📋 Sample keys (first 5):")
    for i, (key, value) in enumerate(list(new_state_dict.items())[:5]):
        if isinstance(value, torch.Tensor):
            print(f"   {i+1}. {key}: {value.shape}")
        else:
            print(f"   {i+1}. {key}: {type(value).__name__}")
    
    # 显示训练信息（如果有）
    if 'step' in checkpoint or 'epoch' in checkpoint:
        print(f"\n📈 Training info from checkpoint:")
        if 'step' in checkpoint:
            print(f"   Step: {checkpoint['step']}")
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
    
    print("\n" + "=" * 80)
    print("✅ Done!")
    print("=" * 80)
    
    return new_state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Extract decoder weights from training checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input checkpoint file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output model.pt file"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Use EMA weights (default: True)"
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Use model weights instead of EMA"
    )
    
    args = parser.parse_args()
    
    use_ema = not args.no_ema
    
    extract_decoder_weights(
        checkpoint_path=args.input,
        output_path=args.output,
        use_ema=use_ema
    )


if __name__ == "__main__":
    main()