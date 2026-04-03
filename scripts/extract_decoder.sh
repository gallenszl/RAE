python scripts/extract_decoder.py \
    --input /root/paddlejob/workspace/shenzhelun/shenzhelun/RAE/results/stage1/013-RAE-bf16/checkpoints/0055000.pt \
    --output /root/paddlejob/workspace/shenzhelun/shenzhelun/RAE/results/stage1/013-RAE-bf16/checkpoints/0055000_decoder.pt \
    --use_ema  # 使用EMA权重(默认)
    # 或 --no_ema  # 使用model权重