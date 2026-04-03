export WANDB_KEY="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"
export ENTITY="rae"
export PROJECT="rae_retraining"
torchrun --standalone --nproc_per_node=8 \
  src/train_stage1.py \
  --config configs/stage1/training/DINOv2-B_decXL.yaml \
  --data-path /root/paddlejob/workspace/shenzhelun/shenzhelun/imagenet_1k_extract/train \
  --results-dir results/stage1 \
  --image-size 256 --precision bf16 \
  --ckpt /root/paddlejob/workspace/shenzhelun/shenzhelun/RAE/results/stage1/011-RAE-bf16/checkpoints/0035000.pt
  
  
#    \
#   --ckpt <optional_ckpt> \