#!/usr/bin/env python3
"""
Run a stage-1 RAE reconstruction from a config file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from stage1 import RAE
import torch.nn.functional as F

DEFAULT_IMAGE = Path("assets/pixabay_cat.png")


def get_device(explicit: str | None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    tensor = transforms.ToTensor()(image).unsqueeze(0)  # (1, C, H, W)
    return tensor

def load_depth(image_path: Path) -> torch.Tensor:
        depth = Image.open(image_path)
        depth = np.array(depth)
        # depth = np.array(depth.resize((resize_w, resize_h), resample=Image.NEAREST))
        mask = depth < 65534 # valid mask, 1=fg, 0=bg

        max_depth = 100
        min_depth = 10
        depth_range = max_depth - min_depth

        depth = depth / 65535.0 * depth_range + min_depth
        depth = depth * mask

        # normalize by first view's translation
        # depth = torch.from_numpy(depth)
        depth = torch.from_numpy(depth).float()  # [H, W]
        depth = depth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return depth


def reconstruct(rae: RAE, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        latent = rae.encode(image)
        # import pdb; pdb.set_trace()  ### shape is [1,768,32,32] no matter the resolution of the input image
        recon = rae.decode(latent)
    return latent, recon

def save_depth_channel(recon: torch.Tensor, path: str, channel: int = 0, normalize: bool = True):
    """
    recon: 形状 [1, 3, H, W] 的 tensor
    path:  保存路径，比如 "out.png"
    channel: 取哪个通道(0/1/2)
    normalize: 是否把当前通道归一化到 [0, 1] 再映射到 0~65535
    """
    # 取出 [C, H, W]
    if recon.dim() == 4:
        # [N, C, H, W]，只取第 0 个 batch
        recon_ch = recon[0, channel]      # [H, W]
    elif recon.dim() == 3:
        # [C, H, W]
        recon_ch = recon[channel]
    else:
        raise ValueError(f"Unexpected recon shape: {recon.shape}")

    depth = recon_ch.detach().cpu().float().numpy()  # [H, W]

    if normalize:
        # 归一化到 [0, 1]，再映射到 0~65535
        d_min = depth.min()
        d_max = depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = depth - d_min  # 全常数就全 0
    # 映射到 0~65535，转为 uint16
    depth_u16 = (depth * 65535.0).clip(0, 65535).astype(np.uint16)

    img = Image.fromarray(depth_u16, mode="I;16")
    img.save(path)

def calc_psnr(img, recon, max_val=1.0):
    # img, recon: [B, C, H, W] 或 [C, H, W]
    # 先保证是 float
    img = img.float()
    recon = recon.float()

    mse = torch.mean((img - recon) ** 2)

    if mse == 0:
        return float('inf')

    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct an input image using a Stage-1 RAE loaded from config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config with a stage_1 section.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Input image to reconstruct (default: {DEFAULT_IMAGE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recon.png"),
        help="Where to save the reconstructed image (default: recon.png).",
    )
    parser.add_argument(
        "--device",
        help="Torch device to use (e.g. cuda, cuda:1, cpu). Auto-detect if omitted.",
    )
    args = parser.parse_args()

    device = get_device(args.device)

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError(
            f"No stage_1 section found in config {args.config}. "
            "Please supply a config with a stage_1 target."
        )

    torch.set_grad_enabled(False)
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    image = load_image(args.image).to(device)
    # image = load_depth(args.image).to(device)
    # import pdb; pdb.set_trace()
    latent, recon = reconstruct(rae, image)
    image = F.interpolate(
        image, size=(256, 256),
        mode="bilinear", align_corners=False
    )
    recon = recon.clamp(0.0, 1.0)

    ### 
    if recon.shape[-2:] != image.shape[-2:]:
        recon = F.interpolate(recon, size=image.shape[-2:], mode="bilinear", align_corners=False)

    # 3) 宽度方向拼接： [B,C,H,W] -> [B,C,H,2W]
    concat = torch.cat([image, recon], dim=3)

    # 4) 转成 numpy 保存（RGB）
    concat_np = (
        concat.mul(255)
            .permute(0, 2, 3, 1)   # BCHW -> BHWC
            .squeeze(0)
            .to("cpu", dtype=torch.uint8)
            .numpy()
    )

    out_concat = args.output.with_name(args.output.stem + "_concat.png")
    Image.fromarray(concat_np).save(out_concat)
    print(f"Saved concat to {out_concat.resolve()}")
    ###

    recon_np = recon.mul(255).permute(0, 2, 3, 1).squeeze(0).to("cpu", dtype=torch.uint8).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(recon_np).save(args.output)
    # save_image(recon, args.output)
    # save_depth_channel(recon, args.output, 0)


    psnr_val = calc_psnr(image, recon, max_val=1.0)
    print(f"PSNR: {psnr_val:.4f} dB")
    print(f"Saved reconstruction to {args.output.resolve()}")
    print(f"Input shape: {tuple(image.shape)}, latent shape: {tuple(latent.shape)}, recon shape: {tuple(recon.shape)}")


if __name__ == "__main__":
    main()
