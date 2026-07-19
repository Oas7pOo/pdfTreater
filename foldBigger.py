# -*- coding: utf-8 -*-

import os
import sys
import gc
import argparse
from pathlib import Path

# 修复 basicsr 与新版 torchvision 的兼容性问题
import torchvision.transforms.functional as torchvision_functional
sys.modules.setdefault("torchvision.transforms.functional_tensor", torchvision_functional)

import cv2
import numpy as np
import torch

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 首先尝试整图；整图失败后，严格按照该顺序切片
TILE_SEQUENCE = [0, 1024, 768, 512, 384, 256, 192, 128]


def print_gpu_memory(prefix=""):
    """打印当前显存占用。"""
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2

    try:
        free, total = torch.cuda.mem_get_info()
        print(
            f"{prefix}已分配={allocated:.2f} MB，"
            f"已保留={reserved:.2f} MB，"
            f"空闲={free / 1024 ** 2:.2f}/{total / 1024 ** 2:.2f} MB"
        )
    except Exception:
        print(f"{prefix}已分配={allocated:.2f} MB，已保留={reserved:.2f} MB")


def clear_memory(upscaler=None):
    """清除 RealESRGAN 内部张量引用和 CUDA 缓存。"""
    if upscaler is not None:
        if hasattr(upscaler, "img"):
            upscaler.img = None
        if hasattr(upscaler, "output"):
            upscaler.output = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_memory_error(error):
    """判断异常是否由显存不足引起。"""
    message = str(error).lower()

    return (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "cuda out of memory" in message
        or "out of memory" in message
        or "output_tile" in message
    )


def setup_upscaler(model_path, use_half=True):
    """加载 Real-ESRGAN 模型。"""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到模型文件：{model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = use_half and device.type == "cuda"

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=6,
        num_grow_ch=32,
        scale=4
    )

    upscaler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=half,
        device=device
    )

    print(f"运行设备：{device}")
    print(f"推理精度：{'FP16' if half else 'FP32'}")
    print_gpu_memory("模型加载后：")

    return upscaler


def enhance_with_tiling(upscaler, image, outscale):
    """
    首先整图处理。

    整图显存不足时依次尝试：
    1024、768、512、384、256、192、128。
    """
    last_error = ""

    for index, tile_size in enumerate(TILE_SEQUENCE):
        clear_memory(upscaler)

        # RealESRGANer 实际读取的属性是 tile_size，不是 tile
        upscaler.tile_size = tile_size

        mode = "整图处理" if tile_size == 0 else f"tile={tile_size}"
        print_gpu_memory(f"当前尝试 {mode}：")

        try:
            with torch.inference_mode():
                result = upscaler.enhance(image, outscale=outscale)

            print(f"  -> {mode} 成功")
            clear_memory(upscaler)
            return result

        except Exception as error:
            if not is_memory_error(error):
                clear_memory(upscaler)
                raise

            last_error = f"{type(error).__name__}: {error}"
            clear_memory(upscaler)

            if index + 1 < len(TILE_SEQUENCE):
                next_tile = TILE_SEQUENCE[index + 1]
                print(f"  -> {mode} 显存不足，继续尝试 tile={next_tile}")
            else:
                print(f"  -> tile={tile_size} 显存仍然不足")

    raise RuntimeError(f"整图及所有 tile 均处理失败。最后错误：{last_error}")


def read_image(path):
    """读取包含中文或特殊字符路径的图片。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception as error:
        print(f"  读取图片失败：{error}")
        return None


def save_image(path, image):
    """保存到包含中文或特殊字符的路径。"""
    path = Path(path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        extension = ".png"
        path = path.with_suffix(extension)

    # JPEG 不支持透明通道
    if extension in {".jpg", ".jpeg"} and image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    try:
        success, buffer = cv2.imencode(extension, image)

        if not success:
            print(f"  图片编码失败：{path}")
            return False

        buffer.tofile(str(path))
        return True

    except Exception as error:
        print(f"  保存图片失败：{error}")
        return False


def upscale_image(input_path, output_path, upscaler, outscale):
    """处理单张图片，并保留 PNG 透明通道。"""
    image = None
    final_image = None

    try:
        image = read_image(input_path)

        if image is None:
            print("  跳过：无法解码图片")
            return False

        height, width = image.shape[:2]
        print(f"  输入尺寸：{width}×{height}，dtype={image.dtype}，shape={image.shape}")

        has_alpha = image.ndim == 3 and image.shape[2] == 4

        if has_alpha:
            rgb = np.ascontiguousarray(image[:, :, :3])
            alpha = np.ascontiguousarray(image[:, :, 3])

            rgb_upscaled, _ = enhance_with_tiling(upscaler, rgb, outscale)

            new_height, new_width = rgb_upscaled.shape[:2]
            alpha_upscaled = cv2.resize(
                alpha,
                (new_width, new_height),
                interpolation=cv2.INTER_CUBIC
            )

            final_image = np.dstack((rgb_upscaled, alpha_upscaled))

            del rgb, alpha, rgb_upscaled, alpha_upscaled
        else:
            final_image, _ = enhance_with_tiling(upscaler, image, outscale)

        return save_image(output_path, final_image)

    except Exception as error:
        print(f"  处理失败：{type(error).__name__}: {error}")
        return False

    finally:
        image = None
        final_image = None
        clear_memory(upscaler)


def main():
    torch.backends.cudnn.enabled = True

    # 关闭自动调优，避免不同图片尺寸触发额外显存工作区
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description="批量图片 Real-ESRGAN 超分")

    parser.add_argument(
        "--input_dir",
        default=r"C:\Users\HASEE\Downloads\“龙与香辛料776打印测试”indesign\Links",
        help="输入目录"
    )
    parser.add_argument(
        "--output_dir",
        default=r"C:\Users\HASEE\Downloads\“龙与香辛料776打印测试”indesign\Link",
        help="输出目录"
    )
    parser.add_argument(
        "--model",
        default="RealESRGAN_x4plus_anime_6B.pth",
        help="模型文件路径"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.0,
        help="最终输出放大倍数"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="强制使用 FP32；默认在 CUDA 上使用 FP16"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    print(f"找到 {len(image_files)} 张图片，开始处理...")

    upscaler = setup_upscaler(args.model, use_half=not args.fp32)

    success_count = 0
    failed_count = 0
    skipped_count = 0

    try:
        for index, image_path in enumerate(image_files, 1):
            output_path = output_dir / image_path.name

            if output_path.exists():
                print(f"[{index}/{len(image_files)}] 已存在，跳过：{image_path.name}")
                skipped_count += 1
                continue

            print(f"[{index}/{len(image_files)}] 处理中：{image_path.name}")

            if upscale_image(image_path, output_path, upscaler, args.scale):
                print("  保存成功")
                success_count += 1
            else:
                print("  处理失败")
                failed_count += 1

    finally:
        clear_memory(upscaler)
        del upscaler
        clear_memory()

    print("\n全部处理完成")
    print(f"成功：{success_count}")
    print(f"失败：{failed_count}")
    print(f"跳过：{skipped_count}")


if __name__ == "__main__":
    main()