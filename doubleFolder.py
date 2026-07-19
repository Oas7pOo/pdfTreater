# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


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


def upscale_image_bicubic(image, scale=2.0):
    """
    使用双三次插值放大图片。
    若图片含有 Alpha 通道，分别对 RGB 和 Alpha 通道放大后合并。
    返回放大后的图像。
    """
    height, width = image.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)

    if image.ndim == 3 and image.shape[2] == 4:
        # 分离 RGB 和 Alpha 通道
        rgb = image[:, :, :3]
        alpha = image[:, :, 3]

        rgb_upscaled = cv2.resize(
            rgb,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )
        alpha_upscaled = cv2.resize(
            alpha,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )

        return np.dstack((rgb_upscaled, alpha_upscaled))
    else:
        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )


def main():
    parser = argparse.ArgumentParser(description="批量图片双三次插值放大 (2x)")

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
        "--scale",
        type=float,
        default=2.0,
        help="放大倍数，默认为 2.0"
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
    print(f"放大倍数：{args.scale}x，插值方法：双三次插值")

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for index, image_path in enumerate(image_files, 1):
        output_path = output_dir / image_path.name

        if output_path.exists():
            print(f"[{index}/{len(image_files)}] 已存在，跳过：{image_path.name}")
            skipped_count += 1
            continue

        print(f"[{index}/{len(image_files)}] 处理中：{image_path.name}")

        image = read_image(image_path)
        if image is None:
            print("  跳过：无法解码图片")
            failed_count += 1
            continue

        height, width = image.shape[:2]
        print(f"  输入尺寸：{width}×{height}，dtype={image.dtype}，shape={image.shape}")

        try:
            result = upscale_image_bicubic(image, scale=args.scale)
            if save_image(output_path, result):
                print("  保存成功")
                success_count += 1
            else:
                print("  处理失败")
                failed_count += 1
        except Exception as error:
            print(f"  处理失败：{type(error).__name__}: {error}")
            failed_count += 1
        finally:
            # 主动释放图像内存
            del image, result

    print("\n全部处理完成")
    print(f"成功：{success_count}")
    print(f"失败：{failed_count}")
    print(f"跳过：{skipped_count}")


if __name__ == "__main__":
    main()