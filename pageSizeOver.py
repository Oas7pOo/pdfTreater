# -*- coding: utf-8 -*-

import sys
import os
import gc
import argparse
import hashlib
import tempfile
from pathlib import Path

# 修复 basicsr 与新版 torchvision 的兼容性问题
import torchvision.transforms.functional as torchvision_functional
sys.modules.setdefault("torchvision.transforms.functional_tensor", torchvision_functional)

import cv2
import fitz
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


# 先尝试整图，失败后严格按照此顺序切片
TILE_SEQUENCE = [0, 1024, 768, 512, 384, 256, 192, 128]


def setup_upscaler(model_path):
    """初始化 Real-ESRGAN。"""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到模型文件：{model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        half=device.type == "cuda",
        device=device
    )

    precision = "FP16" if device.type == "cuda" else "FP32"
    print(f"模型初始化完成：设备={device}，精度={precision}")
    return upscaler


def clear_memory(upscaler=None):
    """清除 Real-ESRGAN 内部张量引用和 CUDA 缓存。"""
    if upscaler is not None:
        if hasattr(upscaler, "img"):
            upscaler.img = None
        if hasattr(upscaler, "output"):
            upscaler.output = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_gpu_memory(prefix):
    """打印当前显存使用情况。"""
    if not torch.cuda.is_available():
        print(prefix)
        return

    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    free, total = torch.cuda.mem_get_info()

    print(
        f"{prefix}，已分配={allocated:.0f} MB，"
        f"已保留={reserved:.0f} MB，"
        f"空闲={free / 1024 ** 2:.0f}/{total / 1024 ** 2:.0f} MB"
    )


def is_memory_error(error):
    """判断异常是否属于显存不足。"""
    message = str(error).lower()

    return (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "cuda out of memory" in message
        or "out of memory" in message
        or "output_tile" in message
    )


def enhance_with_tiling(upscaler, image, outscale):
    """
    先尝试整图处理。

    整图失败后依次尝试：
    1024、768、512、384、256、192、128。
    """
    last_error = ""

    for index, tile_size in enumerate(TILE_SEQUENCE):
        clear_memory(upscaler)

        # RealESRGANer 实际读取的是 tile_size
        upscaler.tile_size = tile_size

        mode = "整图处理" if tile_size == 0 else f"tile={tile_size}"
        print_gpu_memory(f"    尝试 {mode}")

        try:
            with torch.inference_mode():
                result = upscaler.enhance(image, outscale=outscale)

            print(f"    -> {mode} 成功")
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
                print(f"    -> {mode} 显存不足，继续尝试 tile={next_tile}")

    raise RuntimeError(f"整图及所有 tile 均处理失败。最后错误：{last_error}")


def decode_image(data):
    """将图片字节解码为 OpenCV 数组。"""
    image = cv2.imdecode(
        np.frombuffer(data, np.uint8),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise ValueError("OpenCV 无法解码该图片")

    return image


def extract_image_array(doc, xref, smask):
    """
    从 PDF 提取图片。

    如果图片使用独立软蒙版，则将软蒙版合并为透明通道。
    """
    if not smask:
        info = doc.extract_image(xref)

        if not info or not info.get("image"):
            raise ValueError("无法从 PDF 提取图片")

        return decode_image(info["image"])

    base = fitz.Pixmap(doc, xref)
    mask = fitz.Pixmap(doc, smask)

    try:
        if base.colorspace and base.colorspace.n > 3:
            base = fitz.Pixmap(fitz.csRGB, base)

        combined = fitz.Pixmap(base, mask)
        return decode_image(combined.tobytes("png"))

    finally:
        base = None
        mask = None


def get_image_hash(image):
    """
    按解码后的像素计算指纹。

    即使两个图片属于不同 xref，只要实际像素相同，也会命中缓存。
    """
    image = np.ascontiguousarray(image)

    digest = hashlib.sha256()
    digest.update(str(image.shape).encode())
    digest.update(str(image.dtype).encode())
    digest.update(image.data)

    return digest.hexdigest()


def upscale_image(image, upscaler, outscale):
    """超分单张图片，并保留透明通道。"""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    has_alpha = image.ndim == 3 and image.shape[2] == 4

    if has_alpha:
        rgb = np.ascontiguousarray(image[:, :, :3])
        alpha = np.ascontiguousarray(image[:, :, 3])

        rgb_upscaled, _ = enhance_with_tiling(
            upscaler,
            rgb,
            outscale
        )

        height, width = rgb_upscaled.shape[:2]

        alpha_upscaled = cv2.resize(
            alpha,
            (width, height),
            interpolation=cv2.INTER_CUBIC
        )

        result = np.dstack(
            (rgb_upscaled, alpha_upscaled)
        )

    else:
        result, _ = enhance_with_tiling(
            upscaler,
            image,
            outscale
        )

    success, encoded = cv2.imencode(".png", result)

    if not success:
        raise RuntimeError("超分结果编码为 PNG 失败")

    width = result.shape[1]
    height = result.shape[0]

    return encoded.tobytes(), width, height


def scan_pdf_images(doc):
    """
    扫描整份 PDF。

    同一个 xref 可能在不同页面、不同尺寸下重复使用。
    因此记录它在整份 PDF 中的最低有效 DPI，再决定是否处理。
    """
    records = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        seen_on_page = set()

        for item in page.get_images(full=True):
            xref = item[0]
            smask = item[1]
            width = item[2]
            height = item[3]

            # 同一页面资源列表中可能重复出现同一个 xref
            if xref in seen_on_page:
                continue

            seen_on_page.add(xref)
            rects = page.get_image_rects(xref)

            if not rects or width <= 0 or height <= 0:
                continue

            record = records.setdefault(
                xref,
                {
                    "xref": xref,
                    "smask": smask,
                    "width": width,
                    "height": height,
                    "first_page": page_num,
                    "min_dpi": float("inf"),
                    "occurrences": 0
                }
            )

            for rect in rects:
                if rect.width <= 0 or rect.height <= 0:
                    continue

                dpi_x = width * 72.0 / rect.width
                dpi_y = height * 72.0 / rect.height
                effective_dpi = min(dpi_x, dpi_y)

                record["min_dpi"] = min(
                    record["min_dpi"],
                    effective_dpi
                )

                record["occurrences"] += 1

    return sorted(
        records.values(),
        key=lambda item: (
            item["first_page"],
            item["xref"]
        )
    )


def replace_pdf_image(doc, record, image_path):
    """全局替换 PDF 中指定 xref 的图片。"""
    page = doc[record["first_page"]]

    page.replace_image(
        record["xref"],
        filename=str(image_path)
    )


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        description="PDF 低分辨率图片 AI 自动超分"
    )

    parser.add_argument(
        "--input",
        "-i",
        default=r"C:\Users\HASEE\Downloads\龙与香辛料719打印测试\龙与香辛料719_a5.pdf",
        help="输入 PDF 路径"
    )

    parser.add_argument(
        "--output",
        "-o",
        default=r"C:\Users\HASEE\Downloads\龙与香辛料719打印测试\output_a5.pdf",
        help="输出 PDF 路径"
    )

    parser.add_argument(
        "--model",
        default="RealESRGAN_x4plus_anime_6B.pth",
        help="模型文件路径"
    )

    parser.add_argument(
        "--dpi",
        "-d",
        type=float,
        default=300,
        help="低于该 DPI 时进行超分"
    )

    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=2.0,
        help="最终放大倍数"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"输入 PDF 不存在：{input_path}"
        )

    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "输入和输出不能是同一个 PDF 文件"
        )

    if output_path.exists():
        raise FileExistsError(
            f"输出文件已存在，请先删除或更换路径：{output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    upscaler = setup_upscaler(args.model)
    doc = fitz.open(str(input_path))

    success_count = 0
    reused_count = 0
    failed_count = 0
    records = []
    candidates = []

    try:
        print("正在扫描 PDF 图片及其最低有效 DPI...")

        records = scan_pdf_images(doc)

        candidates = [
            record
            for record in records
            if record["min_dpi"] < args.dpi
        ]

        print(
            f"共发现 {len(records)} 个图片对象，"
            f"其中 {len(candidates)} 个需要处理。"
        )

        # 将高清缓存放在临时目录，避免大量高清图片堆积在内存中
        with tempfile.TemporaryDirectory(
            prefix="pdf_upscale_"
        ) as cache_dir:

            cache_dir = Path(cache_dir)

            # image_hash -> 高清 PNG 临时文件路径
            successful_cache = {}

            # 记录已经失败过的重复图片
            failed_cache = set()

            for index, record in enumerate(
                candidates,
                start=1
            ):
                xref = record["xref"]
                image = None
                image_hash = None

                print(
                    f"[{index}/{len(candidates)}] "
                    f"页码 {record['first_page'] + 1}，"
                    f"xref {xref}，"
                    f"最低 DPI={record['min_dpi']:.0f}，"
                    f"出现 {record['occurrences']} 次"
                )

                try:
                    image = extract_image_array(
                        doc,
                        xref,
                        record["smask"]
                    )

                    image_hash = get_image_hash(image)

                    # 相同图片之前已经失败，直接跳过
                    if image_hash in failed_cache:
                        print(
                            "    -> 相同图片此前处理失败，跳过"
                        )
                        failed_count += 1
                        continue

                    # 不同 xref 但图片内容相同，直接复用缓存
                    if image_hash in successful_cache:
                        replace_pdf_image(
                            doc,
                            record,
                            successful_cache[image_hash]
                        )

                        print(
                            "    -> 命中重复图片缓存，直接复用高清结果"
                        )

                        reused_count += 1
                        continue

                    new_bytes, new_width, new_height = upscale_image(
                        image,
                        upscaler,
                        args.scale
                    )

                    cache_path = cache_dir / f"{image_hash}.png"
                    cache_path.write_bytes(new_bytes)

                    replace_pdf_image(
                        doc,
                        record,
                        cache_path
                    )

                    successful_cache[image_hash] = cache_path

                    print(
                        f"    -> 成功，新尺寸："
                        f"{new_width}×{new_height}"
                    )

                    success_count += 1

                except Exception as error:
                    print(
                        f"    -> 处理失败："
                        f"{type(error).__name__}: {error}"
                    )

                    if image_hash is not None:
                        failed_cache.add(image_hash)

                    failed_count += 1

                finally:
                    image = None
                    clear_memory(upscaler)

        print("处理完毕，正在保存输出 PDF...")

        doc.save(
            str(output_path),
            garbage=4,
            deflate=True
        )

    finally:
        doc.close()

        clear_memory(upscaler)
        del upscaler
        clear_memory()

    print("\n全部完成")
    print(f"新处理：{success_count}")
    print(f"缓存复用：{reused_count}")
    print(f"处理失败：{failed_count}")
    print(f"DPI 达标跳过：{len(records) - len(candidates)}")
    print(f"输出文件：{output_path}")


if __name__ == "__main__":
    main()