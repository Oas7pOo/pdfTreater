# -*- coding: utf-8 -*-

import os
import argparse
import hashlib
import tempfile
from pathlib import Path

import cv2
import fitz
import numpy as np


def upscale_image_bicubic(image, scale=2.0):
    """
    使用双三次插值放大图片，返回编码后的 PNG 字节以及新尺寸。
    透明通道会被独立处理后再合并。
    """
    # 灰度图转 RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    height, width = image.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)

    has_alpha = image.ndim == 3 and image.shape[2] == 4

    if has_alpha:
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

        result = np.dstack((rgb_upscaled, alpha_upscaled))
    else:
        result = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )

    success, encoded = cv2.imencode(".png", result)
    if not success:
        raise RuntimeError("放大结果编码为 PNG 失败")

    return encoded.tobytes(), new_width, new_height


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


def scan_pdf_images(doc):
    """
    扫描整份 PDF，记录每个图片对象的尺寸、最低有效 DPI 及出现次数。
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
                record["min_dpi"] = min(record["min_dpi"], effective_dpi)
                record["occurrences"] += 1

    return sorted(
        records.values(),
        key=lambda item: (item["first_page"], item["xref"])
    )


def replace_pdf_image(doc, record, image_path):
    """全局替换 PDF 中指定 xref 的图片。"""
    page = doc[record["first_page"]]
    page.replace_image(record["xref"], filename=str(image_path))


def main():
    parser = argparse.ArgumentParser(
        description="PDF 低分辨率图片双三次插值自动放大"
    )

    parser.add_argument(
        "--input", "-i",
        default=r"C:\Users\HASEE\Downloads\龙与香辛料719打印测试\龙与香辛料719_a5.pdf",
        help="输入 PDF 路径"
    )
    parser.add_argument(
        "--output", "-o",
        default=r"C:\Users\HASEE\Downloads\龙与香辛料719打印测试\output_double_a5.pdf",
        help="输出 PDF 路径"
    )
    parser.add_argument(
        "--dpi", "-d",
        type=float, default=300,
        help="低于该 DPI 时进行放大（默认 300）"
    )
    parser.add_argument(
        "--scale", "-s",
        type=float, default=2.0,
        help="放大倍数（默认 2.0）"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"输入 PDF 不存在：{input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("输入和输出不能是同一个 PDF 文件")
    if output_path.exists():
        raise FileExistsError(f"输出文件已存在，请先删除或更换路径：{output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
            r for r in records
            if r["min_dpi"] < args.dpi
        ]
        print(
            f"共发现 {len(records)} 个图片对象，"
            f"其中 {len(candidates)} 个需要处理。"
        )

        # 临时目录存放放大后的 PNG，供 PDF 替换使用
        with tempfile.TemporaryDirectory(prefix="pdf_bicubic_") as cache_dir:
            cache_dir = Path(cache_dir)
            # image_hash -> 临时高清 PNG 路径
            successful_cache = {}
            # 处理失败的图片哈希
            failed_cache = set()

            for index, record in enumerate(candidates, start=1):
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
                    image = extract_image_array(doc, xref, record["smask"])
                    image_hash = get_image_hash(image)

                    # 相同图片之前已失败
                    if image_hash in failed_cache:
                        print("    -> 相同图片此前处理失败，跳过")
                        failed_count += 1
                        continue

                    # 相同图片已有成功缓存
                    if image_hash in successful_cache:
                        replace_pdf_image(doc, record, successful_cache[image_hash])
                        print("    -> 命中重复图片缓存，直接复用高清结果")
                        reused_count += 1
                        continue

                    # 双三次插值放大
                    new_bytes, new_width, new_height = upscale_image_bicubic(
                        image, scale=args.scale
                    )

                    cache_path = cache_dir / f"{image_hash}.png"
                    cache_path.write_bytes(new_bytes)

                    replace_pdf_image(doc, record, cache_path)
                    successful_cache[image_hash] = cache_path

                    print(f"    -> 成功，新尺寸：{new_width}×{new_height}")
                    success_count += 1

                except Exception as error:
                    print(f"    -> 处理失败：{type(error).__name__}: {error}")
                    if image_hash is not None:
                        failed_cache.add(image_hash)
                    failed_count += 1
                finally:
                    image = None

        print("处理完毕，正在保存输出 PDF...")
        doc.save(str(output_path), garbage=4, deflate=True)

    finally:
        doc.close()

    print("\n全部完成")
    print(f"新处理：{success_count}")
    print(f"缓存复用：{reused_count}")
    print(f"处理失败：{failed_count}")
    print(f"DPI 达标跳过：{len(records) - len(candidates)}")
    print(f"输出文件：{output_path}")


if __name__ == "__main__":
    main()