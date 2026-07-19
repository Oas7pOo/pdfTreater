# -*- coding: utf-8 -*-
"""
将 PDF 按指定成品尺寸重新排版，并生成 TrimBox / BleedBox。

核心原则：
1. 主体内容只映射到 TrimBox，不再为了制造出血而牺牲成品区内容。
2. 使用 show_pdf_page(..., clip=...) 在源页上直接选取裁切窗口，保持矢量。
3. 支持三种书页方向：
   - right_first: 第 1 页为右页；奇数页裁右侧，偶数页裁左侧。
   - left_first : 第 1 页为左页；奇数页裁左侧，偶数页裁右侧。
   - single     : 所有页面居中裁切。
4. 可选“缩放背景式假出血”：主体仍精确落在 TrimBox，放大的背景只显示在出血区。

运行示例：
python resize_book_pdf.py input.pdf output.pdf \
    --trim-w 140 --trim-h 210 --bleed 5 \
    --layout right_first --resize cover --bleed-mode zoom \
    --center-pages 1
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Optional, Set

try:
    import pymupdf as fitz  # 新版推荐名称
except ImportError:  # 兼容旧版
    import fitz  # type: ignore


MM2PT = 72.0 / 25.4
PT2MM = 25.4 / 72.0

LAYOUT_RIGHT_FIRST = "right_first"
LAYOUT_LEFT_FIRST = "left_first"
LAYOUT_SINGLE = "single"
VALID_LAYOUTS = {LAYOUT_RIGHT_FIRST, LAYOUT_LEFT_FIRST, LAYOUT_SINGLE}

RESIZE_COVER = "cover"
RESIZE_CONTAIN = "contain"
VALID_RESIZE_MODES = {RESIZE_COVER, RESIZE_CONTAIN}

BLEED_ZOOM = "zoom"
BLEED_WHITE = "white"
VALID_BLEED_MODES = {BLEED_ZOOM, BLEED_WHITE}

CROP_RIGHT = "crop_right"  # 从源页右侧裁掉，保留左侧（右手页时保护书脊）
CROP_LEFT = "crop_left"    # 从源页左侧裁掉，保留右侧（左手页时保护书脊）
CROP_CENTER = "center"

VERTICAL_TOP = "top"
VERTICAL_CENTER = "center"
VERTICAL_BOTTOM = "bottom"
VALID_VERTICAL_ANCHORS = {VERTICAL_TOP, VERTICAL_CENTER, VERTICAL_BOTTOM}


def mm_to_pt(value_mm: float) -> float:
    return value_mm * MM2PT


def validate_positive(name: str, value: float, allow_zero: bool = False) -> None:
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} 不能小于 0，当前值为 {value}")
    elif value <= 0:
        raise ValueError(f"{name} 必须大于 0，当前值为 {value}")


def parse_page_numbers(text: str) -> Set[int]:
    """
    解析 1-based 页码，例如：
      "1"       -> {1}
      "1,3,5"   -> {1, 3, 5}
      "1-3,8"   -> {1, 2, 3, 8}
      ""        -> set()
    """
    pages: Set[int] = set()
    text = text.strip()
    if not text:
        return pages

    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start <= 0 or end <= 0:
                raise ValueError("页码必须从 1 开始")
            if end < start:
                raise ValueError(f"页码范围错误：{part}")
            pages.update(range(start, end + 1))
        else:
            page = int(part)
            if page <= 0:
                raise ValueError("页码必须从 1 开始")
            pages.add(page)
    return pages


def get_horizontal_crop_strategy(
    page_index: int,
    layout_mode: str,
    center_pages: Set[int],
) -> str:
    """
    返回源页的横向裁切策略。

    page_index 为 0-based；center_pages 使用用户习惯的 1-based 页码。
    """
    page_number = page_index + 1
    if page_number in center_pages:
        return CROP_CENTER

    if layout_mode == LAYOUT_SINGLE:
        return CROP_CENTER

    is_pdf_odd_page = (page_index % 2 == 0)  # index 0 = PDF 第 1 页

    if layout_mode == LAYOUT_RIGHT_FIRST:
        # 第 1、3、5...页是右手页：书脊在左，应该从右侧裁掉。
        return CROP_RIGHT if is_pdf_odd_page else CROP_LEFT

    if layout_mode == LAYOUT_LEFT_FIRST:
        # 第 1、3、5...页是左手页：书脊在右，应该从左侧裁掉。
        return CROP_LEFT if is_pdf_odd_page else CROP_RIGHT

    raise ValueError(f"未知 layout_mode: {layout_mode}")


def crop_rect_to_aspect(
    source_rect: fitz.Rect,
    target_aspect: float,
    horizontal_strategy: str,
    vertical_anchor: str = VERTICAL_CENTER,
) -> fitz.Rect:
    """
    从 source_rect 中选出一个与 target_aspect 完全一致的最大矩形。

    target_aspect = 目标宽 / 目标高。
    当需要横向裁切时，根据 horizontal_strategy 决定裁哪一侧。
    当需要纵向裁切时，根据 vertical_anchor 决定裁上、下或上下均分。
    """
    if source_rect.is_empty or source_rect.is_infinite:
        raise ValueError(f"无效源页面矩形：{source_rect}")
    if target_aspect <= 0:
        raise ValueError("target_aspect 必须大于 0")

    src_w = source_rect.width
    src_h = source_rect.height
    src_aspect = src_w / src_h
    eps = 1e-9

    if abs(src_aspect - target_aspect) <= eps:
        return fitz.Rect(source_rect)

    if src_aspect > target_aspect:
        # 源页面相对更宽：裁左右。
        clip_w = src_h * target_aspect
        removed_w = src_w - clip_w

        if horizontal_strategy == CROP_RIGHT:
            # 保留左侧，裁掉右侧。
            x0 = source_rect.x0
        elif horizontal_strategy == CROP_LEFT:
            # 保留右侧，裁掉左侧。
            x0 = source_rect.x0 + removed_w
        elif horizontal_strategy == CROP_CENTER:
            x0 = source_rect.x0 + removed_w / 2.0
        else:
            raise ValueError(f"未知 horizontal_strategy: {horizontal_strategy}")

        clip = fitz.Rect(x0, source_rect.y0, x0 + clip_w, source_rect.y1)
    else:
        # 源页面相对更高或更窄：裁上下。
        clip_h = src_w / target_aspect
        removed_h = src_h - clip_h

        if vertical_anchor == VERTICAL_TOP:
            y0 = source_rect.y0
        elif vertical_anchor == VERTICAL_BOTTOM:
            y0 = source_rect.y0 + removed_h
        elif vertical_anchor == VERTICAL_CENTER:
            y0 = source_rect.y0 + removed_h / 2.0
        else:
            raise ValueError(f"未知 vertical_anchor: {vertical_anchor}")

        clip = fitz.Rect(source_rect.x0, y0, source_rect.x1, y0 + clip_h)

    # 防止浮点误差使矩形越出源页。
    clip = clip & source_rect
    if clip.is_empty:
        raise RuntimeError("计算得到的裁切区域为空")
    return clip


def fit_rect_contain(source_rect: fitz.Rect, target_rect: fitz.Rect) -> fitz.Rect:
    """保持比例，把完整 source_rect 放入 target_rect，允许留白，不裁内容。"""
    scale = min(
        target_rect.width / source_rect.width,
        target_rect.height / source_rect.height,
    )
    width = source_rect.width * scale
    height = source_rect.height * scale
    x0 = target_rect.x0 + (target_rect.width - width) / 2.0
    y0 = target_rect.y0 + (target_rect.height - height) / 2.0
    return fitz.Rect(x0, y0, x0 + width, y0 + height)


def crop_description(strategy: str) -> str:
    if strategy == CROP_RIGHT:
        return "裁右侧 / 保留左侧书脊"
    if strategy == CROP_LEFT:
        return "裁左侧 / 保留右侧书脊"
    return "左右居中裁切"


def create_print_ready_pdf(
    input_path: str,
    output_path: str,
    trim_w_mm: float,
    trim_h_mm: float,
    bleed_mm: float = 3.0,
    layout_mode: str = LAYOUT_RIGHT_FIRST,
    resize_mode: str = RESIZE_COVER,
    bleed_mode: str = BLEED_ZOOM,
    center_pages: Optional[Iterable[int]] = None,
    vertical_anchor: str = VERTICAL_CENTER,
    password: Optional[str] = None,
    overwrite: bool = True,
    verbose: bool = True,
) -> None:
    """
    创建适合印刷尺寸的 PDF。

    参数说明：
    - trim_w_mm / trim_h_mm：裁切后的成品尺寸。
    - bleed_mm：四周出血。
    - layout_mode：
        right_first：PDF 第 1 页为右手页；奇数页裁右、偶数页裁左。
        left_first ：PDF 第 1 页为左手页；奇数页裁左、偶数页裁右。
        single     ：所有页居中裁切。
    - resize_mode：
        cover   ：铺满 TrimBox，必要时按 layout_mode 裁切。
        contain ：完整页面缩放进 TrimBox，不裁内容，可能留白。
    - bleed_mode：
        zoom  ：先在 MediaBox 放一层放大背景，再把清晰主体放进 TrimBox。
                背景层只用于制造出血，主体内容不会被出血吃掉。
        white ：不生成图像出血，出血区域保持白色。
    - center_pages：强制居中处理的 1-based 页码，如 {1} 可让封面居中，
                    但不改变后续奇偶页方向。
    - vertical_anchor：当比例差导致上下裁切时，选择 top / center / bottom。
    """
    validate_positive("trim_w_mm", trim_w_mm)
    validate_positive("trim_h_mm", trim_h_mm)
    validate_positive("bleed_mm", bleed_mm, allow_zero=True)

    if layout_mode not in VALID_LAYOUTS:
        raise ValueError(f"layout_mode 必须是 {sorted(VALID_LAYOUTS)}")
    if resize_mode not in VALID_RESIZE_MODES:
        raise ValueError(f"resize_mode 必须是 {sorted(VALID_RESIZE_MODES)}")
    if bleed_mode not in VALID_BLEED_MODES:
        raise ValueError(f"bleed_mode 必须是 {sorted(VALID_BLEED_MODES)}")
    if vertical_anchor not in VALID_VERTICAL_ANCHORS:
        raise ValueError(
            f"vertical_anchor 必须是 {sorted(VALID_VERTICAL_ANCHORS)}"
        )

    input_abs = os.path.abspath(os.path.expanduser(input_path))
    output_abs = os.path.abspath(os.path.expanduser(output_path))

    if not os.path.isfile(input_abs):
        raise FileNotFoundError(f"找不到输入文件：{input_abs}")
    if input_abs == output_abs:
        raise ValueError("输出路径不能与输入路径相同")

    output_dir = os.path.dirname(output_abs)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_abs):
        if not overwrite:
            raise FileExistsError(f"输出文件已存在：{output_abs}")
        os.remove(output_abs)

    center_page_set = {int(p) for p in (center_pages or [])}
    if any(p <= 0 for p in center_page_set):
        raise ValueError("center_pages 中的页码必须从 1 开始")

    trim_w_pt = mm_to_pt(trim_w_mm)
    trim_h_pt = mm_to_pt(trim_h_mm)
    bleed_pt = mm_to_pt(bleed_mm)

    media_w_pt = trim_w_pt + 2.0 * bleed_pt
    media_h_pt = trim_h_pt + 2.0 * bleed_pt

    doc_in = None
    doc_out = None

    try:
        doc_in = fitz.open(input_abs)
        if not doc_in.is_pdf:
            raise ValueError("输入文件必须是 PDF")

        if doc_in.needs_pass:
            if not password:
                raise PermissionError("输入 PDF 已加密，请提供 password")
            if doc_in.authenticate(password) <= 0:
                raise PermissionError("PDF 密码错误")

        if doc_in.page_count == 0:
            raise ValueError("输入 PDF 没有页面")

        doc_out = fitz.open()

        if verbose:
            print("=" * 72)
            print(f"输入文件：{input_abs}")
            print(f"输出文件：{output_abs}")
            print(f"页数：{doc_in.page_count}")
            print(f"成品尺寸：{trim_w_mm:.3f} × {trim_h_mm:.3f} mm")
            print(f"出血：四周 {bleed_mm:.3f} mm")
            print(
                f"MediaBox：{trim_w_mm + 2 * bleed_mm:.3f} × "
                f"{trim_h_mm + 2 * bleed_mm:.3f} mm"
            )
            print(f"页面模式：{layout_mode}")
            print(f"缩放模式：{resize_mode}")
            print(f"出血生成：{bleed_mode}")
            if center_page_set:
                print(f"强制居中页：{sorted(center_page_set)}")
            print("=" * 72)

        for page_index in range(doc_in.page_count):
            page_in = doc_in.load_page(page_index)

            # page.rect 是视觉页面矩形，已考虑页面旋转。
            source_rect = fitz.Rect(page_in.rect)
            if source_rect.is_empty or source_rect.width <= 0 or source_rect.height <= 0:
                raise ValueError(f"第 {page_index + 1} 页尺寸无效：{source_rect}")

            crop_strategy = get_horizontal_crop_strategy(
                page_index=page_index,
                layout_mode=layout_mode,
                center_pages=center_page_set,
            )

            page_out = doc_out.new_page(width=media_w_pt, height=media_h_pt)

            # PyMuPDF 写入页面尺寸时会按 PDF 浮点精度保存，可能产生极小舍入差。
            # 因此页面框必须以新页面实际返回的 MediaBox / rect 为准，
            # 不能直接复用理论计算值，否则可能出现“BleedBox not in MediaBox”。
            page_media_rect = fitz.Rect(page_out.rect)
            page_trim_rect = fitz.Rect(
                page_media_rect.x0 + bleed_pt,
                page_media_rect.y0 + bleed_pt,
                page_media_rect.x1 - bleed_pt,
                page_media_rect.y1 - bleed_pt,
            )
            page_trim_aspect = page_trim_rect.width / page_trim_rect.height
            page_media_aspect = page_media_rect.width / page_media_rect.height

            # 新页面默认 CropBox 为整个物理页面。
            page_out.set_trimbox(page_trim_rect)
            page_out.set_bleedbox(page_out.mediabox)

            if resize_mode == RESIZE_COVER:
                # 主体裁切窗口严格按 TrimBox 的宽高比计算。
                main_clip = crop_rect_to_aspect(
                    source_rect=source_rect,
                    target_aspect=page_trim_aspect,
                    horizontal_strategy=crop_strategy,
                    vertical_anchor=vertical_anchor,
                )
                main_target = page_trim_rect
            else:
                # contain：不裁源页，只缩放到 TrimBox 内。
                main_clip = source_rect
                main_target = fit_rect_contain(source_rect, page_trim_rect)

            if bleed_mm > 0 and bleed_mode == BLEED_ZOOM:
                # 背景仅用于填充出血区。
                # 再按 MediaBox 比例从主裁切区中取一个窗口，使其完全铺满 MediaBox。
                background_clip = crop_rect_to_aspect(
                    source_rect=main_clip,
                    target_aspect=page_media_aspect,
                    horizontal_strategy=CROP_CENTER,
                    vertical_anchor=vertical_anchor,
                )
                page_out.show_pdf_page(
                    page_media_rect,
                    doc_in,
                    page_index,
                    clip=background_clip,
                    keep_proportion=True,
                    overlay=False,
                )

            # 主体内容始终精确落在 TrimBox 内，避免出血侵占成品内容。
            page_out.show_pdf_page(
                main_target,
                doc_in,
                page_index,
                clip=main_clip,
                keep_proportion=True,
                overlay=True,
            )

            if verbose:
                left_crop_mm = max(0.0, main_clip.x0 - source_rect.x0) * PT2MM
                right_crop_mm = max(0.0, source_rect.x1 - main_clip.x1) * PT2MM
                top_crop_mm = max(0.0, main_clip.y0 - source_rect.y0) * PT2MM
                bottom_crop_mm = max(0.0, source_rect.y1 - main_clip.y1) * PT2MM

                print(
                    f"[{page_index + 1:>4}/{doc_in.page_count}] "
                    f"原页 {source_rect.width * PT2MM:.2f} × "
                    f"{source_rect.height * PT2MM:.2f} mm | "
                    f"{crop_description(crop_strategy)} | "
                    f"源页裁切 L={left_crop_mm:.2f}, R={right_crop_mm:.2f}, "
                    f"T={top_crop_mm:.2f}, B={bottom_crop_mm:.2f} mm"
                )

        # 保留基础元数据。
        try:
            metadata = dict(doc_in.metadata or {})
            if metadata:
                doc_out.set_metadata(metadata)
        except Exception as exc:
            if verbose:
                print(f"警告：未能复制 PDF 元数据：{exc}")

        # 页面数量和顺序未变，复制简单目录（标题、层级、页码）。
        try:
            toc = doc_in.get_toc(simple=True)
            if toc:
                doc_out.set_toc(toc)
        except Exception as exc:
            if verbose:
                print(f"警告：未能复制目录：{exc}")

        save_kwargs = {
            "garbage": 4,
            "deflate": True,
        }
        # 新版 PyMuPDF 可把小对象放进 object streams，通常能进一步减小体积。
        try:
            doc_out.save(output_abs, use_objstms=1, **save_kwargs)
        except TypeError:
            # 兼容不支持 use_objstms 的旧版本。
            doc_out.save(output_abs, **save_kwargs)

        if verbose:
            print("=" * 72)
            print(f"处理完成：{output_abs}")
            print("提示：show_pdf_page 不会复制批注、表单控件和可点击链接。")
            print("=" * 72)

    except Exception:
        # 避免保留损坏或未完成的输出文件。
        if os.path.exists(output_abs):
            try:
                os.remove(output_abs)
            except OSError:
                pass
        raise
    finally:
        if doc_out is not None:
            doc_out.close()
        if doc_in is not None:
            doc_in.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按书籍奇偶页方向裁切 PDF，并生成 TrimBox / BleedBox。"
    )
    parser.add_argument("input_pdf", help="输入 PDF 路径")
    parser.add_argument("output_pdf", help="输出 PDF 路径")
    parser.add_argument("--trim-w", type=float, required=True, help="成品宽度，毫米")
    parser.add_argument("--trim-h", type=float, required=True, help="成品高度，毫米")
    parser.add_argument("--bleed", type=float, default=3.0, help="四周出血，毫米")
    parser.add_argument(
        "--layout",
        choices=sorted(VALID_LAYOUTS),
        default=LAYOUT_RIGHT_FIRST,
        help=(
            "right_first=第1页右手页；left_first=第1页左手页；"
            "single=全部居中"
        ),
    )
    parser.add_argument(
        "--resize",
        choices=sorted(VALID_RESIZE_MODES),
        default=RESIZE_COVER,
        help="cover=铺满并裁切；contain=完整缩放并允许留白",
    )
    parser.add_argument(
        "--bleed-mode",
        choices=sorted(VALID_BLEED_MODES),
        default=BLEED_ZOOM,
        help="zoom=放大背景生成出血；white=出血区留白",
    )
    parser.add_argument(
        "--center-pages",
        default="",
        help="强制居中的 1-based 页码，例如 1 或 1,3,5 或 1-3,8",
    )
    parser.add_argument(
        "--vertical-anchor",
        choices=sorted(VALID_VERTICAL_ANCHORS),
        default=VERTICAL_CENTER,
        help="需要上下裁切时保留顶部、居中或底部",
    )
    parser.add_argument("--password", default=None, help="加密 PDF 密码")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="输出文件已存在时停止，而不是覆盖",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少控制台输出",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        center_pages = parse_page_numbers(args.center_pages)
        create_print_ready_pdf(
            input_path=args.input_pdf,
            output_path=args.output_pdf,
            trim_w_mm=args.trim_w,
            trim_h_mm=args.trim_h,
            bleed_mm=args.bleed,
            layout_mode=args.layout,
            resize_mode=args.resize,
            bleed_mode=args.bleed_mode,
            center_pages=center_pages,
            vertical_anchor=args.vertical_anchor,
            password=args.password,
            overwrite=not args.no_overwrite,
            verbose=not args.quiet,
        )
        return 0
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())