import fitz  # PyMuPDF
import argparse
import sys

# 标准纸张尺寸（单位：毫米）
PAPER_SIZES_MM = {
    "a4": (210.0, 297.0),
    "a5": (148.0, 210.0),
    "letter": (215.9, 279.4),
}

# 毫米转磅的转换系数 (1英寸 = 25.4毫米 = 72磅)
MM2PT = 72.0 / 25.4

def main():
    parser = argparse.ArgumentParser(
        description="检查PDF图片在目标印刷尺寸下的有效DPI，添加蒙版和标注"
    )
    parser.add_argument("--input", default="C:\\Users\\HASEE\\Downloads\\龙与香辛料719打印测试\\龙与香辛料719_a5.pdf", help="输入PDF文件路径")
    parser.add_argument("--output", default="C:\\Users\\HASEE\\Downloads\\龙与香辛料719打印测试\\719_a5_output.pdf", help="输出PDF文件路径（带标注）") 
    
    # 需求1：将 required=True 移除，增加 default="auto"，并在 choices 中加入 "auto"
    parser.add_argument(
        "--size", "-s",
        default="auto",
        choices=["a4", "a5", "letter", "auto"],
        help="目标印刷尺寸（a4/a5/letter）。不填或设为auto则使用PDF页面自身原始尺寸。"
    )
    parser.add_argument(
        "--dpi", "-d",
        type=float,
        default=300,
        help="最低DPI阈值（默认：300）"
    )
    args = parser.parse_args()

    # 如果不是 auto，提前校验输入的纸张尺寸是否在字典中
    if args.size.lower() != "auto" and args.size.lower() not in PAPER_SIZES_MM:
        print(f"错误：不支持的纸张尺寸 '{args.size}'")
        sys.exit(1)
        
    min_dpi = args.dpi

    # 打开PDF文档
    try:
        doc = fitz.open(args.input)
    except Exception as e:
        print(f"无法打开PDF文件：{e}")
        sys.exit(1)

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 获取当前页面的原始尺寸（单位：磅）
        page_w = page.rect.width
        page_h = page.rect.height
        if page_w == 0 or page_h == 0:
            continue

        # 需求2：根据参数决定目标尺寸
        if args.size.lower() == "auto":
            # 默认：使用PDF自己的尺寸 (此时缩放比例为1)
            target_w, target_h = page_w, page_h
        else:
            # 指定尺寸：读取毫米尺寸，并转换为磅(Points)
            target_w_mm, target_h_mm = PAPER_SIZES_MM[args.size.lower()]
            target_w = target_w_mm * MM2PT
            target_h = target_h_mm * MM2PT

        # 模拟打印机的自动旋转行为 (短边对短边，长边对长边)
        page_min, page_max = min(page_w, page_h), max(page_w, page_h)
        target_min, target_max = min(target_w, target_h), max(target_w, target_h)
        
        # 计算缩放到目标纸张的适应比例
        scale_min_side = target_min / page_min
        scale_max_side = target_max / page_max
        s = min(scale_min_side, scale_max_side)  # 适应模式

        # 获取页面中的所有图片
        images = page.get_images(full=True)
        for img in images:
            xref = img[0]          # 图片对象引用
            pix_w = img[2]         # 原始像素宽度
            pix_h = img[3]         # 原始像素高度
            if pix_w == 0 or pix_h == 0:
                continue

            # 获取图片在页面上的定位矩形
            rects = page.get_image_rects(xref)
            if not rects:
                continue

            for rect in rects:
                img_w_pts = rect.width
                img_h_pts = rect.height
                if img_w_pts == 0 or img_h_pts == 0:
                    continue

                # 图片在页面上的原始放置DPI（不缩放时）
                place_dpi_x = pix_w / (img_w_pts / 72.0)
                place_dpi_y = pix_h / (img_h_pts / 72.0)

                # 考虑页面整体缩放后的有效DPI
                # 如果是 auto 模式，s=1.0，eff_dpi 就等于 place_dpi
                eff_dpi_x = place_dpi_x / s
                eff_dpi_y = place_dpi_y / s
                effective_dpi = min(eff_dpi_x, eff_dpi_y)

                # 判断是否满足要求
                meets = effective_dpi >= min_dpi
                # 蒙版颜色：绿色满足，红色不满足
                color = (0, 1, 0) if meets else (1, 0, 0)

                # 绘制半透明蒙版
                page.draw_rect(
                    rect,
                    fill=color,
                    fill_opacity=0.3,
                    color=None,
                    width=0
                )

                # 在图片左上角添加DPI标注
                label = f"DPI: {effective_dpi:.0f}"
                
                # 限制文本框宽度，防止在极小图片上越界
                box_width = min(80, rect.width)
                box_height = min(14, rect.height)
                text_rect = fitz.Rect(
                    rect.x0, rect.y0,
                    rect.x0 + box_width, rect.y0 + box_height
                )
                
                # 先画半透明背景框
                page.draw_rect(
                    text_rect,
                    fill=(1, 1, 1),
                    fill_opacity=0.8,
                    color=None,
                    width=0
                )
                
                # 再插入文字
                page.insert_textbox(
                    text_rect,
                    label,
                    fontsize=8,
                    fontname="helv",
                    color=(0, 0, 0),
                    align=0
                )

    # 保存结果
    doc.save(args.output, deflate=True)
    doc.close()
    print(f"处理完成，输出文件：{args.output}")

if __name__ == "__main__":
    main()