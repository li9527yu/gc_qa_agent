# app/utils/markdown_converter.py

from pathlib import Path
from markdownify import markdownify as md
import os

def convert_text_to_md(text: str, save_path: Path):
    """
    将纯文本或 HTML 文本保存为 Markdown 文件
    """
    # 转换成 Markdown 格式
    markdown_text = md(text, heading_style="ATX")  # 支持 HTML 转 Markdown

    # 确保保存目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存 Markdown 文件
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
