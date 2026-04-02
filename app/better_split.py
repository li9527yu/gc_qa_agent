import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from app.config import ROOT_PATH

class MarkdownTableWithSmartTextSplitter:
    """
    一个智能的 Markdown 文本分割器。

    该分割器能够识别并完整保留 Markdown 表格，避免将其从中间拆分。
    它会将整个表格及其前后相邻的非表格行文本（作为上下文）合并成一个单独的 chunk。
    对于普通文本，它会使用 LangChain 的 RecursiveCharacterTextSplitter 进行切分。

    Attributes:
        text_splitter: 用于切分普通文本的 RecursiveCharacterTextSplitter 实例。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化 MarkdownTableWithSmartTextSplitter。

        Args:
            chunk_size (int): 普通文本块的最大长度。
            chunk_overlap (int): 普通文本块之间的重叠长度。
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "。", "，", ","]
        )

    def split_text(self, text: str) -> List[str]:
        """
        切分给定的文本。

        Args:
            text (str): 需要切分的原始 Markdown 文本。

        Returns:
            List[str]: 切分后的文本块列表。
        """
        lines = text.splitlines()
        chunks = []
        buffer = []
        i = 0

        while i < len(lines):
            line = lines[i]
            # 使用正则表达式检测 Markdown 表格行
            if "|" in line and re.match(r"^\s*\|.*\|\s*$", line):
                # 1. 首先处理缓冲区中的普通文本
                if buffer:
                    plain_text = "\n".join(buffer).strip()
                    if plain_text:
                        chunks.extend(self.text_splitter.split_text(plain_text))
                    buffer = [] # 清空缓冲区

                # 2. 收集完整的表格部分
                table_lines = [line]
                i += 1
                # 持续收集直到行不再包含 '|'
                while i < len(lines) and "|" in lines[i] and re.match(r"^\s*\|.*\|\s*$", lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                table_text = "\n".join(table_lines).strip()

                # 3. 寻找表格的上下文（前后的非空、非表格行）
                prev_text = self._find_context_line(lines, i - len(table_lines) - 1, -1)
                next_text = self._find_context_line(lines, i, 1)

                # 4. 将上下文和表格合并为一个 chunk
                full_chunk = "\n".join(filter(None, [prev_text, table_text, next_text]))
                chunks.append(full_chunk)
            else:
                buffer.append(line)
                i += 1

        # 5. 处理循环结束后剩余在缓冲区中的文本
        if buffer:
            plain_text = "\n".join(buffer).strip()
            if plain_text:
                chunks.extend(self.text_splitter.split_text(plain_text))

        return chunks

    def _find_context_line(self, lines: List[str], start_index: int, step: int) -> str:
        """在指定方向上查找第一个非空、非表格的行作为上下文。"""
        j = start_index
        while 0 <= j < len(lines):
            line = lines[j].strip()
            if line and "|" not in line:
                return line
            j += step
        return ""


if __name__ == '__main__':
    # --- 以下是测试代码 ---
    # 仅当直接运行此文件时执行，import 时不执行
    
    # 示例 Markdown 文本
    sample_text = open(ROOT_PATH+"/app/dataset/工程造价/09测量测绘费用标准.md", encoding="utf-8").read()

    
    # 实例化分割器
    splitter = MarkdownTableWithSmartTextSplitter(chunk_size=500, chunk_overlap=100)
    
    # 执行切分
    chunks = splitter.split_text(sample_text)
    
    # 打印结果
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---\n{chunk}\n")