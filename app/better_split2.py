from langchain.schema import Document
import re
import os
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional
from app.config import ROOT_PATH

class MarkdownTableWithSmartTextSplitterV2:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "。", "，", ","]
        )

    def split_text_to_documents(self, text: str, source: Optional[str] = None) -> List[Document]:
        lines = text.splitlines()
        documents = []
        buffer = []
        i = 0
        chunk_index = 0
        
        # 提取文件类型
        source = source or "unknown"
        file_ext = os.path.splitext(source)[1].lower().lstrip('.')
        if file_ext not in ['md', 'txt', 'pdf']:
            file_ext = 'unknown'
        upload_time = datetime.now().isoformat()

        while i < len(lines):
            line = lines[i]
            if "|" in line and re.match(r"^\s*\|.*\|\s*$", line):
                # flush text buffer
                if buffer:
                    plain_text = "\n".join(buffer).strip()
                    if plain_text:
                        text_chunks = self.text_splitter.split_text(plain_text)
                        for chunk in text_chunks:
                            documents.append({
                                'page_content': chunk,
                                'metadata': {
                                    # 临时 chunk_index，会被 _assign_global_chunk_ids 覆盖
                                    'chunk_index': chunk_index,
                                    'source': source,
                                    'file_type': file_ext,
                                    'upload_time': upload_time,
                                    'content_type': 'text',
                                    # 保留旧字段兼容性
                                    'type': 'text',
                                    'chunk_id': chunk_index,
                                }
                            })
                            chunk_index += 1
                    buffer = []

                # collect table block
                table_lines = [line]
                i += 1
                while i < len(lines) and "|" in lines[i] and re.match(r"^\s*\|.*\|\s*$", lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                table_text = "\n".join(table_lines).strip()

                # collect context
                prev_text = self._find_context_line(lines, i - len(table_lines) - 1, -1)
                next_text = self._find_context_line(lines, i, 1)
                full_chunk = "\n".join(filter(None, [prev_text, table_text, next_text]))

                documents.append({
                    'page_content': full_chunk,
                    'metadata': {
                        'chunk_index': chunk_index,
                        'source': source,
                        'file_type': file_ext,
                        'upload_time': upload_time,
                        'content_type': 'table',
                        # 保留旧字段兼容性
                        'type': 'table',
                        'chunk_id': chunk_index,
                    }
                })
                chunk_index += 1
            else:
                buffer.append(line)
                i += 1

        # flush remaining text
        if buffer:
            plain_text = "\n".join(buffer).strip()
            if plain_text:
                text_chunks = self.text_splitter.split_text(plain_text)
                for chunk in text_chunks:
                    documents.append({
                        'page_content': chunk,
                        'metadata': {
                            'chunk_index': chunk_index,
                            'source': source,
                            'file_type': file_ext,
                            'upload_time': upload_time,
                            'content_type': 'text',
                            # 保留旧字段兼容性
                            'type': 'text',
                            'chunk_id': chunk_index,
                        }
                    })
                    chunk_index += 1

        return documents

    def _find_context_line(self, lines: List[str], start_index: int, step: int) -> str:
        j = start_index
        while 0 <= j < len(lines):
            line = lines[j].strip()
            if line and "|" not in line:
                return line
            j += step
        return ""


if __name__ == '__main__':
    import os
    from pathlib import Path

    # 示例 Markdown 文件路径
    file_path = ROOT_PATH+"/app/dataset/工程造价/09测量测绘费用标准.md"
    
    # 安全检查路径是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在：{file_path}")
        exit(1)

    # 读取 Markdown 文本
    with open(file_path, encoding="utf-8") as f:
        sample_text = f.read()

    # 提取文件名作为 source 元信息
    source_name = Path(file_path).name

    # 实例化分割器
    splitter = MarkdownTableWithSmartTextSplitterV2(chunk_size=500, chunk_overlap=100)

    # 执行切分并生成 Document 列表
    documents = splitter.split_text_to_documents(sample_text, source=source_name)
    serialized = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]
    # 打印每个 Document 内容及其元数据
    for i, doc in enumerate(documents):
        print(f"\n--- Chunk {i+1} ---")
        meta = doc.metadata
        print(f"[内容类型] {meta.get('content_type')} | [文件类型] {meta.get('file_type')} | "
              f"[序号] {meta.get('chunk_index')} | [来源] {meta.get('source')}")
        print(doc.page_content)
