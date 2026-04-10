import re
from tqdm import tqdm
import PyPDF2
import pandas as pd
import os
import json
import hashlib
from datetime import datetime
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .better_split import MarkdownTableWithSmartTextSplitter
from .better_split2 import MarkdownTableWithSmartTextSplitterV2
import logging

logger = logging.getLogger(__name__)

def compute_folder_hash(folder_path: str) -> str:
    """计算文件夹内所有文件的哈希值"""
    files_info = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            mtime = os.path.getmtime(file_path)
            size = os.path.getsize(file_path)
            files_info.append(f"{file_path}:{mtime}:{size}")
    files_info.sort()  # 排序以确保相同内容产生相同的哈希值
    return hashlib.md5("".join(files_info).encode()).hexdigest()

class Reader:
    def __init__(self, corpus_path: str,cache_dir: str = None):
        # 如果没有指定 cache_dir，使用默认路径
        if cache_dir is None:
            self.cache_dir = os.path.join(os.path.expanduser("~"), ".easy_rag_cache")
        else:
            self.cache_dir = cache_dir
        
        os.makedirs(self.cache_dir, exist_ok=True)

        # 如果是目录，尝试使用缓存
        if os.path.isdir(corpus_path):
            folder_hash = compute_folder_hash(corpus_path)
            cache_file = os.path.join(self.cache_dir, f"corpus_cache_{folder_hash}.json")
            
            # 检查是否存在缓存
            if os.path.exists(cache_file):
                logger.info("正在从缓存加载语料库...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.corpus = json.load(f)
                logger.info(f"已从缓存加载 {len(self.corpus)} 段文本")
                
                # 验证缓存的有效性
                if self._validate_cache(corpus_path, self.corpus):
                    return  # 缓存有效，直接返回
                else:
                    logger.warning("缓存验证失败，重新处理文档...")
                    # 删除无效缓存
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
            
            # 如果没有缓存或缓存无效，则处理文档
            logger.info("未找到缓存或缓存无效，开始处理文档...")
            self.corpus = []
            self._process_corpus(corpus_path)
            
            # 保存缓存
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.corpus, f, ensure_ascii=False, indent=2)
            logger.info(f"已将 {len(self.corpus)} 段文本缓存到 {cache_file}")
        else:
            # 处理单个文件
            self.corpus = []
            self._process_corpus(corpus_path)

    def _validate_cache(self, corpus_path: str, cached_corpus: list) -> bool:
        """
        验证缓存的有效性
        
        Args:
            corpus_path: 语料库路径
            cached_corpus: 缓存的语料库
            
        Returns:
            bool: 缓存是否有效
        """
        try:
            # 检查缓存是否为空
            if not cached_corpus:
                logger.warning("缓存为空")
                return False
            
            # 检查缓存中的文档格式是否正确
            for doc in cached_corpus[:5]:  # 只检查前5个文档
                if not isinstance(doc, dict) or 'page_content' not in doc:
                    logger.warning("缓存中的文档格式不正确")
                    return False
            
            # 检查文件数量是否合理（可选）
            if os.path.isdir(corpus_path):
                file_count = sum(1 for _ in os.walk(corpus_path))
                if len(cached_corpus) < file_count * 0.5:  # 如果缓存的文档数少于文件数的一半，可能有问题
                    logger.warning(f"缓存文档数 ({len(cached_corpus)}) 与文件数 ({file_count}) 不匹配")
                    return False
            
            logger.info("缓存验证通过")
            return True
            
        except Exception as e:
            logger.error(f"缓存验证时出错: {e}")
            return False

    def _process_corpus(self, corpus_path: str):
        if os.path.isdir(corpus_path):
            self.corpus = self.read_folder(corpus_path)
        elif corpus_path.endswith('.pdf'):
            self.corpus = self._normalize_chunks(
                self.extract_pdf_page_text(corpus_path),
                os.path.basename(corpus_path)
            )
        elif corpus_path.endswith('.md'):
            with open(corpus_path, 'r', encoding='utf-8') as file:
                text = file.read()
            source = os.path.basename(corpus_path)
            chunks = self.split_by_markdown_tableV2(text, source)
            self.corpus = self._assign_global_chunk_ids(chunks, source)
        elif corpus_path.endswith('.txt'):
            with open(corpus_path, 'r', encoding='utf-8') as file:
                text = file.read()
            source = os.path.basename(corpus_path)
            chunks = self.split_by_recursive_character_splitter(text)
            self.corpus = self._normalize_chunks(chunks, source)
        elif 'Multi-CPR' in corpus_path:
            self.corpus = self._normalize_chunks(
                self.extract_multiCPR_text(corpus_path),
                os.path.basename(corpus_path)
            )
        elif 'train_data_chunk' in corpus_path:
            self.corpus = self._normalize_chunks(
                self.extract_text0328(corpus_path),
                os.path.basename(corpus_path)
            )
        else:
            self.corpus = self.extract_my_file(corpus_path)
    
    def _normalize_chunks(self, chunks: List[str], source_file: str) -> List[dict]:
        """将纯文本列表转换为统一格式的chunk字典列表"""
        result = []
        # 提取文件类型
        file_ext = os.path.splitext(source_file)[1].lower().lstrip('.')
        if file_ext not in ['md', 'txt', 'pdf']:
            file_ext = 'unknown'
        
        for i, content in enumerate(chunks):
            chunk_hash = hashlib.md5(
                f"{source_file}:{i}:{content[:100]}".encode()
            ).hexdigest()[:16]
            
            # 生成 chunk_id（仅用于内部索引，不入 Milvus）
            chunk_id = f"{source_file}_{chunk_hash}"
            
            result.append({
                'page_content': content if isinstance(content, str) else str(content),
                'metadata': {
                    # === 内部使用字段（不入 Milvus）===
                    'chunk_id': chunk_id,
                    'source_file': source_file,  # kb_manager 依赖
                    
                    # === 存入 Milvus 的字段 ===
                    'source': source_file,       # 文件名（前端展示）
                    'file_type': file_ext,       # 文件类型 (md/pdf/txt)
                    'chunk_index': i,            # 文件内 chunk 序号
                    'upload_time': datetime.now().isoformat(),  # 上传时间
                    'content_type': 'text',      # 内容类型
                }
            })
        return result
    
    def _assign_global_chunk_ids(self, chunks: List[dict], source_file: str) -> List[dict]:
        """为已有的chunk字典分配全局唯一的chunk_id"""
        result = []
        # 提取文件类型
        file_ext = os.path.splitext(source_file)[1].lower().lstrip('.')
        if file_ext not in ['md', 'txt', 'pdf']:
            file_ext = 'unknown'
        
        # 获取内容类型（如果已有）
        first_chunk_meta = chunks[0].get('metadata', {}) if chunks and isinstance(chunks[0], dict) else {}
        inherited_content_type = first_chunk_meta.get('content_type', 'text')
        
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                content = chunk.get('page_content', '')
                metadata = chunk.get('metadata', {})
                # 保留原始内容类型（text/table）
                content_type = metadata.get('content_type', inherited_content_type)
            else:
                content = str(chunk)
                metadata = {}
                content_type = inherited_content_type
            
            # 生成全局唯一ID
            chunk_hash = hashlib.md5(
                f"{source_file}:{i}:{content[:100]}".encode()
            ).hexdigest()[:16]
            
            chunk_id = f"{source_file}_{chunk_hash}"
            
            metadata.update({
                # === 内部使用字段（不入 Milvus）===
                'chunk_id': chunk_id,
                'source_file': source_file,  # kb_manager 依赖
                
                # === 存入 Milvus 的字段 ===
                'source': source_file,       # 文件名
                'file_type': file_ext,       # 文件类型
                'chunk_index': i,            # 文件内 chunk 序号
                'upload_time': datetime.now().isoformat(),
                'content_type': content_type,  # 保留原始内容类型
            })
            
            result.append({
                'page_content': content,
                'metadata': metadata
            })
        return result

    def read_folder(self, folder_path, verbose=False):
        all_chunks = []
        total_files = 0
        # 预收集所有文件路径，才能用 tqdm 显示准确进度
        all_file_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                all_file_paths.append(os.path.join(root, file))
 
        for filepath in tqdm(all_file_paths, desc="Processing files"):
            file = os.path.basename(filepath)
            chunks = []  # 默认空列表
            if file.endswith('.pdf'):
                raw_chunks = self.extract_pdf_page_text(filepath)
                chunks = self._normalize_chunks(raw_chunks, file)
            elif file.endswith('.md'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                raw_chunks = self.split_by_markdown_tableV2(text, file)
                chunks = self._assign_global_chunk_ids(raw_chunks, file)
            elif file.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                raw_chunks = self.split_by_recursive_character_splitter(text)
                chunks = self._normalize_chunks(raw_chunks, file)
            else:
                if verbose:
                    logger.info(f"[跳过] 不支持的文件格式: {file}")
                continue
            all_chunks.extend(chunks)
            total_files += 1
            if verbose:
                logger.info(f"[已处理] {file} → {len(chunks)} 段")        
        if verbose:
            logger.info(f"[总计] 处理文档数：{len(all_chunks)} 段文本")
        return all_chunks

    def extract_pdf_page_text(self, filepath, max_len=256, overlap_len=100):
        page_content  = []
        with open(filepath, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in tqdm(pdf_reader.pages, desc='解析PDF文件...'):
                page_text = page.extract_text().strip()
                raw_text = [text.strip() for text in page_text.split('\n')]
                new_text = '\n'.join(raw_text)
                new_text = re.sub(r'\n\d{2,3}\s?', '\n', new_text)
                if len(new_text) > 10 and '..............' not in new_text:
                    page_content.append(new_text)

        cleaned_chunks = []
        i = 0
        # 暴力将整个pdf当做一个字符串，然后按照固定大小的滑动窗口切割
        all_str = ''.join(page_content)
        all_str = all_str.replace('\n', '')
        while i < len(all_str):
            cur_s = all_str[i:i+max_len]
            if len(cur_s) > 10:
                cleaned_chunks.append(cur_s)
            i += (max_len - overlap_len)
        print('cleaned_chunks',len(cleaned_chunks))

        return cleaned_chunks

    def extract_multiCPR_text(self, filepath):

        corpus = pd.read_csv(filepath, sep='\t', header=None)
        corpus.columns = ['pid', 'passage']
        return corpus.passage.values.tolist()


    def extract_my_file(self, filepath):

        raise NotImplementedError
    
    # 2：基于Markdown表格的智能文本分割器V2: 
    def split_by_markdown_tableV2(self,text,source,chunk_size=500, chunk_overlap=100):
        # splitter = MarkdownTableWithSmartTextSplitterV2(chunk_size=1024, chunk_overlap=200)
        splitter = MarkdownTableWithSmartTextSplitterV2(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text_to_documents(text,source)
    # 1：基于Markdown表格的智能文本分割器
    def split_by_markdown_table(self, text):
        splitter = MarkdownTableWithSmartTextSplitter(chunk_size=500, chunk_overlap=100)
        return splitter.split_text(text)

    # 2：基于递归字符分割器（langchain）
    def split_by_recursive_character_splitter(self,text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        return text_splitter.split_text(text)
    
    def extract_text0328(self, filepath):
        page_content  = []
        with open(filepath, 'rb') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    page_content.append(line)
        print('page_content',len(page_content),len(page_content[0]))
        return page_content