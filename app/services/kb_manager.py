"""
知识库管理器 - 支持增量更新
管理文件索引、变更检测和增量更新协调
"""

import os
import json
import hashlib
import logging
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from datetime import datetime
from app.config import DATA_PATH, ROOT_PATH

logger = logging.getLogger(__name__)


class FileIndex:
    """单个文件的索引信息"""
    def __init__(self, file_hash: str, chunk_ids: List[str], doc_count: int, 
                 last_modified: str, file_size: int):
        self.file_hash = file_hash
        self.chunk_ids = chunk_ids  # 该文件产生的所有chunk的唯一ID
        self.doc_count = doc_count
        self.last_modified = last_modified
        self.file_size = file_size
    
    def to_dict(self) -> dict:
        return {
            "file_hash": self.file_hash,
            "chunk_ids": self.chunk_ids,
            "doc_count": self.doc_count,
            "last_modified": self.last_modified,
            "file_size": self.file_size
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FileIndex":
        return cls(
            file_hash=data["file_hash"],
            chunk_ids=data["chunk_ids"],
            doc_count=data["doc_count"],
            last_modified=data["last_modified"],
            file_size=data["file_size"]
        )


class KnowledgeBaseManager:
    """
    知识库管理器
    
    职责：
    1. 维护文件级索引（file_hash -> chunk_ids映射）
    2. 检测文件变更（新增、修改、删除）
    3. 协调增量更新操作
    4. 管理chunk的生命周期
    """
    
    def __init__(self, index_path: str = None, data_path: str = None):
        self.data_path = data_path or DATA_PATH
        self.index_path = index_path or os.path.join(ROOT_PATH, "app/dataset/kb_index.json")
        self.index: Dict[str, FileIndex] = {}
        self._chunk_to_file: Dict[str, str] = {}
        self._load_index()
    
    def _load_index(self):
        """加载索引文件"""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.index = {
                        k: FileIndex.from_dict(v) 
                        for k, v in data.get("files", {}).items()
                    }
                    self._rebuild_chunk_index()
                logger.info(f"已加载知识库索引，共 {len(self.index)} 个文件")
            except Exception as e:
                logger.error(f"加载索引失败: {e}，将创建新索引")
                self.index = {}
        else:
            logger.info("索引文件不存在，将创建新索引")
            self.index = {}
    
    def _save_index(self):
        """保存索引到文件"""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            data = {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "files": {k: v.to_dict() for k, v in self.index.items()}
            }
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
    
    def _rebuild_chunk_index(self):
        """重建chunk到文件的反向索引"""
        self._chunk_to_file = {}
        for filename, file_index in self.index.items():
            for chunk_id in file_index.chunk_ids:
                self._chunk_to_file[chunk_id] = filename
    
    @staticmethod
    def compute_file_hash(filepath: str) -> str:
        """计算文件内容的MD5哈希"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"计算文件哈希失败 {filepath}: {e}")
            return ""
    
    def get_current_files(self) -> Set[str]:
        """获取数据目录下当前所有的有效文件"""
        current_files = set()
        if not os.path.exists(self.data_path):
            return current_files
        
        for filename in os.listdir(self.data_path):
            filepath = os.path.join(self.data_path, filename)
            if os.path.isfile(filepath) and filename.endswith(('.md', '.txt', '.pdf')):
                current_files.add(filename)
        return current_files
    
    def detect_changes(self) -> Dict[str, List[str]]:
        """
        检测文件变更
        
        Returns:
            dict with keys: added, deleted, modified, unchanged
        """
        current_files = self.get_current_files()
        indexed_files = set(self.index.keys())
        
        added = list(current_files - indexed_files)
        deleted = list(indexed_files - current_files)
        modified = []
        unchanged = []
        
        for filename in current_files & indexed_files:
            filepath = os.path.join(self.data_path, filename)
            current_hash = self.compute_file_hash(filepath)
            
            if current_hash != self.index[filename].file_hash:
                modified.append(filename)
            else:
                unchanged.append(filename)
        
        return {
            "added": added,
            "deleted": deleted,
            "modified": modified,
            "unchanged": unchanged
        }
    
    def add_file_record(self, filename: str, chunks: List[dict]) -> List[str]:
        """添加文件记录"""
        filepath = os.path.join(self.data_path, filename)
        if not os.path.exists(filepath):
            logger.error(f"文件不存在: {filepath}")
            return []
        
        # 生成chunk IDs
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            chunk_hash = hashlib.md5(
                f"{filename}:{i}:{chunk.get('page_content', '')[:100]}".encode()
            ).hexdigest()[:16]
            chunk_id = f"{filename}_{chunk_hash}"
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            chunk['metadata']['chunk_id'] = chunk_id
            chunk['metadata']['source_file'] = filename
            chunk_ids.append(chunk_id)
        
        file_stat = os.stat(filepath)
        file_index = FileIndex(
            file_hash=self.compute_file_hash(filepath),
            chunk_ids=chunk_ids,
            doc_count=len(chunks),
            last_modified=datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            file_size=file_stat.st_size
        )
        
        self.index[filename] = file_index
        for chunk_id in chunk_ids:
            self._chunk_to_file[chunk_id] = filename
        
        self._save_index()
        logger.info(f"添加文件记录: {filename}, {len(chunks)} chunks")
        return chunk_ids
    
    def remove_file_record(self, filename: str) -> List[str]:
        """删除文件记录，返回关联的chunk_ids"""
        if filename not in self.index:
            return []
        
        chunk_ids = self.index[filename].chunk_ids
        del self.index[filename]
        
        for chunk_id in chunk_ids:
            if chunk_id in self._chunk_to_file:
                del self._chunk_to_file[chunk_id]
        
        self._save_index()
        logger.info(f"删除文件记录: {filename}, {len(chunk_ids)} chunks")
        return chunk_ids
    
    def update_file_record(self, filename: str, chunks: List[dict]) -> Tuple[List[str], List[str]]:
        """更新文件记录，返回(新chunk_ids, 旧chunk_ids)"""
        old_chunk_ids = self.remove_file_record(filename)
        new_chunk_ids = self.add_file_record(filename, chunks)
        return new_chunk_ids, old_chunk_ids
    
    def get_chunk_ids_by_file(self, filename: str) -> List[str]:
        """获取文件关联的所有chunk IDs"""
        if filename in self.index:
            return self.index[filename].chunk_ids
        return []
    
    def get_file_by_chunk_id(self, chunk_id: str) -> Optional[str]:
        """通过chunk ID查找源文件"""
        return self._chunk_to_file.get(chunk_id)
    
    def get_all_chunk_ids(self) -> Set[str]:
        """获取所有chunk IDs"""
        return set(self._chunk_to_file.keys())
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        total_chunks = sum(f.doc_count for f in self.index.values())
        total_size = sum(f.file_size for f in self.index.values())
        return {
            "total_files": len(self.index),
            "total_chunks": total_chunks,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "last_updated": max((f.last_modified for f in self.index.values()), default=None)
        }


# 全局单例
_kb_manager = None

def get_kb_manager() -> KnowledgeBaseManager:
    """获取知识库管理器单例"""
    global _kb_manager
    if _kb_manager is None:
        _kb_manager = KnowledgeBaseManager()
    return _kb_manager
