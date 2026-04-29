"""
增量检索器 - 支持知识库的增量更新
基于原Retriever扩展，添加增删改能力
"""

import os
import logging
import pickle
from typing import List, Dict, Optional
import jieba
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from pymilvus import connections, utility

from app.retriever import Retriever, preprocess_corpus, TextEmbedding
from app.services.kb_manager import get_kb_manager

logger = logging.getLogger(__name__)


class IncrementalRetriever(Retriever):
    """
    支持增量更新的检索器
    
    扩展功能：
    1. add_documents - 增量添加文档
    2. delete_documents - 按chunk_id删除文档（标记重建）
    3. update_documents - 更新文档（删除旧+添加新）
    4. incremental_update - 根据变更列表执行批量更新
    
    注意：删除操作会标记_need_milvus_rebuild，下次添加时重建Milvus
          BM25会立即更新，只有Milvus使用延迟重建策略
    """
    
    def __init__(self, *args, **kwargs):
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        self.kb_manager = get_kb_manager()
        self._need_milvus_rebuild = False  # 标记是否需要重建Milvus
        
    def add_documents(self, chunks: List[dict], source_file: str) -> bool:
        """
        增量添加文档
        
        Args:
            chunks: 文档chunks列表，每个chunk包含page_content和metadata
            source_file: 源文件名，用于记录
            
        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"开始增量添加文档: {source_file}, {len(chunks)} chunks")
            
            # 1. 为chunks生成唯一ID并记录
            chunk_ids = self.kb_manager.add_file_record(source_file, chunks)
            
            # 2. 更新语料库缓存
            self.corpus_dicts.extend(chunks)
            self.corpus.extend([doc['page_content'] for doc in chunks])
            
            new_langchain_docs = [
                Document(page_content=t['page_content'], metadata=t['metadata']) 
                for t in chunks
            ]
            self.langchain_corpus.extend(new_langchain_docs)
            
            # 3. 增量更新BM25（需要重建）
            self._rebuild_bm25()
            
            # 4. 更新Milvus向量库
            if self._need_milvus_rebuild:
                logger.info("需要重建Milvus（之前执行过删除操作）")
                self._rebuild_milvus()
                self._need_milvus_rebuild = False
            else:
                self._add_to_milvus(new_langchain_docs)
            
            logger.info(f"增量添加完成: {source_file}")
            return True
            
        except Exception as e:
            logger.error(f"增量添加文档失败 {source_file}: {e}")
            return False
    
    def delete_documents(self, chunk_ids: List[str]) -> bool:
        """
        按chunk_id删除文档（单文件删除，会立即重建BM25）
        
        注意：由于Milvus auto_id，无法精确删除，采用延迟重建策略：
        - 立即从BM25和语料库中移除
        - 标记需要重建Milvus（下次添加时重建）
        
        批量删除时请使用 delete_documents_batch() 以提高性能
        
        Args:
            chunk_ids: 要删除的chunk ID列表
            
        Returns:
            bool: 是否成功
        """
        if not chunk_ids:
            return True
            
        try:
            logger.info(f"开始删除文档，chunk_ids数量: {len(chunk_ids)}")
            
            # 1. 从语料库中移除
            chunk_id_set = set(chunk_ids)
            self.corpus_dicts = [
                c for c in self.corpus_dicts 
                if c.get('metadata', {}).get('chunk_id') not in chunk_id_set
            ]
            self.corpus = [c['page_content'] for c in self.corpus_dicts]
            self.langchain_corpus = [
                Document(page_content=t['page_content'], metadata=t['metadata']) 
                for t in self.corpus_dicts
            ]
            
            # 2. 重建BM25（立即生效）
            self._rebuild_bm25()
            
            # 3. 标记Milvus需要重建（延迟重建）
            self._need_milvus_rebuild = True
            logger.info(f"标记Milvus需要重建（删除了 {len(chunk_ids)} 个chunks）")
            
            logger.info(f"删除完成（BM25已更新，Milvus将在下次添加时重建）")
            return True
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    def delete_documents_batch(self, file_chunk_map: Dict[str, List[str]]) -> Dict[str, int]:
        """
        批量删除多个文件的文档（性能优化版）
        
        与逐个调用 delete_documents 相比，此方法只重建一次 BM25，
        大幅减少批量删除时的耗时。
        
        Args:
            file_chunk_map: 文件到 chunk_ids 的映射，例如：
                {'file1.md': ['id1', 'id2'], 'file2.md': ['id3', 'id4']}
            
        Returns:
            Dict[str, int]: 统计信息，包含 deleted_files（成功删除文件数）、
                           deleted_chunks（删除的chunk总数）、
                           failed_files（失败的文件数）
        """
        if not file_chunk_map:
            return {"deleted_files": 0, "deleted_chunks": 0, "failed_files": 0}
        
        try:
            total_chunks = sum(len(chunks) for chunks in file_chunk_map.values())
            logger.info(f"开始批量删除，文件数: {len(file_chunk_map)}, chunks总数: {total_chunks}")
            
            # 1. 收集所有要删除的 chunk_ids
            all_chunk_ids = set()
            for chunk_ids in file_chunk_map.values():
                all_chunk_ids.update(chunk_ids)
            
            # 2. 从语料库中一次性移除
            self.corpus_dicts = [
                c for c in self.corpus_dicts 
                if c.get('metadata', {}).get('chunk_id') not in all_chunk_ids
            ]
            self.corpus = [c['page_content'] for c in self.corpus_dicts]
            self.langchain_corpus = [
                Document(page_content=t['page_content'], metadata=t['metadata']) 
                for t in self.corpus_dicts
            ]
            
            # 3. 只重建一次 BM25
            self._rebuild_bm25()
            
            # 4. 标记 Milvus 需要重建
            self._need_milvus_rebuild = True
            
            logger.info(f"批量删除完成，删除了 {len(file_chunk_map)} 个文件的 {len(all_chunk_ids)} 个chunks")
            return {
                "deleted_files": len(file_chunk_map),
                "deleted_chunks": len(all_chunk_ids),
                "failed_files": 0
            }
            
        except Exception as e:
            logger.error(f"批量删除文档失败: {e}")
            return {
                "deleted_files": 0,
                "deleted_chunks": 0,
                "failed_files": len(file_chunk_map)
            }
    
    def update_documents(self, new_chunks: List[dict], source_file: str) -> bool:
        """
        更新文档（删除旧版本+添加新版本）
        
        Args:
            new_chunks: 新的文档chunks
            source_file: 源文件名
            
        Returns:
            bool: 是否成功
        """
        try:
            # 1. 获取旧chunk IDs并删除
            old_chunk_ids = self.kb_manager.get_chunk_ids_by_file(source_file)
            if old_chunk_ids:
                self.delete_documents(old_chunk_ids)
                self.kb_manager.remove_file_record(source_file)
            
            # 2. 添加新版本
            return self.add_documents(new_chunks, source_file)
            
        except Exception as e:
            logger.error(f"更新文档失败 {source_file}: {e}")
            return False
    
    def incremental_update(self, changes: Dict[str, List[str]]) -> Dict[str, int]:
        """
        根据变更列表执行批量增量更新
        
        Args:
            changes: 变更字典，包含 added, deleted, modified 文件列表
            
        Returns:
            统计信息 dict
        """
        from app.read_corpus import Reader
        
        stats = {"added": 0, "deleted": 0, "modified": 0, "failed": []}
        
        logger.info(f"开始增量更新，变更情况: {changes}")
        
        # 1. 批量处理删除和修改旧版本，避免对每个文件都重建一次 BM25
        files_to_remove = changes.get("deleted", []) + changes.get("modified", [])
        file_chunk_map = {}
        for filename in files_to_remove:
            chunk_ids = self.kb_manager.get_chunk_ids_by_file(filename)
            if chunk_ids:
                file_chunk_map[filename] = chunk_ids

        if file_chunk_map:
            delete_result = self.delete_documents_batch(file_chunk_map)
            if delete_result["failed_files"] > 0:
                failed_delete_files = set(file_chunk_map.keys())
                stats["failed"].extend([f"delete:{filename}" for filename in failed_delete_files])
            else:
                for filename in changes.get("deleted", []):
                    self.kb_manager.remove_file_record(filename)
                    stats["deleted"] += 1
                for filename in changes.get("modified", []):
                    self.kb_manager.remove_file_record(filename)
        else:
            logger.info("本次增量更新没有需要删除的旧chunks")
        
        # 2. 处理新增和修改的文件（统一作为新增处理）
        files_to_add = changes.get("added", []) + changes.get("modified", [])
        
        for filename in files_to_add:
            try:
                filepath = os.path.join(self.kb_manager.data_path, filename)
                reader = Reader(filepath)
                chunks = reader.corpus
                
                if chunks:
                    if self.add_documents(chunks, filename):
                        if filename in changes.get("modified", []):
                            stats["modified"] += 1
                        else:
                            stats["added"] += 1
                    else:
                        stats["failed"].append(f"add:{filename}")
                else:
                    logger.warning(f"文件没有解析出内容: {filename}")
                    
            except Exception as e:
                logger.error(f"处理文件失败 {filename}: {e}")
                stats["failed"].append(f"add:{filename}")
        
        logger.info(f"增量更新完成: {stats}")
        return stats
    
    def _rebuild_bm25(self):
        """重建BM25索引"""
        logger.info(f"重建BM25索引，当前语料库大小: {len(self.corpus_dicts)}")
        
        if self.lan == 'zh':
            # 删除旧缓存，强制重新分词
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            tokenized_documents = preprocess_corpus(self.corpus_dicts, self.cache_file)
        else:
            tokenized_documents = [doc.split() for doc in self.corpus]
        
        self.bm25 = BM25Okapi(tokenized_documents)
        logger.info("BM25索引重建完成")
    
    # 内部字段：仅用于知识库管理，不存入 Milvus
    _INTERNAL_META_KEYS = {'chunk_id', 'source_file'}
    
    def _add_to_milvus(self, documents: List[Document]):
        """向Milvus添加文档
        
        注意：会过滤掉内部字段（chunk_id, source_file），这些字段仅用于
        知识库管理器维护文件-chunk映射关系，不存入 Milvus。
        """
        try:
            # 过滤内部字段，只保留需要存入 Milvus 的 metadata
            cleaned_documents = []
            for doc in documents:
                cleaned_metadata = {
                    k: v for k, v in doc.metadata.items() 
                    if k not in self._INTERNAL_META_KEYS
                }
                cleaned_documents.append(Document(
                    page_content=doc.page_content,
                    metadata=cleaned_metadata
                ))
            
            # 使用LangChain的Milvus.add_documents
            # 不传ids，让Milvus自动生成
            self.db.add_documents(cleaned_documents)
            logger.info(f"已添加 {len(documents)} 个文档到Milvus")
        except Exception as e:
            logger.error(f"添加文档到Milvus失败: {e}")
            raise
    
    def _rebuild_milvus(self):
        """重建Milvus向量库（删除后重建时使用）"""
        try:
            logger.info(f"开始重建Milvus，文档数: {len(self.langchain_corpus)}")
            
            # 删除旧集合
            try:
                connections.connect(**self.connection_args)
                if utility.has_collection(self.collection_name):
                    utility.drop_collection(self.collection_name)
                    logger.info(f"已删除旧Milvus集合: {self.collection_name}")
            except Exception as e:
                logger.warning(f"删除旧集合时出错: {e}")
            
            # 过滤内部字段，只保留需要存入 Milvus 的 metadata
            cleaned_corpus = []
            for doc in self.langchain_corpus:
                cleaned_metadata = {
                    k: v for k, v in doc.metadata.items() 
                    if k not in self._INTERNAL_META_KEYS
                }
                cleaned_corpus.append(Document(
                    page_content=doc.page_content,
                    metadata=cleaned_metadata
                ))
            
            # 重新创建集合（使用当前语料库）
            from langchain_community.vectorstores import Milvus
            self.db = Milvus.from_documents(
                documents=cleaned_corpus,
                embedding=self.emb_model,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                drop_old=True,
                index_params=self.index_params
            )
            
            logger.info(f"Milvus重建完成，共 {len(cleaned_corpus)} 个文档")
        except Exception as e:
            logger.error(f"重建Milvus失败: {e}")
            raise
    
    def sync_with_storage(self) -> Dict[str, int]:
        """
        与存储同步，自动检测并应用所有变更
        
        Returns:
            统计信息
        """
        changes = self.kb_manager.detect_changes()
        return self.incremental_update(changes)
