from abc import ABC
from modelscope import AutoTokenizer, AutoModel
import torch
import jieba
from langchain.schema.embeddings import Embeddings
from langchain.schema import Document
from typing import List
import numpy as np
from rank_bm25 import BM25Okapi
# import bm25s
from langchain_community.vectorstores import Milvus
import hashlib
import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pickle
from app.config import ROOT_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def tokenize_doc(doc, stopwords=None):
    """
    分词函数：对单个文档进行分词，并根据需要去除停用词。
    
    参数：
        doc (Dict): 包含 'page_content' 字段的文档。
        stopwords (set): 停用词集合。
        
    返回：
        List[str]: 分词结果列表。
    """
    tokens = jieba.lcut(doc['page_content'])
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords and t.strip()]
    return tokens
def preprocess_corpus(
    corpus,
    cache_file="tokenized_docs.pkl",
    use_stopwords=True,
    stopwords_file=ROOT_PATH+"/rag_cache/stopwords/stopwords_hit.txt",
    num_workers=4):
    """
    预处理语料：中文分词 + 可选停用词过滤 + 缓存机制
    
    参数：
        corpus (List[Dict]): 包含 'page_content' 字段的文档列表。
        cache_file (str): 缓存文件路径。
        use_stopwords (bool): 是否启用停用词过滤。
        stopwords_file (str): 停用词文件路径。
    
    返回：
        List[List[str]]: 分词后的语料列表。
    """
    if os.path.exists(cache_file):
        logging.info(f"发现缓存文件 {cache_file}，正在加载...")
        with open(cache_file, "rb") as f:
            tokenized_documents = pickle.load(f)
        return tokenized_documents

    logging.info("未找到缓存文件，开始预处理语料...")

    # 加载停用词表（如果启用）
    stopwords = set()
    if use_stopwords:
        if os.path.exists(stopwords_file):
            with open(stopwords_file, "r", encoding="utf-8") as f:
                for line in f:
                    stopwords.add(line.strip())
            logging.info(f"已加载 {len(stopwords)} 个停用词。")
        else:
            logging.warning(f"停用词文件 {stopwords_file} 不存在，跳过停用词过滤。")

    # 分词 + 过滤（可选）
    tokenized_documents = []
    for doc in tqdm(corpus, desc="分词处理中"):
        tokens = jieba.lcut(doc['page_content'])
        if use_stopwords:
            tokens = [t for t in tokens if t not in stopwords and t.strip()]
        tokenized_documents.append(tokens)
    # 多线程分词：
    # tokenized_documents = []
    # with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     futures = {executor.submit(tokenize_doc, doc, stopwords): doc for doc in corpus}
    #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(corpus), desc="多线程分词中"):
    #         try:
    #             result = future.result()
    #             tokenized_documents.append(result)
    #         except Exception as exc:
    #             logging.error(f"生成分词结果时发生错误: {exc}")

    # 保存缓存
    logging.info(f"正在将分词结果缓存到 {cache_file} ...")
    with open(cache_file, "wb") as f:
        pickle.dump(tokenized_documents, f)

    return tokenized_documents 

def compute_corpus_hash(corpus: List[dict]) -> str:
    """计算语料库的哈希值"""
    # 使用文档内容和元数据计算哈希值
    corpus_info = []
    for doc in corpus:
        content = doc.get('page_content', '')
        metadata = str(doc.get('metadata', {}))
        corpus_info.append(f"{content}:{metadata}")
    corpus_str = "".join(sorted(corpus_info))  # 排序以确保相同内容产生相同的哈希值
    return hashlib.md5(corpus_str.encode('utf-8')).hexdigest()

def save_corpus_hash(hash_value: str, collection_name: str):
    """保存语料库哈希值"""
    cache_dir = os.path.join(os.path.expanduser("~"), ".easy_rag_cache")
    os.makedirs(cache_dir, exist_ok=True)
    hash_file = os.path.join(cache_dir, f"{collection_name}_hash.txt")
    with open(hash_file, "w") as f:
        f.write(hash_value)

def load_corpus_hash(collection_name: str) -> str:
    """加载语料库哈希值"""
    cache_dir = os.path.join(os.path.expanduser("~"), ".easy_rag_cache")
    hash_file = os.path.join(cache_dir, f"{collection_name}_hash.txt")
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            return f.read().strip()
    return ""

class TextEmbedding(Embeddings, ABC):
    def __init__(self, emb_model_name_or_path, batch_size=64, max_len=512, device='cuda', **kwargs):

        super().__init__(**kwargs)
        self.model = AutoModel.from_pretrained(emb_model_name_or_path, trust_remote_code=True).half().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model_name_or_path, trust_remote_code=True)
        if 'bge' in emb_model_name_or_path:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："
        else:
            self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH = ""
        self.emb_model_name_or_path = emb_model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len
        logger.info("Successfully loaded embedding model")

    def compute_kernel_bias(self, vecs, n_components=384):
        """
            bertWhitening: https://spaces.ac.cn/archives/8069
            计算kernel和bias
            vecs.shape = [num_samples, embedding_size]，
            最后的变换：y = (x + bias).dot(kernel)
        """
        mu = vecs.mean(axis=0, keepdims=True)
        cov = np.cov(vecs.T)
        u, s, vh = np.linalg.svd(cov)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))
        return W[:, :n_components], -mu

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
            Compute corpus embeddings using a HuggingFace transformer model.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        num_texts = len(texts)
        texts = [t.replace("\n", " ") for t in texts]
        sentence_embeddings = []

        for start in range(0, num_texts, self.batch_size):
            end = min(start + self.batch_size, num_texts)
            batch_texts = texts[start:end]
            encoded_input = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True,
                                           return_tensors='pt').to(self.device)

            with torch.no_grad():

                model_output = self.model(**encoded_input)
                # Perform pooling. In this case, cls pooling.
                if 'gte' in self.emb_model_name_or_path:
                    batch_embeddings = model_output.last_hidden_state[:, 0]
                else:
                    batch_embeddings = model_output[0][:, 0]

                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                sentence_embeddings.extend(batch_embeddings.tolist())

        # sentence_embeddings = np.array(sentence_embeddings)
        # self.W, self.mu = self.compute_kernel_bias(sentence_embeddings)
        # sentence_embeddings = (sentence_embeddings+self.mu) @ self.W
        # self.W, self.mu = torch.from_numpy(self.W).cuda(), torch.from_numpy(self.mu).cuda()
        return sentence_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
            Compute query embeddings using a HuggingFace transformer model.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        if 'bge' in self.emb_model_name_or_path:
            encoded_input = self.tokenizer([self.DEFAULT_QUERY_BGE_INSTRUCTION_ZH + text], padding=True,
                                           truncation=True, return_tensors='pt').to(self.device)
        else:
            encoded_input = self.tokenizer([text], padding=True,
                                           truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        # sentence_embeddings = (sentence_embeddings + self.mu) @ self.W
        return sentence_embeddings[0].tolist()

class SentenceTransformerEmbedding(Embeddings, ABC):
    def __init__(self, emb_model_name_or_path, device='cuda', **kwargs):
        super().__init__(**kwargs)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(emb_model_name_or_path)
        if torch.cuda.is_available() and device.startswith('cuda'):
            self.model = self.model.to(device)
        self.device = device
        logger.info(f"Successfully loaded SentenceTransformer embedding model on {device}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        使用SentenceTransformer计算文档嵌入向量
        使用小批次处理避免显存溢出
        """
        texts = [t.replace("\n", " ") for t in texts]
        # 使用小批次处理，避免一次性送入过多文本导致显存溢出
        batch_size = 2  # 可根据显存大小调整
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            embeddings.extend(batch_embeddings.tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        使用SentenceTransformer计算查询嵌入向量
        """
        text = text.replace("\n", " ")
        embedding = self.model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()

class Retriever:
    def __init__(self, emb_model_name_or_path, corpus,tokenized_path, device='cuda', lan='zh', 
                 collection_name='easy_rag_milvus', milvus_mode='standalone', 
                 embedding_type='modelscope', force_rebuild=False):
        """
        Args:
            emb_model_name_or_path: 嵌入模型的名称或路径
            corpus: 文档语料库
            device: 设备类型 ('cuda' 或 'cpu')
            lan: 语言类型 ('zh' 或 'en')
            collection_name: Milvus集合名称
            milvus_mode: 'standalone' 或 'lite'
            embedding_type: 'modelscope' 或 'sentence_transformer'
            force_rebuild: 是否强制重建向量数据库
        """
        self.device = device
        self.langchain_corpus = [Document(page_content=t['page_content'],metadata=t['metadata']) for t in corpus]
        # self.langchain_corpus = corpus
        # 保存原始字典格式的语料库（用于返回完整信息）
        self.corpus_dicts = corpus
        # 提取 page_content 用于 BM25（BM25Okapi 需要字符串列表）
        self.corpus = [doc['page_content'] for doc in corpus]
        self.lan = lan
        self.collection_name = collection_name
        self.cache_file=tokenized_path
        # 计算语料库哈希值
        corpus_hash = compute_corpus_hash(corpus)
        cached_hash = load_corpus_hash(collection_name)
        
        # 设置 Milvus 连接参数
        if milvus_mode == 'lite':
            self.connection_args = {"uri": ROOT_PATH+"/milvus_rag.db"}
            self.index_params = {
                "metric_type": "L2",
                "index_type": "FLAT",
                "params": {}
            }
        else:
            self.connection_args = {"host": "127.0.0.1", "port": "19530"}
            self.index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }

        # 初始化嵌入模型
        if embedding_type == 'modelscope':
            self.emb_model = TextEmbedding(emb_model_name_or_path=emb_model_name_or_path, device=device)
        elif embedding_type == 'sentence_transformer':
            self.emb_model = SentenceTransformerEmbedding(emb_model_name_or_path=emb_model_name_or_path, device=device)
        elif embedding_type == 'qwen3':
            from app.qwen3_embedding import Qwen3Embedding
            # 从配置中读取维度
            from app.embedding_config import EMBEDDING_DIMENSION, QWEN3_EMBEDDING_DIM
            # 如果使用完整维度，用 QWEN3_EMBEDDING_DIM，否则用配置的 EMBEDDING_DIMENSION
            use_full_dim = EMBEDDING_DIMENSION >= 2048
            self.emb_model = Qwen3Embedding(
                model_path=emb_model_name_or_path,
                device=device,
                embedding_dim=QWEN3_EMBEDDING_DIM if use_full_dim else EMBEDDING_DIMENSION,
                use_fp16=True,
                batch_size=8 if '0.6b' in emb_model_name_or_path.lower() else 4
            )
        else:
            raise ValueError(f"Unsupported embedding_type: {embedding_type}")
        logger.info("Rebuilding vector database...")
        
        # 检查是否需要重建分词缓存
        need_rebuild_cache = force_rebuild or corpus_hash != cached_hash
        
        if need_rebuild_cache:
            if corpus_hash != cached_hash:
                logger.info(f"语料库哈希值发生变化，需要重建缓存")
                logger.info(f"当前哈希: {corpus_hash}, 缓存哈希: {cached_hash}")
            if force_rebuild:
                logger.info("强制重建模式，删除所有缓存")
            
            # 删除分词缓存文件
            if os.path.exists(self.cache_file):
                logger.info(f"删除分词缓存文件: {self.cache_file}")
                os.remove(self.cache_file)
        
        if lan == 'zh':
            tokenized_documents=preprocess_corpus(self.corpus_dicts,self.cache_file)
        else:
            tokenized_documents = [doc.split() for doc in self.corpus]
        
        # 验证分词结果与语料库长度一致
        if len(tokenized_documents) != len(self.corpus):
            logger.error(f"分词结果长度 ({len(tokenized_documents)}) 与语料库长度 ({len(self.corpus)}) 不匹配！")
            logger.info("删除分词缓存并重新构建...")
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            tokenized_documents = preprocess_corpus(self.corpus_dicts, self.cache_file)
        
        self.bm25 = BM25Okapi(tokenized_documents)
        # self.bm25 = bm25s.BM25(
        #         k1=1.5,
        #         b=0.75,
        #     )
        # self.bm25.index(tokenized_documents)
        
        # 重建向量数据库
        if need_rebuild_cache:
            logger.info("重建向量数据库...")
            # 确保删除旧的集合
            try:
                from pymilvus import connections, utility
                connections.connect(**self.connection_args)
                if utility.has_collection(self.collection_name):
                    utility.drop_collection(self.collection_name)
                    logger.info(f"已删除旧集合: {self.collection_name}")
            except Exception as e:
                logger.warning(f"删除旧集合时出错: {e}")
            
            # 过滤内部字段（仅用于知识库管理，不入 Milvus）
            internal_meta_keys = {'chunk_id', 'source_file'}
            cleaned_corpus = []
            for doc in self.langchain_corpus:
                cleaned_metadata = {
                    k: v for k, v in doc.metadata.items() 
                    if k not in internal_meta_keys
                }
                cleaned_corpus.append(Document(
                    page_content=doc.page_content,
                    metadata=cleaned_metadata
                ))
            
            self.db = Milvus.from_documents(
                documents=cleaned_corpus,
                embedding=self.emb_model,
                collection_name=collection_name,
                connection_args=self.connection_args,
                drop_old=True,
                index_params=self.index_params
                # 注意：不传ids，让Milvus自动生成
                # 删除时通过重建Milvus实现（因为无法通过metadata删除）
            )
            
            # 保存新的哈希值
            save_corpus_hash(corpus_hash, collection_name)
            logger.info(f"向量数据库构建完成，共 {len(corpus)} 个文档")
        else:
            logger.info("语料库未变化，直接加载现有向量数据库...")
            self.db = Milvus(
                embedding_function=self.emb_model,
                collection_name=collection_name,
                connection_args=self.connection_args,
                auto_id=True,
            )
            
        # 检查是否需要重建向量数据库
        # if force_rebuild or corpus_hash != cached_hash:
        #     logger.info("Rebuilding vector database...")
        #     if lan == 'zh':
        #         tokenized_documents = [jieba.lcut(doc) for doc in corpus]
        #     else:
        #         tokenized_documents = [doc.split() for doc in corpus]
            
        #     self.bm25 = BM25Okapi(tokenized_documents)
            
        #     # 重建向量数据库
        #     self.db = Milvus.from_documents(
        #         documents=self.langchain_corpus,
        #         embedding=self.emb_model,
        #         collection_name=collection_name,
        #         connection_args=self.connection_args,
        #         drop_old=True,
        #         index_params=self.index_params
        #     )
            
        #     # 保存新的哈希值
        #     save_corpus_hash(corpus_hash, collection_name)
        #     logger.info(f"Vector database rebuilt successfully with {len(corpus)} documents")
        # else:
        #     logger.info("Using cached vector database...")
        #     # 直接加载现有的向量数据库
        #     self.db = Milvus(
        #         embedding_function=self.emb_model,
        #         collection_name=collection_name,
        #         connection_args=self.connection_args,
        #     )
            
        #     # 初始化 BM25
        #     if lan == 'zh':
        #         tokenized_documents = [jieba.lcut(doc) for doc in corpus]
        #     else:
        #         tokenized_documents = [doc.split() for doc in corpus]
        #     self.bm25 = BM25Okapi(tokenized_documents)

    # def update_corpus(self, new_corpus):
    #     """
    #     用于更新 Retriever 的语料库，并重建 bm25 和 Milvus 向量库。
    #     """
    #     logger.info("正在更新 Retriever 的语料库...")
    #     self.corpus = new_corpus
    #     self.langchain_corpus = [Document(page_content=t['page_content'], metadata=t['metadata']) for t in new_corpus]

    #     # 重建 BM25
    #     if self.lan == 'zh':
    #         tokenized_documents = preprocess_corpus(new_corpus, self.cache_file)
    #     else:
    #         tokenized_documents = [doc.split() for doc in new_corpus]

    #     self.bm25 = BM25Okapi(tokenized_documents)

    #     # 重建向量库
    #     self.db = Milvus.from_documents(
    #         documents=self.langchain_corpus,
    #         embedding=self.emb_model,
    #         collection_name=self.collection_name,
    #         connection_args=self.connection_args,
    #         drop_old=True,
    #         index_params=self.index_params
    #     )

    #     logger.info(f"Retriever 已成功更新，新语料文档数: {len(new_corpus)}")

    def bm25_retrieval(self, query, n=20):
        try:
            # 此处中文使用jieba分词
            query = jieba.lcut(query) if self.lan == 'zh' else query.split()
            # BM25Okapi.get_top_n 返回的是原始文档列表中的元素（字符串）
            res_texts = self.bm25.get_top_n(query, self.corpus, n=n)
            # 将文本映射回字典格式，与后续处理兼容
            res = []
            for text in res_texts:
                # 在 corpus_dicts 中找到对应的字典
                for doc in self.corpus_dicts:
                    if doc['page_content'] == text:
                        res.append(doc)
                        break
            return res
        except AssertionError as e:
            if "The documents given don't match the index corpus" in str(e):
                logger.error("BM25索引与语料库不匹配，正在重建索引...")
                # 删除分词缓存并重新构建
                if os.path.exists(self.cache_file):
                    os.remove(self.cache_file)
                if self.lan == 'zh':
                    tokenized_documents = preprocess_corpus(self.corpus_dicts, self.cache_file)
                else:
                    tokenized_documents = [doc.split() for doc in self.corpus]
                self.bm25 = BM25Okapi(tokenized_documents)
                # 重试检索
                query = jieba.lcut(query) if self.lan == 'zh' else query.split()
                res_texts = self.bm25.get_top_n(query, self.corpus, n=n)
                # 将文本映射回字典格式
                res = []
                for text in res_texts:
                    for doc in self.corpus_dicts:
                        if doc['page_content'] == text:
                            res.append(doc)
                            break
                return res
            else:
                raise e

    def emb_retrieval(self, query, k=20):
        search_docs = self.db.similarity_search(query, k=k)
        # 返回字典格式，与 bm25_retrieval 保持一致
        res = []
        for doc in search_docs:
            res.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
        return res

    def retrieval(self, query, methods=None):
        if methods is None:
            methods = ['bm25', 'emb']

        search_res = []
        seen_docs = set()

        for method in methods:
            if method == 'bm25':
                current_res = self.bm25_retrieval(query)
            elif method == 'emb':
                current_res = self.emb_retrieval(query)
            else:
                logger.warning(f"未知检索方法: {method}，已跳过")
                continue

            logger.info(f"检索方法 {method} 返回 {len(current_res)} 个候选文档")

            for item in current_res:
                metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
                dedup_key = (
                    metadata.get("chunk_id")
                    or (
                        metadata.get("source"),
                        metadata.get("page"),
                        item.get("page_content", "") if isinstance(item, dict) else str(item),
                    )
                )
                if dedup_key in seen_docs:
                    continue
                seen_docs.add(dedup_key)
                search_res.append(item)

        logger.info(f"混合检索完成，methods={methods}，去重后返回 {len(search_res)} 个文档")
        return search_res
