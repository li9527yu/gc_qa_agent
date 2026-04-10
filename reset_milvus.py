#!/usr/bin/env python3
"""
重置Milvus集合
- 删除旧集合（如果有）
- 下次启动时会自动创建新的 auto_id=False 集合
"""

from pymilvus import connections, utility

def reset_milvus():
    print("连接Milvus...")
    connections.connect(host="127.0.0.1", port="19530")
    
    collection_name = "easy_rag_milvus"
    
    if utility.has_collection(collection_name):
        print(f"删除旧集合: {collection_name}")
        utility.drop_collection(collection_name)
        print("✅ 旧集合已删除")
    else:
        print(f"集合 {collection_name} 不存在，无需删除")
    
    # 也清除语料库哈希，强制重建
    import os
    cache_dir = os.path.expanduser("~/.easy_rag_cache")
    hash_file = os.path.join(cache_dir, f"{collection_name}_hash.txt")
    if os.path.exists(hash_file):
        os.remove(hash_file)
        print(f"✅ 已删除哈希缓存: {hash_file}")
    
    print("\n下次启动服务时将自动创建新的集合（auto_id=False）")
    print("请重启RAG服务：python3 start_server.py")

if __name__ == "__main__":
    reset_milvus()
