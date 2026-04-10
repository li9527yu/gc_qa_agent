#!/usr/bin/env python3
"""
RAG服务启动脚本
- 检查环境依赖
- 启动FastAPI服务
"""

import os
import sys
import subprocess
import requests
import time

def check_milvus():
    """检查Milvus是否运行"""
    try:
        # Milvus gRPC端口检查
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 19530))
        sock.close()
        return result == 0
    except:
        return False

def check_llm():
    """检查LLM服务是否运行"""
    try:
        resp = requests.get("http://localhost:9839/v1/models", timeout=5)
        return resp.status_code == 200
    except:
        return False

def main():
    print("="*60)
    print("RAG服务启动脚本")
    print("="*60)
    
    # 1. 检查Milvus
    print("\n📦 检查Milvus...")
    if check_milvus():
        print("   ✅ Milvus已运行 (localhost:19530)")
    else:
        print("   ❌ Milvus未运行!")
        print("   请启动Milvus: docker-compose up -d milvus-standalone")
        return 1
    
    # 2. 检查LLM服务
    print("\n🤖 检查LLM服务...")
    if check_llm():
        print("   ✅ LLM服务已运行 (localhost:9839)")
        try:
            resp = requests.get("http://localhost:9839/v1/models", timeout=5)
            models = resp.json().get('data', [])
            for m in models[:3]:
                print(f"      - {m.get('id')}")
        except:
            pass
    else:
        print("   ⚠️ LLM服务未运行 (localhost:9839)")
        print("   注意: 知识问答功能将不可用")
    
    # 3. 检查数据目录
    print("\n📁 检查数据目录...")
    data_dir = "app/dataset/data"
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith(('.md', '.txt', '.pdf'))]
        print(f"   ✅ 数据目录存在，包含 {len(files)} 个文件")
    else:
        print(f"   ⚠️ 数据目录不存在: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        print("   已创建数据目录")
    
    # 4. 启动服务
    print("\n🚀 启动RAG服务...")
    print("="*60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.api_server:app", 
            "--host", "0.0.0.0", 
            "--port", "8001",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
