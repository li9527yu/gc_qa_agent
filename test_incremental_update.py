#!/usr/bin/env python3
"""
增量更新功能测试脚本

测试场景：
1. 上传新文件（增量添加）
2. 删除文件（增量删除）
3. 检查索引更新
"""

import os
import sys
import json
import time
import requests
from pathlib import Path

BASE_URL = "http://localhost:8001"
DATA_DIR = "app/dataset/data"
TEST_FILE_1 = "test_incremental_1.md"
TEST_FILE_2 = "test_incremental_2.md"

def create_test_file(filename, content):
    """创建测试文件"""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"   创建测试文件: {filepath}")
    return filepath

def cleanup_test_files():
    """清理测试文件"""
    for filename in [TEST_FILE_1, TEST_FILE_2]:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"   清理: {filepath}")

def test_service_status():
    """测试服务状态"""
    print("\n" + "="*60)
    print("测试1: 服务状态检查")
    print("="*60)
    
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/service/status")
        data = resp.json()
        print(f"   状态码: {resp.status_code}")
        print(f"   响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
        return data.get('retriever_ready', False)
    except Exception as e:
        print(f"   ❌ 请求失败: {e}")
        return False

def test_upload():
    """测试文件上传（增量添加）"""
    print("\n" + "="*60)
    print("测试2: 文件上传（增量添加）")
    print("="*60)
    
    # 创建测试文件
    content = """# 测试文档

这是一个用于测试增量更新的文档。

## 章节1

这是第一个章节的内容。

## 章节2

这是第二个章节的内容。
"""
    create_test_file(TEST_FILE_1, content)
    
    # 上传文件
    filepath = os.path.join(DATA_DIR, TEST_FILE_1)
    try:
        with open(filepath, 'rb') as f:
            files = [('files', (TEST_FILE_1, f, 'text/markdown'))]
            resp = requests.post(f"{BASE_URL}/api/v1/upload", files=files)
        
        data = resp.json()
        print(f"   状态码: {resp.status_code}")
        print(f"   响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        task_id = data.get('task_id')
        if task_id:
            print(f"   任务ID: {task_id}")
            return task_id
    except Exception as e:
        print(f"   ❌ 上传失败: {e}")
    return None

def test_check_task_status(task_id):
    """检查任务状态"""
    print("\n" + "="*60)
    print("测试3: 检查任务状态")
    print("="*60)
    
    max_retries = 30
    for i in range(max_retries):
        try:
            resp = requests.get(f"{BASE_URL}/api/v1/task_status", params={"task_id": task_id})
            data = resp.json()
            status = data.get('status', 'unknown')
            print(f"   [{i+1}/{max_retries}] 状态: {status}")
            
            if status in ['success', 'success (no changes)']:
                print("   ✅ 任务完成")
                return True
            elif 'failed' in status:
                print(f"   ❌ 任务失败: {status}")
                return False
            
            time.sleep(2)
        except Exception as e:
            print(f"   ❌ 请求失败: {e}")
            return False
    
    print("   ⚠️ 等待超时")
    return False

def test_file_list():
    """测试文件列表"""
    print("\n" + "="*60)
    print("测试4: 获取文件列表")
    print("="*60)
    
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/files")
        data = resp.json()
        print(f"   状态码: {resp.status_code}")
        print(f"   文件总数: {data.get('total_files', 0)}")
        
        # 查找测试文件
        files = data.get('files', [])
        test_files = [f for f in files if 'test_incremental' in f.get('filename', '')]
        print(f"   测试文件数量: {len(test_files)}")
        for f in test_files:
            print(f"      - {f.get('filename')}")
        
        return len(test_files)
    except Exception as e:
        print(f"   ❌ 请求失败: {e}")
        return 0

def test_delete():
    """测试文件删除（增量删除）"""
    print("\n" + "="*60)
    print("测试5: 文件删除（增量删除）")
    print("="*60)
    
    try:
        resp = requests.delete(
            f"{BASE_URL}/api/v1/delete/batch",
            json={"filenames": [TEST_FILE_1]}
        )
        data = resp.json()
        print(f"   状态码: {resp.status_code}")
        print(f"   响应: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        task_id = data.get('task_id')
        if task_id:
            print(f"   任务ID: {task_id}")
            return task_id
    except Exception as e:
        print(f"   ❌ 删除失败: {e}")
    return None

def test_kb_index():
    """检查知识库索引"""
    print("\n" + "="*60)
    print("测试6: 检查知识库索引")
    print("="*60)
    
    index_path = "app/dataset/kb_index.json"
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
            files = index.get('files', {})
            print(f"   索引文件存在")
            print(f"   索引文件数: {len(files)}")
            
            # 检查测试文件是否在索引中
            test_in_index = [f for f in files if 'test_incremental' in f]
            print(f"   测试文件在索引中: {len(test_in_index)}")
            for f in test_in_index:
                info = files[f]
                print(f"      - {f}: {info.get('doc_count')} chunks")
            
            return True
        except Exception as e:
            print(f"   ❌ 读取索引失败: {e}")
    else:
        print(f"   ⚠️ 索引文件不存在: {index_path}")
    return False

def test_knowledge_qa():
    """测试知识问答"""
    print("\n" + "="*60)
    print("测试7: 知识问答测试")
    print("="*60)
    
    try:
        # 首先检查意图
        resp = requests.post(
            f"{BASE_URL}/api/v1/query/intent",
            json={"question": "测试文档的章节1讲了什么"}
        )
        intent_data = resp.json()
        print(f"   意图识别: {intent_data.get('intent')}")
        
        if intent_data.get('intent') == 'knowledge_qa':
            print("   正在测试流式查询...")
            resp = requests.post(
                f"{BASE_URL}/api/v1/query/stream",
                json={"question": "测试文档的章节1讲了什么", "num_docs": 3},
                stream=True
            )
            
            print("   流式响应:")
            content = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        # 尝试解析JSON（最后的meta信息）
                        data = json.loads(line)
                        if 'contexts' in data:
                            print(f"\n   检索到的上下文数量: {len(data.get('contexts', []))}")
                    except:
                        # 普通文本块
                        text = line.decode('utf-8')
                        content += text
                        print(text, end='', flush=True)
            
            print("\n   ✅ 查询完成")
            return True
    except Exception as e:
        print(f"   ❌ 查询失败: {e}")
    return False

def main():
    print("="*60)
    print("RAG增量更新功能测试")
    print("="*60)
    print(f"服务地址: {BASE_URL}")
    print(f"数据目录: {DATA_DIR}")
    
    # 清理之前的测试文件
    print("\n🧹 清理之前的测试文件...")
    cleanup_test_files()
    
    # 运行测试
    results = []
    
    # 测试1: 服务状态
    results.append(("服务状态", test_service_status()))
    
    # 测试2: 上传文件
    task_id = test_upload()
    results.append(("文件上传", task_id is not None))
    
    # 测试3: 检查任务状态
    if task_id:
        results.append(("任务完成", test_check_task_status(task_id)))
    
    # 测试4: 文件列表
    file_count = test_file_list()
    results.append(("文件列表", file_count > 0))
    
    # 测试5: 知识库索引
    results.append(("知识库索引", test_kb_index()))
    
    # 测试6: 知识问答
    results.append(("知识问答", test_knowledge_qa()))
    
    # 测试7: 删除文件
    delete_task_id = test_delete()
    results.append(("文件删除", delete_task_id is not None))
    
    # 测试8: 检查删除任务
    if delete_task_id:
        results.append(("删除完成", test_check_task_status(delete_task_id)))
    
    # 测试9: 再次检查文件列表
    print("\n" + "="*60)
    print("测试: 删除后检查文件列表")
    print("="*60)
    file_count_after = test_file_list()
    results.append(("删除后文件列表", file_count_after == 0))
    
    # 清理
    print("\n🧹 清理测试文件...")
    cleanup_test_files()
    
    # 汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {status}: {name}")
    
    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\n总计: {passed_count}/{total_count} 项测试通过")
    
    return 0 if passed_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())
