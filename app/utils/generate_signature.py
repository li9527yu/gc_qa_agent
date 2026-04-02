#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Key 签名生成工具
直接运行此脚本即可生成签名
"""

import hmac
import hashlib
import base64
import json
import sys
 
SECRET_KEY = "uwRbSmxx9-X5NEFPTkjpqhj6GBaVVI6V5ulxtczgY5I" # 私钥
# ================================================================

def generate_signature(secret_key, params, module, service, operator, debug=False):
    """
    生成 API 签名
    
    参数:
        secret_key: 私钥
        params: 参数字典
        module: 模块名（如 "largeModelMaterial"）
        service: 服务名（如 "infor", "factory", "zhichengInfo"）
        operator: 操作名（如 "getInforMaterial"）
        debug: 是否打印调试信息（默认 False）
    
    返回:
        str: Base64 编码的签名
    """
    # 1. 过滤并排序参数（排除 signature，排除 null 值）
    filtered_params = {k: v for k, v in params.items() 
                       if k not in ['signature'] and v is not None}
    sorted_params = sorted(filtered_params.items())
    
    # 2. 拼接字符串
    param_string = '&'.join([f'{k}={v}' for k, v in sorted_params])
    sign_string = f'{param_string}&module={module}&service={service}&operator={operator}'
    
    if debug:
        print(f"签名原始字符串: {sign_string}")
    
    # 3. HMAC-SHA256签名
    hmac_obj = hmac.new(
        secret_key.encode('utf-8'),
        sign_string.encode('utf-8'),
        hashlib.sha256
    )
    
    # 4. Base64编码
    return base64.b64encode(hmac_obj.digest()).decode('utf-8')

def main():

    
    # 示例1：调用 largeModelMaterial.infor.getInforMaterial
    print("示例1: largeModelMaterial.infor.getInforMaterial")
    print("-" * 70)
    
    params1 = {
        "accountId": "testoxidmwedxdkseucdnvksfnzmdfnzd",
        "matchMethod": 1,
        "categoryOneLevelName": "",
        "categoryTwoLevelName": "",
        "categoryThreeLevelName": "",
        "minPrice": None,
        "maxPrice": None,
        "releaseDepartment": "",
        "startReleaseDate": "",
        "endReleaseDate": "",
        "province": "",
        "city": "",
        "brand": "",
        "supplyName": "",
        "materialModelSpec": "",
        "materialName": "铝合金幕墙型材",
        "returnNumber": 10000,
        "returnTotalCount": 1
    }
        
    module1 = "largeModelMaterial"
    service1 = "infor"
    operator1 = "getInforMaterial"
    
    signature1 = generate_signature(SECRET_KEY, params1, module1, service1, operator1)
    print(f"signature: {signature1}")
   

if __name__ == "__main__":
    main()

