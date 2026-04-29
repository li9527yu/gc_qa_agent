#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试当前最新的价格材料接口可用性。

依据：
1. 最新 Apifox 文档：
   /material/factoryRelatedMaterials/getRelatedMaterial
2. 认证方式：
   header 传 signature
3. 必传/固定参数：
   - accountId=dce1a70c6507f10d266385d9eec7db09
   - matchMethod=2
   - checkState=1
   - belongDataPool=2
   - excludeExactMatch=false
4. 分页要求：
   - 每页最多 2000
   - 可多页拉取
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

import requests

from app.config import API_SECRET_KEY, PRICE_API_ACCOUNT_ID, REAL_API_BASE_URL
from app.utils.generate_signature import generate_signature


ENDPOINT_PATH = "/material/factoryRelatedMaterials/getRelatedMaterial"
DEFAULT_TIMEOUT = 30
DEFAULT_PAGE_SIZE = 2000


def build_payload(args: argparse.Namespace, material_name: str, page: int) -> Dict[str, Any]:
    return {
        "accountId": PRICE_API_ACCOUNT_ID,
        "matchMethod": 2,
        "excludeExactMatch": False,
        "checkState": 1,
        "belongDataPool": 2,
        "categoryOneLevelId": None,
        "categoryTwoLevelId": None,
        "categoryThreeLevelId": None,
        "categoryOneLevelName": args.category_one_level_name or None,
        "categoryTwoLevelName": args.category_two_level_name or None,
        "categoryThreeLevelName": args.category_three_level_name or None,
        "materialName": material_name,
        "materialCode": args.material_code or None,
        "materialModelSpec": args.material_model_spec or None,
        "domain": args.domain or None,
        "releaseDepartment": args.release_department or None,
        "startReleaseDate": args.start_release_date or None,
        "endReleaseDate": args.end_release_date or None,
        "minPrice": args.min_price,
        "maxPrice": args.max_price,
        "province": args.province or None,
        "city": args.city or None,
        "provinceId": None,
        "cityId": None,
        "enterpriseId": None,
        "enterpriseName": None,
        "queryAccountId": None,
        "accountName": None,
        "brand": args.brand or None,
        "supplyName": args.supply_name or None,
        "page": page,
        "pageSize": min(args.page_size, DEFAULT_PAGE_SIZE),
    }


def build_headers(payload: Dict[str, Any]) -> Dict[str, str]:
    signature = generate_signature(
        secret_key=API_SECRET_KEY,
        params=payload,
        module="material",
        service="factoryRelatedMaterials",
        operator="getRelatedMaterial",
        debug=False,
    )
    return {
        "Content-Type": "application/json",
        "signature": signature,
    }


def request_page(args: argparse.Namespace, material_name: str, page: int) -> Dict[str, Any]:
    payload = build_payload(args, material_name, page)
    headers = build_headers(payload)
    url = f"{REAL_API_BASE_URL}{ENDPOINT_PATH}"

    started_at = time.time()
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=args.timeout,
    )
    elapsed_ms = int((time.time() - started_at) * 1000)

    response.raise_for_status()
    result = response.json()
    result["_debug"] = {
        "url": url,
        "page": page,
        "page_size": payload["pageSize"],
        "elapsed_ms": elapsed_ms,
        "payload": payload,
        "headers": {
            "Content-Type": headers["Content-Type"],
            "signature": headers["signature"],
        },
    }
    return result


def extract_records(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = result.get("data") or {}
    records = data.get("list")
    if isinstance(records, list):
        return records
    return []


def extract_total_count(result: Dict[str, Any]) -> Optional[int]:
    data = result.get("data") or {}
    total_count = data.get("totalCount")
    if isinstance(total_count, int):
        return total_count
    return None


def summarize_records(records: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    summary = []
    for item in records[:limit]:
        summary.append(
            {
                "materialName": item.get("materialName"),
                "materialModelSpec": item.get("materialModelSpec"),
                "brand": item.get("brand"),
                "supplyName": item.get("supplyName"),
                "price": item.get("price"),
                "unit": item.get("unit"),
                "province": item.get("province"),
                "city": item.get("city"),
                "releaseDate": item.get("releaseDate"),
            }
        )
    return summary


def print_single_result(result: Dict[str, Any], material_name: str, raw: bool) -> None:
    if raw:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    records = extract_records(result)
    total_count = extract_total_count(result)
    debug = result.get("_debug") or {}

    print("=== 接口可用性测试结果 ===")
    print(f"material_name: {material_name}")
    print("HTTP: 200")
    print(f"业务 code: {result.get('code')}")
    print(f"message: {result.get('message')}")
    print(f"page: {debug.get('page')}")
    print(f"page_size: {debug.get('page_size')}")
    print(f"elapsed_ms: {debug.get('elapsed_ms')}")
    print(f"totalCount: {total_count}")
    print(f"current_page_records: {len(records)}")
    print("records_sample:")
    print(json.dumps(summarize_records(records), ensure_ascii=False, indent=2))


def run_paginated(args: argparse.Namespace, material_name: str) -> int:
    all_records: List[Dict[str, Any]] = []
    total_count: Optional[int] = None

    for page in range(1, args.max_pages + 1):
        result = request_page(args, material_name, page)
        code = result.get("code")
        if code != 200:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"\n分页测试失败：业务 code != 200，停止于第 {page} 页", file=sys.stderr)
            return 2

        records = extract_records(result)
        page_total_count = extract_total_count(result)
        if isinstance(page_total_count, int) and page_total_count >= 0:
            total_count = page_total_count
        all_records.extend(records)

        debug = result.get("_debug") or {}
        print(
            f"[page {page}] material={material_name} code={code} "
            f"records={len(records)} totalCount={page_total_count} "
            f"elapsed_ms={debug.get('elapsed_ms')}"
        )

        if not records:
            break
        if len(records) < min(args.page_size, DEFAULT_PAGE_SIZE):
            break
        if total_count is not None and len(all_records) >= total_count:
            break

    print("\n=== 分页汇总 ===")
    print(f"material_name: {material_name}")
    print(f"accumulated_records: {len(all_records)}")
    print(f"reported_totalCount: {total_count}")
    print("records_sample:")
    print(json.dumps(summarize_records(all_records), ensure_ascii=False, indent=2))
    return 0


def run_batch(args: argparse.Namespace, material_names: List[str]) -> int:
    overall_code = 0
    for material_name in material_names:
        print(f"\n===== 批量测试: {material_name} =====")
        try:
            result = request_page(args, material_name, page=1)
            print_single_result(result, material_name, raw=False)
            if result.get("code") != 200:
                overall_code = 2
        except requests.HTTPError as exc:
            overall_code = 3
            response_text = exc.response.text[:1000] if exc.response is not None else ""
            print(f"HTTP 请求失败: {exc}", file=sys.stderr)
            if response_text:
                print(response_text, file=sys.stderr)
        except requests.RequestException as exc:
            overall_code = 4
            print(f"网络请求失败: {exc}", file=sys.stderr)
        except Exception as exc:
            overall_code = 5
            print(f"脚本执行失败: {exc}", file=sys.stderr)
    return overall_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试最新 signature 认证的材料价格接口"
    )
    parser.add_argument("--material-name", default="水泥", help="材料名称")
    parser.add_argument(
        "--batch-materials",
        default="",
        help="批量测试材料名，英文逗号分隔，例如: 水泥,电缆",
    )
    parser.add_argument("--province", default="", help="省份")
    parser.add_argument("--city", default="", help="城市")
    parser.add_argument("--material-model-spec", default="", help="规格型号")
    parser.add_argument("--material-code", default="", help="材料编码")
    parser.add_argument("--brand", default="", help="品牌")
    parser.add_argument("--supply-name", default="", help="供应商")
    parser.add_argument("--release-department", default="", help="发布单位")
    parser.add_argument("--category-one-level-name", default="", help="一级分类")
    parser.add_argument("--category-two-level-name", default="", help="二级分类")
    parser.add_argument("--category-three-level-name", default="", help="三级分类")
    parser.add_argument("--domain", default="", help="领域")
    parser.add_argument("--start-release-date", default="", help="开始发布日期")
    parser.add_argument("--end-release-date", default="", help="结束发布日期")
    parser.add_argument("--min-price", type=float, default=None, help="最低价")
    parser.add_argument("--max-price", type=float, default=None, help="最高价")
    parser.add_argument("--page-size", type=int, default=20, help="每页条数，最大 2000")
    parser.add_argument("--max-pages", type=int, default=10, help="分页模式下最多拉取页数")
    parser.add_argument("--paginate", action="store_true", help="分页拉取所有结果")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="请求超时秒数")
    parser.add_argument("--raw", action="store_true", help="打印完整响应 JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    material_names = [
        item.strip() for item in args.batch_materials.split(",") if item.strip()
    ]

    print("=== 最新材料接口测试 ===")
    print(f"base_url: {REAL_API_BASE_URL}")
    print(f"endpoint: {ENDPOINT_PATH}")
    print(f"accountId: {PRICE_API_ACCOUNT_ID}")
    print("fixed_params: matchMethod=2, excludeExactMatch=false, checkState=1, belongDataPool=2")
    print(f"material_name: {args.material_name}")
    print(f"batch_materials: {material_names}")
    print(f"page_size: {min(args.page_size, DEFAULT_PAGE_SIZE)}")
    print()

    if material_names:
        return run_batch(args, material_names)

    try:
        if args.paginate:
            return run_paginated(args, args.material_name)

        result = request_page(args, args.material_name, page=1)
        print_single_result(result, args.material_name, raw=args.raw)
        if result.get("code") != 200:
            return 2
        return 0
    except requests.HTTPError as exc:
        response_text = exc.response.text[:1000] if exc.response is not None else ""
        print(f"HTTP 请求失败: {exc}", file=sys.stderr)
        if response_text:
            print(response_text, file=sys.stderr)
        return 3
    except requests.RequestException as exc:
        print(f"网络请求失败: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"脚本执行失败: {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
