#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试两个材料价格接口：
1. 厂家报价-模糊查询材料
2. 信息价-模糊查询材料

默认使用 signature 认证；信息价接口如后端要求 token，可通过 --token 额外传入。

示例：
python test_material_price_dual_api.py --channel factory --material-name 水泥
python test_material_price_dual_api.py --channel information --material-name 铝合金门窗型材 --release-department 深圳市造价站
python test_material_price_dual_api.py --channel both --material-name 水泥 --province 广东省 --city 深圳市
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


DEFAULT_TIMEOUT = 30
DEFAULT_PAGE_SIZE = 2000

ENDPOINTS: Dict[str, Dict[str, Any]] = {
    "factory": {
        "path": "/material/factoryRelatedMaterials/getRelatedMaterial",
        "module": "material",
        "service": "factoryRelatedMaterials",
        "operator": "getRelatedMaterial",
        "sample_fields": [
            "materialName",
            "materialModelSpec",
            "brand",
            "supplyName",
            "price",
            "unit",
            "province",
            "city",
            "releaseTime",
        ],
    },
    "information": {
        "path": "/material/infoRelatedMaterials/getRelatedMaterial",
        "module": "material",
        "service": "infoRelatedMaterials",
        "operator": "getRelatedMaterial",
        "sample_fields": [
            "materialName",
            "materialModelSpec",
            "releaseDepartment",
            "price",
            "unit",
            "province",
            "city",
            "releaseTime",
        ],
    },
}


def build_payload(args: argparse.Namespace, material_name: str, page: int, channel: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
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
        "province": args.province or None,
        "city": args.city or None,
        "provinceId": args.province_id or None,
        "cityId": args.city_id or None,
        "enterpriseId": None,
        "enterpriseName": None,
        "queryAccountId": None,
        "accountName": None,
        "domain": args.domain or None,
        "materialName": material_name or None,
        "materialCode": args.material_code or None,
        "materialModelSpec": args.material_model_spec or None,
        "minPrice": args.min_price,
        "maxPrice": args.max_price,
        "page": page,
        "pageSize": min(args.page_size, DEFAULT_PAGE_SIZE),
    }

    if channel == "factory":
        payload.update(
            {
                "releaseDepartment": args.release_department or None,
                "startReleaseDate": args.start_release_date or None,
                "endReleaseDate": args.end_release_date or None,
                "brand": args.brand or None,
                "supplyName": args.supply_name or None,
            }
        )
    elif channel == "information":
        payload.update(
            {
                "releaseDepartment": args.release_department or None,
                "startReleaseDate": args.start_release_date or None,
                "endReleaseDate": args.end_release_date or None,
            }
        )
    else:
        raise ValueError(f"不支持的 channel: {channel}")

    return payload


def build_headers(payload: Dict[str, Any], channel: str, token: str) -> Dict[str, str]:
    config = ENDPOINTS[channel]
    signature = generate_signature(
        secret_key=API_SECRET_KEY,
        params=payload,
        module=config["module"],
        service=config["service"],
        operator=config["operator"],
        debug=False,
    )
    headers = {
        "Content-Type": "application/json",
        "signature": signature,
    }
    if token:
        headers["token"] = token
    return headers


def request_page(
    args: argparse.Namespace,
    material_name: str,
    page: int,
    channel: str,
) -> Dict[str, Any]:
    payload = build_payload(args, material_name, page, channel)
    headers = build_headers(payload, channel, token=args.token)
    url = f"{REAL_API_BASE_URL}{ENDPOINTS[channel]['path']}"

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
        "channel": channel,
        "url": url,
        "page": page,
        "page_size": payload["pageSize"],
        "elapsed_ms": elapsed_ms,
        "payload": payload,
        "headers": {
            "Content-Type": headers["Content-Type"],
            "signature": headers.get("signature"),
            "token": headers.get("token"),
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
    for key in ("totalCount", "total"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    return None


def summarize_records(records: List[Dict[str, Any]], channel: str, limit: int = 3) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    sample_fields = ENDPOINTS[channel]["sample_fields"]
    for item in records[:limit]:
        summary.append({field: item.get(field) for field in sample_fields})
    return summary


def print_single_result(result: Dict[str, Any], material_name: str, channel: str, raw: bool) -> None:
    if raw:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    records = extract_records(result)
    total_count = extract_total_count(result)
    debug = result.get("_debug") or {}

    print("=== 接口可用性测试结果 ===")
    print(f"channel: {channel}")
    print(f"material_name: {material_name}")
    print("HTTP: 200")
    print(f"业务 code: {result.get('code')}")
    print(f"message: {result.get('message')}")
    print(f"page: {debug.get('page')}")
    print(f"page_size: {debug.get('page_size')}")
    print(f"elapsed_ms: {debug.get('elapsed_ms')}")
    print(f"total: {total_count}")
    print(f"current_page_records: {len(records)}")
    print("records_sample:")
    print(json.dumps(summarize_records(records, channel=channel), ensure_ascii=False, indent=2))


def run_paginated(args: argparse.Namespace, material_name: str, channel: str) -> int:
    all_records: List[Dict[str, Any]] = []
    total_count: Optional[int] = None

    for page in range(1, args.max_pages + 1):
        result = request_page(args, material_name, page, channel)
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
            f"[{channel}][page {page}] material={material_name} code={code} "
            f"records={len(records)} total={page_total_count} "
            f"elapsed_ms={debug.get('elapsed_ms')}"
        )

        if not records:
            break
        if len(records) < min(args.page_size, DEFAULT_PAGE_SIZE):
            break
        if total_count is not None and len(all_records) >= total_count:
            break

    print("\n=== 分页汇总 ===")
    print(f"channel: {channel}")
    print(f"material_name: {material_name}")
    print(f"accumulated_records: {len(all_records)}")
    print(f"reported_total: {total_count}")
    print("records_sample:")
    print(json.dumps(summarize_records(all_records, channel=channel), ensure_ascii=False, indent=2))
    return 0


def run_single_channel(args: argparse.Namespace, material_name: str, channel: str) -> int:
    try:
        if args.paginate:
            return run_paginated(args, material_name, channel)

        result = request_page(args, material_name, page=1, channel=channel)
        print_single_result(result, material_name, channel, raw=args.raw)
        if result.get("code") != 200:
            return 2
        return 0
    except requests.HTTPError as exc:
        response_text = exc.response.text[:1000] if exc.response is not None else ""
        print(f"[{channel}] HTTP 请求失败: {exc}", file=sys.stderr)
        if response_text:
            print(response_text, file=sys.stderr)
        return 3
    except requests.RequestException as exc:
        print(f"[{channel}] 网络请求失败: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"[{channel}] 脚本执行失败: {exc}", file=sys.stderr)
        return 5


def run_batch(args: argparse.Namespace, material_names: List[str], channels: List[str]) -> int:
    overall_code = 0
    for channel in channels:
        for material_name in material_names:
            print(f"\n===== 批量测试: channel={channel} material={material_name} =====")
            code = run_single_channel(args, material_name, channel)
            overall_code = max(overall_code, code)
    return overall_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试厂家报价和信息价两个材料价格接口"
    )
    parser.add_argument(
        "--channel",
        choices=["factory", "information", "both"],
        default="both",
        help="测试渠道",
    )
    parser.add_argument("--material-name", default="水泥", help="材料名称")
    parser.add_argument(
        "--batch-materials",
        default="",
        help="批量测试材料名，英文逗号分隔，例如: 水泥,电缆",
    )
    parser.add_argument("--province", default="", help="省份")
    parser.add_argument("--city", default="", help="城市")
    parser.add_argument("--province-id", default="", help="省份ID")
    parser.add_argument("--city-id", default="", help="城市ID")
    parser.add_argument("--material-model-spec", default="", help="规格型号")
    parser.add_argument("--material-code", default="", help="材料编码")
    parser.add_argument("--brand", default="", help="品牌，仅厂家报价接口使用")
    parser.add_argument("--supply-name", default="", help="供应商，仅厂家报价接口使用")
    parser.add_argument("--release-department", default="", help="发布单位/机构")
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
    parser.add_argument("--token", default="", help="可选 token，信息价接口如需 token 可传入")
    parser.add_argument("--raw", action="store_true", help="打印完整响应 JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    channels = ["factory", "information"] if args.channel == "both" else [args.channel]
    material_names = [
        item.strip() for item in args.batch_materials.split(",") if item.strip()
    ]

    print("=== 材料价格接口测试 ===")
    print(f"base_url: {REAL_API_BASE_URL}")
    print(f"channels: {channels}")
    print(f"accountId: {PRICE_API_ACCOUNT_ID}")
    print("fixed_params: matchMethod=2, excludeExactMatch=false, checkState=1, belongDataPool=2")
    print(f"material_name: {args.material_name}")
    print(f"batch_materials: {material_names}")
    print(f"page_size: {min(args.page_size, DEFAULT_PAGE_SIZE)}")
    print()

    if material_names:
        return run_batch(args, material_names, channels)

    overall_code = 0
    for channel in channels:
        if len(channels) > 1:
            print(f"\n===== 单次测试: channel={channel} =====")
        code = run_single_channel(args, args.material_name, channel)
        overall_code = max(overall_code, code)
    return overall_code


if __name__ == "__main__":
    raise SystemExit(main())
