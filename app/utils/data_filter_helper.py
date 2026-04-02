"""
数据量控制和筛选辅助模块
用于处理大规模价格数据的交互式筛选
"""
import json
from typing import Dict, Any, Optional, List
from enum import Enum


class DataVolumeLevel(str, Enum):
    """数据量级别"""
    OPTIMAL = "optimal"          # 最优：≤ 100 条
    ACCEPTABLE = "acceptable"    # 可接受：100-1000 条
    TOO_LARGE = "too_large"      # 过大：1000-10000 条
    UNACCEPTABLE = "unacceptable"  # 不可接受：> 10000 条


class TokenEstimator:
    """Token 估算器"""
    
    # 单条价格记录的平均 token 数
    TOKENS_PER_RECORD = 120
    
    # DeepSeek 的限制
    MAX_CONTEXT_TOKENS = 128000
    
    # 安全边界（保留空间给系统提示词和用户问题）
    SAFE_CONTEXT_TOKENS = 100000
    
    @classmethod
    def estimate_tokens(cls, record_count: int) -> int:
        """估算给定记录数的 token 消耗"""
        return record_count * cls.TOKENS_PER_RECORD
    
    @classmethod
    def max_safe_records(cls) -> int:
        """返回安全的最大记录数"""
        return cls.SAFE_CONTEXT_TOKENS // cls.TOKENS_PER_RECORD
    
    @classmethod
    def get_volume_level(cls, record_count: int) -> DataVolumeLevel:
        """判断数据量级别"""
        if record_count <= 100:
            return DataVolumeLevel.OPTIMAL
        elif record_count <= 1000:
            return DataVolumeLevel.ACCEPTABLE
        elif record_count <= 10000:
            return DataVolumeLevel.TOO_LARGE
        else:
            return DataVolumeLevel.UNACCEPTABLE
    
    @classmethod
    def can_process(cls, record_count: int) -> bool:
        """判断是否可以直接处理"""
        return record_count <= 1000  # 推荐阈值


class DataFilterGuide:
    """数据筛选引导器"""
    
    def __init__(self):
        self.filter_prompts = {
            "materialName": "材料名称（如：水泥、钢筋、混凝土等）",
            "materialModelSpec": "规格型号（如：P.O 42.5、HRB400 等）",
            "province": "省份（如：江苏省、浙江省等）",
            "city": "城市（如：苏州市、南京市等）",
            "minPrice": "最低价格（元）",
            "maxPrice": "最高价格（元）",
            "startReleaseDate": "开始时间（格式：YYYY-MM-DD）",
            "endReleaseDate": "结束时间（格式：YYYY-MM-DD）",
            "brand": "品牌（厂商报价专用）",
            "releaseDepartment": "发布部门"
        }
    
    def generate_refinement_message(
        self, 
        current_count: int, 
        target_count: int,
        current_entities: Dict[str, Any],
        channel: str
    ) -> str:
        """生成数据筛选引导消息"""
        
        # 计算需要缩减的比例
        reduction_ratio = (current_count - target_count) / current_count * 100
        
        # 已经提供的筛选条件
        provided_filters = []
        for key, value in current_entities.items():
            if value and value != "" and key in self.filter_prompts:
                provided_filters.append(f"✅ {self.filter_prompts[key]}: {value}")
        
        # 可以补充的筛选条件
        available_filters = []
        for key, label in self.filter_prompts.items():
            if key not in current_entities or not current_entities.get(key):
                # 根据渠道过滤不适用的字段
                if channel == "manufacturer_price" or key != "brand":
                    available_filters.append(f"  - {label}")
        
        # 估算 token
        estimated_tokens = TokenEstimator.estimate_tokens(current_count)
        max_tokens = TokenEstimator.SAFE_CONTEXT_TOKENS
        
        message = f"""
📊 **数据量过大提示**

当前查询到 **{current_count:,}** 条价格记录，预计消耗 **{estimated_tokens:,}** tokens（超出限制 {max_tokens:,} tokens）。

为了提供精准的价格推荐，建议您补充以下筛选条件，将数据量缩减至 **{target_count}** 条以内：

**当前已提供的筛选条件：**
{chr(10).join(provided_filters) if provided_filters else "  ⚠️ 暂无具体筛选条件"}

**建议补充以下条件（任选其一或多个）：**
{chr(10).join(available_filters[:5])}

---

💡 **示例问题：**
- "请帮我查询 **江苏省苏州市** 的 {current_entities.get('materialName', '该材料')} 价格"
- "我需要规格为 **P.O 42.5** 的 {current_entities.get('materialName', '水泥')} 在 **2024年** 的价格"
- "价格范围在 **400-600元** 的 {current_entities.get('materialName', '该材料')}"

请提供更具体的筛选条件，我将为您重新查询。
        """.strip()
        
        return message
    
    def suggest_next_filter(self, current_entities: Dict[str, Any]) -> List[str]:
        """建议下一个应该添加的筛选条件（按优先级排序）"""
        priority_order = [
            "materialName",      # 最重要
            "province",
            "city", 
            "materialModelSpec",
            "startReleaseDate",
            "endReleaseDate",
            "minPrice",
            "maxPrice"
        ]
        
        missing_filters = []
        for key in priority_order:
            if key not in current_entities or not current_entities.get(key):
                missing_filters.append(key)
        
        return missing_filters[:3]  # 返回前3个建议


class DataFilterValidator:
    """数据筛选验证器"""
    
    @staticmethod
    def validate_price_range(min_price: Optional[float], max_price: Optional[float]) -> bool:
        """验证价格范围是否合理"""
        if min_price is None or max_price is None:
            return True
        return 0 <= min_price < max_price
    
    @staticmethod
    def validate_date_range(start_date: Optional[str], end_date: Optional[str]) -> bool:
        """验证日期范围是否合理"""
        if not start_date or not end_date:
            return True
        try:
            from datetime import datetime
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            return start < end
        except:
            return False
    
    @staticmethod
    def estimate_reduction_impact(
        current_entities: Dict[str, Any],
        new_filter_key: str,
        new_filter_value: Any
    ) -> float:
        """估算添加新筛选条件后的数据缩减比例（启发式）"""
        reduction_factors = {
            "materialName": 0.01,      # 材料名称：缩减到 1%
            "materialModelSpec": 0.10,  # 规格型号：缩减到 10%
            "province": 0.05,           # 省份：缩减到 5%
            "city": 0.02,               # 城市：缩减到 2%
            "minPrice": 0.30,           # 价格范围：缩减到 30%
            "maxPrice": 0.30,
            "startReleaseDate": 0.20,   # 时间范围：缩减到 20%
            "brand": 0.05               # 品牌：缩减到 5%
        }
        return reduction_factors.get(new_filter_key, 0.50)


# 便捷函数
def check_data_volume_and_guide(
    record_count: int,
    current_entities: Dict[str, Any],
    channel: str = "information_price"
) -> Dict[str, Any]:
    """
    检查数据量并生成引导信息
    
    返回：
    {
        "can_process": bool,
        "volume_level": str,
        "estimated_tokens": int,
        "guidance_message": str (如果需要筛选),
        "suggested_filters": list
    }
    """
    estimator = TokenEstimator()
    guide = DataFilterGuide()
    
    volume_level = estimator.get_volume_level(record_count)
    can_process = estimator.can_process(record_count)
    estimated_tokens = estimator.estimate_tokens(record_count)
    
    result = {
        "can_process": can_process,
        "volume_level": volume_level.value,
        "estimated_tokens": estimated_tokens,
        "current_count": record_count,
        "max_safe_count": estimator.max_safe_records()
    }
    
    if not can_process:
        target_count = 100  # 目标数据量
        result["guidance_message"] = guide.generate_refinement_message(
            current_count=record_count,
            target_count=target_count,
            current_entities=current_entities,
            channel=channel
        )
        result["suggested_filters"] = guide.suggest_next_filter(current_entities)
    
    return result


if __name__ == "__main__":
    # 测试代码
    print("=== Token 估算测试 ===")
    for count in [100, 1000, 10000, 100000, 1000000]:
        tokens = TokenEstimator.estimate_tokens(count)
        level = TokenEstimator.get_volume_level(count)
        can_process = TokenEstimator.can_process(count)
        print(f"{count:,} 条记录 → {tokens:,} tokens → {level.value} → 可处理: {can_process}")
    
    print("\n=== 筛选引导测试 ===")
    test_entities = {
        "materialName": "水泥"
    }
    result = check_data_volume_and_guide(
        record_count=50000,
        current_entities=test_entities,
        channel="information_price"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))



