"""
智能渠道推断模块
当用户未明确指定渠道时，基于多维度特征智能推断渠道类型
"""

import re
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("easy_rag_api")


class ChannelType(str, Enum):
    ZC_PRICE = "zc_price"
    MANUFACTURER_PRICE = "manufacturer_price"
    INFORMATION_PRICE = "information_price"
    UNKNOWN = "unknown"


@dataclass
class ChannelInferenceResult:
    """渠道推断结果"""
    channel: ChannelType
    confidence: float  # 0-1
    reason: str  # 推断理由（用于展示给用户）
    matched_keywords: List[str]  # 匹配到的关键词
    suggestion: Optional[str] = None  # 给用户的建议


class ChannelInferencer:
    """智能渠道推断器"""
    
    # 渠道特征词库（按权重排序）
    CHANNEL_KEYWORDS: Dict[ChannelType, List[Tuple[str, float]]] = {
        ChannelType.INFORMATION_PRICE: [
            # 高权重词
            ("信息价", 1.0), ("政府指导价", 1.0), ("造价站", 1.0), 
            ("官方价格", 0.9), ("定额", 0.9), ("清单计价", 0.9),
            # 中权重词
            ("指导价", 0.7), ("基准价", 0.7), ("信息价库", 0.7),
            ("GBT", 0.6), ("GB", 0.6), ("规范", 0.6), ("标准", 0.6),
            # 低权重词
            ("参考价", 0.4), ("市场价走势", 0.4),
        ],
        ChannelType.MANUFACTURER_PRICE: [
            # 高权重词
            ("厂商报价", 1.0), ("厂家报价", 1.0), ("品牌报价", 1.0),
            ("供应商", 0.9), ("经销商", 0.9), ("厂家直供", 0.9),
            # 中权重词
            ("询价", 0.7), ("报价单", 0.7), ("品牌", 0.6),
            # 常见品牌（作为强特征）
            ("三棵树", 0.8), ("宝钢", 0.8), ("鞍钢", 0.8), ("立邦", 0.8),
            ("海螺", 0.8), ("南方水泥", 0.8), ("金隅", 0.8),
            # 低权重词
            ("零售价", 0.4), ("批发价", 0.4),
        ],
        ChannelType.ZC_PRICE: [
            # 高权重词
            ("智诚", 1.0), ("智诚信息价", 1.0), ("第三方价格", 0.9),
            # 中权重词
            ("实际成交价", 0.7), ("市场行情", 0.7), ("行情价", 0.7),
            ("市场价", 0.6), ("市场均价", 0.6),
            # 低权重词
            ("行情", 0.4), ("走势", 0.4),
        ]
    }
    
    # 默认渠道（当无法推断时）
    DEFAULT_CHANNEL = ChannelType.INFORMATION_PRICE
    DEFAULT_CONFIDENCE = 0.5
    
    def __init__(self):
        self.logger = logging.getLogger("easy_rag_api")
    
    def infer(
        self, 
        question: str, 
        parsed_entities: Optional[Dict] = None
    ) -> ChannelInferenceResult:
        """
        智能推断渠道类型
        
        Args:
            question: 用户原始问题
            parsed_entities: 已解析的实体（可选）
        
        Returns:
            ChannelInferenceResult: 推断结果
        """
        question = question.lower()
        parsed_entities = parsed_entities or {}
        
        # 1. 首先尝试精确匹配
        exact_match = self._exact_match(question)
        if exact_match:
            return ChannelInferenceResult(
                channel=exact_match,
                confidence=1.0,
                reason=f"根据关键词「{self._get_matched_keyword(question, exact_match)}」明确识别",
                matched_keywords=[self._get_matched_keyword(question, exact_match)],
                suggestion=None
            )
        
        # 2. 特征词加权评分
        scores: Dict[ChannelType, float] = {ch: 0.0 for ch in ChannelType 
                                            if ch != ChannelType.UNKNOWN}
        matched_keywords: Dict[ChannelType, List[str]] = {ch: [] for ch in scores}
        
        for channel, keywords in self.CHANNEL_KEYWORDS.items():
            for keyword, weight in keywords:
                if keyword.lower() in question:
                    scores[channel] += weight
                    matched_keywords[channel].append(keyword)
        
        # 3. 基于实体特征推断
        entity_bonus, entity_reason = self._infer_from_entities(parsed_entities)
        if entity_bonus:
            for ch in scores:
                if ch in entity_bonus:
                    scores[ch] += entity_bonus[ch]
                    if entity_reason:
                        matched_keywords[ch].append(f"[实体特征]{entity_reason}")
        
        # 4. 选择最佳匹配
        if scores:
            best_channel = max(scores, key=scores.get)
            best_score = scores[best_channel]
            
            # 归一化置信度（最高分可能超过1，需要压缩到0-1）
            confidence = min(best_score / 1.5, 0.95) if best_score > 0 else 0
            
            if best_score > 0.3:  # 有明确特征
                return ChannelInferenceResult(
                    channel=best_channel,
                    confidence=confidence,
                    reason=f"根据关键词「{', '.join(matched_keywords[best_channel][:3])}」推断",
                    matched_keywords=matched_keywords[best_channel],
                    suggestion=None
                )
        
        # 5. 默认兜底策略
        return self._default_inference(parsed_entities, matched_keywords)
    
    def _exact_match(self, question: str) -> Optional[ChannelType]:
        """精确匹配渠道关键词"""
        # 信息价精确匹配
        if any(kw in question for kw in ["信息价", "造价站", "政府指导价"]):
            return ChannelType.INFORMATION_PRICE
        
        # 厂商报价精确匹配
        if any(kw in question for kw in ["厂商报价", "厂家报价", "品牌报价"]):
            return ChannelType.MANUFACTURER_PRICE
        
        # 智诚信息价精确匹配
        if "智诚" in question:
            return ChannelType.ZC_PRICE
        
        return None
    
    def _get_matched_keyword(self, question: str, channel: ChannelType) -> str:
        """获取匹配到的关键词（用于展示）"""
        keywords = {
            ChannelType.INFORMATION_PRICE: "信息价",
            ChannelType.MANUFACTURER_PRICE: "厂商报价",
            ChannelType.ZC_PRICE: "智诚",
        }
        return keywords.get(channel, "未知")
    
    def _infer_from_entities(
        self, 
        entities: Dict
    ) -> Tuple[Optional[Dict[ChannelType, float]], Optional[str]]:
        """
        基于实体特征推断渠道
        
        Returns:
            (bonus_scores, reason)
        """
        bonus = {}
        reason = None
        
        # 有品牌信息 → 倾向厂商报价
        if entities.get("brand"):
            bonus[ChannelType.MANUFACTURER_PRICE] = 0.5
            reason = "检测到品牌信息"
        
        # 有规格型号 + 材料名称 → 倾向信息价
        if entities.get("materialModelSpec") and entities.get("materialName"):
            bonus[ChannelType.INFORMATION_PRICE] = 0.2
        
        # 有时间范围（近1-3月）→ 倾向市场价/智诚
        if entities.get("startReleaseDate") or entities.get("endReleaseDate"):
            bonus[ChannelType.ZC_PRICE] = 0.15
            bonus[ChannelType.INFORMATION_PRICE] = 0.1
        
        return bonus if bonus else None, reason
    
    def _default_inference(
        self, 
        entities: Dict,
        matched_keywords: Dict[ChannelType, List[str]]
    ) -> ChannelInferenceResult:
        """
        默认兜底推断策略
        
        策略优先级：
        1. 有材料名称 → 默认信息价（最常见场景）
        2. 完全无信息 → 提示用户选择
        """
        if entities.get("materialName"):
            # 有材料名称，默认信息价
            return ChannelInferenceResult(
                channel=self.DEFAULT_CHANNEL,
                confidence=self.DEFAULT_CONFIDENCE,
                reason="未明确指定渠道，默认查询信息价（政府指导价）",
                matched_keywords=[],
                suggestion="如需查询厂商报价或智诚信息价，请明确说明"
            )
        else:
            # 完全无法推断
            return ChannelInferenceResult(
                channel=ChannelType.UNKNOWN,
                confidence=0.0,
                reason="无法识别查询渠道",
                matched_keywords=[],
                suggestion="请明确指定查询渠道：信息价、厂商报价、或智诚信息价"
            )
    
    def get_channel_suggestion_text(self, result: ChannelInferenceResult) -> str:
        """生成给用户看的渠道推断说明文本"""
        if result.confidence >= 1.0:
            return f"已识别渠道：{self._channel_display_name(result.channel)}"
        elif result.confidence >= 0.7:
            return f"推断渠道：{self._channel_display_name(result.channel)}（{result.reason}）"
        elif result.confidence > 0:
            return f"默认使用：{self._channel_display_name(result.channel)}，{result.suggestion or ''}"
        else:
            return result.suggestion or "请明确指定查询渠道"
    
    def _channel_display_name(self, channel: ChannelType) -> str:
        """获取渠道的显示名称"""
        names = {
            ChannelType.INFORMATION_PRICE: "信息价",
            ChannelType.MANUFACTURER_PRICE: "厂商报价",
            ChannelType.ZC_PRICE: "智诚信息价",
            ChannelType.UNKNOWN: "未知渠道"
        }
        return names.get(channel, "未知")
    
    def should_confirm(self, result: ChannelInferenceResult) -> bool:
        """判断是否需要用户确认渠道"""
        # 置信度低于0.7时建议确认
        return result.confidence < 0.7 and result.channel != ChannelType.UNKNOWN


# 全局单例
_channel_inferencer: Optional[ChannelInferencer] = None


def get_channel_inferencer() -> ChannelInferencer:
    """获取渠道推断器单例"""
    global _channel_inferencer
    if _channel_inferencer is None:
        _channel_inferencer = ChannelInferencer()
    return _channel_inferencer
