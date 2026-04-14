"""
MCP 工具定义 - 材料价格查询相关工具
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger("easy_rag_api")


class ToolStatus(str, Enum):
    """工具执行状态"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"  # 部分成功（如数据量过大）


@dataclass
class ToolResult:
    """工具执行结果"""
    status: ToolStatus
    data: Any = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    suggested_next_steps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "data": self.data,
            "message": self.message,
            "metadata": self.metadata,
            "suggested_next_steps": self.suggested_next_steps
        }


@dataclass
class ToolDefinition:
    """工具定义（用于MCP协议）"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]
    examples: List[Dict[str, Any]] = field(default_factory=list)


class MaterialPriceTools:
    """
    材料价格查询工具集合
    封装所有与价格查询相关的操作，提供MCP标准接口
    """
    
    def __init__(self, rag_service):
        """
        初始化工具集
        
        Args:
            rag_service: RAGService 实例
        """
        self.rag_service = rag_service
        self.logger = logging.getLogger("easy_rag_api")
    
    # ============================================================================
    # 工具定义（供MCP Server注册）
    # ============================================================================
    
    @property
    def available_tools(self) -> List[ToolDefinition]:
        """获取所有可用工具定义"""
        return [
            self._define_quick_search_tool(),
            self._define_detailed_query_tool(),
            self._define_analyze_prices_tool(),
            self._define_get_candidates_tool(),
            self._define_extract_entities_tool(),
            self._define_identify_channel_tool(),
        ]
    
    def _define_quick_search_tool(self) -> ToolDefinition:
        """定义快速搜索工具"""
        return ToolDefinition(
            name="quick_search_materials",
            description="快速搜索材料，返回匹配的材料名称列表和相关关键词建议",
            parameters={
                "keyword": {
                    "type": "string",
                    "description": "用户输入的关键词（如：钢筋、水泥、铝合金）"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量上限",
                    "default": 10
                }
            },
            required=["keyword"],
            examples=[
                {"keyword": "钢筋", "limit": 5},
                {"keyword": "混凝土", "limit": 10}
            ]
        )
    
    def _define_detailed_query_tool(self) -> ToolDefinition:
        """定义详细查询工具"""
        return ToolDefinition(
            name="query_price_data",
            description="根据完整实体条件查询价格数据",
            parameters={
                "channel": {
                    "type": "string",
                    "description": "查询渠道：information_price(信息价)/manufacturer_price(厂商价)/zc_price(智诚价)",
                    "enum": ["information_price", "manufacturer_price", "zc_price"]
                },
                "material_name": {
                    "type": "string",
                    "description": "材料名称"
                },
                "province": {
                    "type": "string",
                    "description": "省份（可选）"
                },
                "city": {
                    "type": "string",
                    "description": "城市（可选）"
                },
                "material_model_spec": {
                    "type": "string",
                    "description": "规格型号（可选）"
                },
                "brand": {
                    "type": "string",
                    "description": "品牌（可选，厂商价渠道有效）"
                }
            },
            required=["channel", "material_name"],
            examples=[
                {
                    "channel": "information_price",
                    "material_name": "钢筋",
                    "province": "广东省",
                    "city": "深圳市"
                }
            ]
        )
    
    def _define_analyze_prices_tool(self) -> ToolDefinition:
        """定义价格分析工具"""
        return ToolDefinition(
            name="analyze_prices",
            description="对价格数据进行统计分析，包括K-means聚类、价格区间、异常值检测等",
            parameters={
                "price_data": {
                    "type": "array",
                    "description": "价格数据列表，每项包含price字段"
                },
                "unit": {
                    "type": "string",
                    "description": "计量单位（用于分组分析）",
                    "default": "auto"
                }
            },
            required=["price_data"],
            examples=[
                {
                    "price_data": [{"price": 100}, {"price": 105}, {"price": 98}],
                    "unit": "auto"
                }
            ]
        )
    
    def _define_get_candidates_tool(self) -> ToolDefinition:
        """定义获取候选材料工具"""
        return ToolDefinition(
            name="get_material_candidates",
            description="获取与关键词相关的候选材料名称列表",
            parameters={
                "keyword": {
                    "type": "string",
                    "description": "用户输入的关键词"
                },
                "channel": {
                    "type": "string",
                    "description": "查询渠道（可选，用于精确匹配）",
                    "enum": ["information_price", "manufacturer_price", "zc_price"]
                },
                "limit": {
                    "type": "integer",
                    "description": "返回数量上限",
                    "default": 10
                }
            },
            required=["keyword"],
            examples=[
                {"keyword": "混凝土", "limit": 10}
            ]
        )
    
    def _define_extract_entities_tool(self) -> ToolDefinition:
        """定义实体提取工具"""
        return ToolDefinition(
            name="extract_entities",
            description="从用户问题中提取价格查询所需的实体信息",
            parameters={
                "question": {
                    "type": "string",
                    "description": "用户的问题文本"
                }
            },
            required=["question"],
            examples=[
                {"question": "查一下广东省深圳市的钢筋价格"}
            ]
        )
    
    def _define_identify_channel_tool(self) -> ToolDefinition:
        """定义渠道识别工具"""
        return ToolDefinition(
            name="identify_channel",
            description="识别用户查询的价格渠道类型",
            parameters={
                "question": {
                    "type": "string",
                    "description": "用户的问题文本"
                }
            },
            required=["question"],
            examples=[
                {"question": "从信息价查钢筋价格"}
            ]
        )
    
    # ============================================================================
    # 工具实现
    # ============================================================================
    
    async def quick_search_materials(
        self, 
        keyword: str, 
        limit: int = 10
    ) -> ToolResult:
        """
        快速搜索材料 - 用于首次查询，返回候选材料和常见统计信息
        
        Args:
            keyword: 用户输入的关键词
            limit: 返回结果数量上限
            
        Returns:
            ToolResult: 包含匹配的材料名称、常见省份、常见规格和相关关键词建议
        """
        try:
            self.logger.info(f"[MCP Tool] quick_search_materials: keyword={keyword}, limit={limit}")
            
            # 1. 提取可能的实体（用于后续精确查询）
            entities = await self.rag_service.extract_entities(keyword)
            
            # 2. 获取候选材料名称
            candidates = await self._get_material_name_candidates(keyword, limit)
            
            # 3. 基于候选获取统计数据（省份、规格等）
            common_provinces = []
            common_specs = []
            if candidates:
                # 用第一个最相关的候选去查样本数据
                top_name = candidates[0]["name"]
                sample_data = await self._fetch_sample_for_stats(top_name, limit=100)
                from collections import Counter
                provinces = [item.get("province") for item in sample_data if item.get("province")]
                specs = [item.get("materialModelSpec") for item in sample_data if item.get("materialModelSpec")]
                common_provinces = [p for p, _ in Counter(provinces).most_common(5)]
                common_specs = [s for s, _ in Counter(specs).most_common(5)]
            
            # 4. 生成相关关键词建议
            related_keywords = self._generate_related_keywords(keyword, candidates)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "keyword": keyword,
                    "extracted_entities": entities,
                    "candidates": candidates,
                    "common_provinces": common_provinces,
                    "common_specs": common_specs,
                    "related_keywords": related_keywords
                },
                message=f"找到 {len(candidates)} 个相关材料，常见省份：{', '.join(common_provinces) if common_provinces else '无'}",
                metadata={
                    "total_candidates": len(candidates),
                    "has_exact_match": keyword in [c["name"] for c in candidates]
                },
                suggested_next_steps=[
                    "如果材料名称正确，请补充省份和城市信息",
                    "如需查询具体价格，请调用 query_price_data"
                ]
            )
            
        except Exception as e:
            self.logger.error(f"快速搜索失败: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                message=f"搜索失败: {str(e)}",
                suggested_next_steps=["请尝试更具体的关键词"]
            )
    
    async def query_price_data(
        self,
        channel: str,
        material_name: str,
        province: Optional[str] = None,
        city: Optional[str] = None,
        material_model_spec: Optional[str] = None,
        brand: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        查询价格数据
        
        Args:
            channel: 查询渠道
            material_name: 材料名称
            province: 省份
            city: 城市
            material_model_spec: 规格型号
            brand: 品牌
            
        Returns:
            ToolResult: 包含价格数据和分析结果
        """
        try:
            self.logger.info(f"[MCP Tool] query_price_data: channel={channel}, material={material_name}")
            
            # 构建查询实体
            entities = {
                "materialName": material_name,
                "province": province or "",
                "city": city or "",
                "materialModelSpec": material_model_spec or "",
                "brand": brand or ""
            }
            
            # 调用价格查询
            result = await self.rag_service.process_price_recommendation(channel, entities)
            
            if not result.get("success"):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    message=result.get("error_message", "查询失败"),
                    data=result
                )
            
            total_count = result.get("total_count", 0)
            
            # 根据数据量决定下一步建议
            suggested_steps = []
            if total_count == 0:
                suggested_steps.append("未查询到数据，请尝试放宽条件或使用其他材料名称")
            elif total_count > 1000:
                suggested_steps.append("数据量较大，建议补充更多筛选条件（如规格、城市）")
            else:
                suggested_steps.append("数据查询成功，可以调用 analyze_prices 进行深度分析")
            
            return ToolResult(
                status=ToolStatus.SUCCESS if total_count > 0 else ToolStatus.PARTIAL,
                data=result,
                message=f"查询到 {total_count} 条价格数据",
                metadata={
                    "channel": channel,
                    "total_count": total_count,
                    "entities": entities
                },
                suggested_next_steps=suggested_steps
            )
            
        except Exception as e:
            self.logger.error(f"价格查询失败: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                message=f"查询失败: {str(e)}"
            )
    
    async def analyze_prices(
        self,
        price_data: List[Dict[str, Any]],
        unit: str = "auto"
    ) -> ToolResult:
        """
        分析价格数据
        
        Args:
            price_data: 价格数据列表
            unit: 计量单位
            
        Returns:
            ToolResult: 包含统计分析结果
        """
        try:
            self.logger.info(f"[MCP Tool] analyze_prices: data_count={len(price_data)}")
            
            # 导入分析工具
            from app.utils.price_tools import analyze_by_unit
            
            # 执行分析
            analysis_result = analyze_by_unit(price_data, unit_col="unit")
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=analysis_result,
                message="价格分析完成",
                metadata={
                    "most_common_unit": analysis_result.get("most_common_unit"),
                    "unit_count": len(analysis_result.get("results", {}))
                },
                suggested_next_steps=["分析完成，可以生成可视化图表或价格推荐"]
            )
            
        except Exception as e:
            self.logger.error(f"价格分析失败: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                message=f"分析失败: {str(e)}"
            )
    
    async def get_material_candidates(
        self,
        keyword: str,
        channel: Optional[str] = None,
        limit: int = 10
    ) -> ToolResult:
        """
        获取候选材料名称
        
        Args:
            keyword: 关键词
            channel: 查询渠道（可选）
            limit: 返回数量上限
            
        Returns:
            ToolResult: 包含候选材料列表
        """
        try:
            candidates = await self._get_material_name_candidates(keyword, limit, channel)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=candidates,
                message=f"找到 {len(candidates)} 个候选材料",
                suggested_next_steps=["请从候选列表中选择最匹配的材料名称"]
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                message=f"获取候选材料失败: {str(e)}"
            )
    
    async def extract_entities(self, question: str) -> ToolResult:
        """
        从问题中提取实体
        
        Args:
            question: 用户问题
            
        Returns:
            ToolResult: 包含提取的实体
        """
        try:
            entities = await self.rag_service.extract_entities(question)
            
            # 检查缺失的字段
            required_fields = ["materialName", "province", "city"]
            missing = [f for f in required_fields if not entities.get(f)]
            # 如果用户没有明确指定渠道，也建议补充
            if not entities.get("channel"):
                missing.append("channel")
            
            suggested_steps = []
            if missing:
                missing_labels = []
                for f in missing:
                    if f == "channel":
                        missing_labels.append("查询渠道（信息价/厂商报价）")
                    else:
                        missing_labels.append(f)
                suggested_steps.append(f"还需要补充以下信息: {', '.join(missing_labels)}")
            else:
                suggested_steps.append("实体信息完整，可以进行价格查询")
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=entities,
                message="实体提取完成",
                metadata={
                    "extracted_fields": list(entities.keys()),
                    "missing_fields": missing
                },
                suggested_next_steps=suggested_steps
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                message=f"实体提取失败: {str(e)}"
            )
    
    async def identify_channel(self, question: str) -> ToolResult:
        """
        识别查询渠道
        
        Args:
            question: 用户问题
            
        Returns:
            ToolResult: 包含渠道识别结果
        """
        try:
            channel_result, inference_detail = await self.rag_service.identify_channel(question)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data={
                    "channel": channel_result.value,
                    "confidence": inference_detail.confidence,
                    "reason": inference_detail.reason,
                    "matched_keywords": inference_detail.matched_keywords
                },
                message=f"识别到渠道: {channel_result.value}",
                suggested_next_steps=["渠道识别完成，可以进行价格查询"]
            )
            
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                message=f"渠道识别失败: {str(e)}"
            )
    
    # ============================================================================
    # 辅助方法
    # ============================================================================
    
    async def _get_material_name_candidates(
        self, 
        keyword: str, 
        limit: int = 10,
        channel: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取材料名称候选列表
        
        这里可以实现：
        1. 从API获取候选
        2. 使用模糊匹配
        3. 使用向量检索相似材料名
        """
        # 简化实现：调用API获取数据后提取材料名
        try:
            # 先尝试精确匹配
            from app.utils.fillter_tools import filter_items
            
            # 构建基础查询实体
            entities = {"materialName": keyword}
            
            # 尝试从各渠道获取数据
            candidates = []
            channels = [channel] if channel else ["information_price", "manufacturer_price", "zc_price"]
            
            material_names = set()
            for ch in channels:
                try:
                    result = await self._quick_query(ch, entities, return_limit=50)
                    if result and result.get("success"):
                        for item in result.get("data", []):
                            name = item.get("materialName", "")
                            if name and keyword.lower() in name.lower():
                                material_names.add(name)
                except Exception as e:
                    self.logger.warning(f"从 {ch} 获取候选失败: {e}")
            
            # 格式化候选列表
            for name in list(material_names)[:limit]:
                score = 1.0 if name == keyword else 0.8 if name.startswith(keyword) else 0.6
                candidates.append({
                    "name": name,
                    "match_score": score,
                    "exact_match": name == keyword
                })
            
            # 按匹配分数排序
            candidates.sort(key=lambda x: x["match_score"], reverse=True)
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"获取候选材料失败: {e}")
            return []
    
    async def _fetch_sample_for_stats(
        self,
        material_name: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取样本数据用于统计分析常见省份和规格"""
        try:
            # 使用默认信息价渠道，只查材料名，不限制其他条件
            entities = {"materialName": material_name, "returnNumber": limit}
            result = await asyncio.to_thread(
                self.rag_service._get_infor_material, entities
            )
            if result and result.get("code") == 200:
                return result.get("data", {}).get("list", [])[:limit]
            return []
        except Exception as e:
            self.logger.warning(f"获取样本统计失败: {e}")
            return []

    async def _quick_query(
        self, 
        channel: str, 
        entities: Dict[str, str],
        return_limit: int = 10
    ) -> Optional[Dict]:
        """快速查询（只返回少量数据用于候选提取）"""
        try:
            # 修改实体，限制返回数量
            query_entities = entities.copy()
            query_entities["returnNumber"] = return_limit
            
            result = await self.rag_service.process_price_recommendation(channel, query_entities)
            return result
        except Exception as e:
            self.logger.error(f"快速查询失败: {e}")
            return None
    
    def _generate_related_keywords(
        self, 
        keyword: str, 
        candidates: List[Dict[str, Any]]
    ) -> List[str]:
        """生成相关关键词建议"""
        related = set()
        
        # 从候选中提取常见前缀/后缀
        for cand in candidates[:5]:
            name = cand["name"]
            if keyword in name and name != keyword:
                # 提取关键词以外的部分作为相关词
                remaining = name.replace(keyword, "").strip()
                if remaining:
                    related.add(remaining)
        
        return list(related)[:5]
