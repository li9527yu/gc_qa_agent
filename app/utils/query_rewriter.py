"""
查询重写模块 - 用于多轮对话中的查询理解与重写

功能：将依赖上下文的用户问题（含代词、省略）重写为独立的、
      不依赖历史也能被理解的完整问题，以提升RAG检索效果。
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("easy_rag_api")


@dataclass
class RewriteResult:
    """查询重写结果"""
    original_question: str
    rewritten_question: str
    was_rewritten: bool  # 是否被重写
    reason: str  # 重写原因


class QueryRewriter:
    """
    查询重写器
    
    核心逻辑：
    1. 检测当前问题是否需要重写（含代词、省略主语等）
    2. 结合对话历史，使用LLM重写为独立完整的问题
    3. 如果无需重写，直接返回原问题
    """
    
    # 需要重写的信号词（代词、指示词等）
    COREFERENCE_SIGNALS = [
        "它", "这个", "那个", "这些", "那些", 
        "其", "此", "上述", "前面", "之前",
        "后者", "前者", "他们", "它们"
    ]
    
    # 省略主语的信号（以动词开头的问题）
    ELLIPSIS_SIGNALS = [
        "有哪些", "是什么", "怎么", "如何", "为什么",
        "在哪里", "多少钱", "多长时间", "需要", "可以"
    ]
    
    def __init__(self, llm_predictor):
        """
        初始化查询重写器
        
        Args:
            llm_predictor: LLM预测器实例，用于执行重写
        """
        self.llm = llm_predictor
        self.logger = logging.getLogger("easy_rag_api")
    
    def _needs_rewrite(self, question: str) -> tuple[bool, str]:
        """
        判断问题是否需要重写
        
        Args:
            question: 用户当前问题
            
        Returns:
            (是否需要重写, 原因)
        """
        # 检查代词
        for signal in self.COREFERENCE_SIGNALS:
            if signal in question:
                return True, f"包含指代词'{signal}'"
        
        # 检查是否是省略主语的问句（以某些动词/助词开头）
        for signal in self.ELLIPSIS_SIGNALS:
            if question.strip().startswith(signal):
                return True, f"以省略信号词'{signal}'开头"
        
        return False, "无需重写"
    
    def _format_history(self, messages: List[Dict], max_turns: int = 3) -> str:
        """
        格式化对话历史为文本
        
        Args:
            messages: 消息列表，每项为 {"role": "user"/"assistant", "content": "..."}
            max_turns: 最多包含的轮数
            
        Returns:
            格式化后的对话历史文本
        """
        if not messages:
            return "无历史对话"
        
        # 取最近的几轮对话
        recent = messages[-max_turns * 2:] if len(messages) > max_turns * 2 else messages
        
        lines = []
        for msg in recent:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")[:150]  # 截断避免过长
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def _build_rewrite_prompt(self, current_question: str, history: str) -> str:
        """
        构建查询重写Prompt
        
        Args:
            current_question: 当前问题
            history: 格式化的对话历史
            
        Returns:
            完整的Prompt文本
        """
        return f"""你是一个专业的查询重写助手。请将用户的当前问题结合对话历史，重写为一个独立完整的问题。

【任务说明】
用户正在进行多轮对话，当前问题可能包含代词（它、这个等）或省略了主语。你需要将其重写为不依赖上下文也能理解的完整问题。

【对话历史】
{history}

【当前问题】
{current_question}

【重写规则】
1. 将代词（它、这个、那个、其、此、上述等）替换为具体指代的实体名词
2. 补充省略的主语和宾语，使问题语义完整
3. 保留用户问题的原始意图和语气
4. 如果当前问题已经完整独立，直接返回原问题

【示例】
历史：
用户: 什么是工程造价？
助手: 工程造价是指在工程项目建设过程中所花费的全部费用。
当前问题：它有哪些组成部分？
重写后：工程造价有哪些组成部分？

【当前任务】
请只返回重写后的问题，不要添加任何解释、前缀或引号。

重写后的问题："""
    
    async def rewrite(
        self, 
        current_question: str, 
        messages: List[Dict],
        force_rewrite: bool = False
    ) -> RewriteResult:
        """
        执行查询重写
        
        Args:
            current_question: 用户当前问题
            messages: 对话历史消息列表
            force_rewrite: 是否强制重写（跳过检测）
            
        Returns:
            RewriteResult: 重写结果
        """
        # 如果历史太短，无需重写
        if len(messages) < 2 and not force_rewrite:
            return RewriteResult(
                original_question=current_question,
                rewritten_question=current_question,
                was_rewritten=False,
                reason="历史消息太少"
            )
        
        # 检测是否需要重写
        needs_rewrite, reason = self._needs_rewrite(current_question)
        
        if not needs_rewrite and not force_rewrite:
            self.logger.debug(f"查询无需重写: {current_question[:50]}... ({reason})")
            return RewriteResult(
                original_question=current_question,
                rewritten_question=current_question,
                was_rewritten=False,
                reason=reason
            )
        
        # 执行重写
        try:
            history = self._format_history(messages)
            prompt = self._build_rewrite_prompt(current_question, history)
            
            self.logger.info(f"开始查询重写: '{current_question[:50]}...' ({reason})")
            
            # 调用LLM重写
            rewritten = await self.llm.predict_prompt(
                prompt_template=prompt,
                variables={},  # Prompt已经完整构建
                max_tokens=200  # 重写问题通常不会太长
            )
            
            # 清理结果
            rewritten = rewritten.strip().strip('"').strip("'")
            
            # 验证重写结果
            if not rewritten or len(rewritten) < 3:
                self.logger.warning(f"重写结果异常，使用原问题: {rewritten}")
                return RewriteResult(
                    original_question=current_question,
                    rewritten_question=current_question,
                    was_rewritten=False,
                    reason="重写结果异常"
                )
            
            self.logger.info(f"查询重写完成: '{current_question[:50]}...' → '{rewritten[:50]}...'")
            
            return RewriteResult(
                original_question=current_question,
                rewritten_question=rewritten,
                was_rewritten=True,
                reason=reason
            )
            
        except Exception as e:
            self.logger.error(f"查询重写失败: {e}", exc_info=True)
            # 失败时返回原问题，保证服务可用
            return RewriteResult(
                original_question=current_question,
                rewritten_question=current_question,
                was_rewritten=False,
                reason=f"重写失败: {str(e)}"
            )
    
    async def rewrite_with_context(
        self,
        current_question: str,
        session_context: Dict[str, Any]
    ) -> RewriteResult:
        """
        基于会话上下文重写（适配ConversationSession）
        
        Args:
            current_question: 当前问题
            session_context: 会话上下文对象，需包含messages
            
        Returns:
            RewriteResult: 重写结果
        """
        messages = session_context.get("messages", [])
        return await self.rewrite(current_question, messages)


# 便捷函数：快速创建重写器
def create_query_rewriter(llm_predictor) -> QueryRewriter:
    """创建查询重写器实例"""
    return QueryRewriter(llm_predictor)
