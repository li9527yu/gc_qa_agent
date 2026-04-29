from openai import OpenAI
from .llm_openai import openai_chat_by_api_as_astream, InferenceParams
from .utils.prompt_template import Price_Answer_TEMPLATE
from .config import OPENAI_API_BASE, OPENAI_API_KEY, LLM_MODEL_NAME, USE_LLM
import logging
import httpx
from jinja2 import Template
from typing import Optional

def build_template() -> Template:
    """
    使用 Jinja2 构建 RAG Prompt 模板
    优势：支持复杂逻辑、避免大括号转义冲突、更好的可维护性
    """
    template_str = """你是一个严格基于检索文档回答问题的智能助手。

【回答原则】
1. 只能依据 <Documents> 中能够明确支持的内容作答，不得补充训练知识、常识推断、行业经验或外部事实。
2. 如果文档仅支持部分问题，只回答被文档支持的部分，并明确说明其余部分文档未提供依据。
3. 如果文档不足以支撑答案，必须直接回答："【信息不足】提供的文档信息不足以回答该问题。"
4. 如果文档之间存在矛盾，请分别列出冲突内容，并说明来源，不得自行裁决哪一方正确。
5. 如果文档内容明显是 OCR 噪声、截图残片或与问题无关，忽略这部分内容，不要据此扩写答案。

【输出要求】
- 所有回答使用中文
- 不要提及“根据我的训练知识”“我认为”“通常来说”这类脱离文档依据的表述
- 回答中涉及结论时，优先使用“文档显示”“文档提到”“文档未提及”这类可追溯表述

<Documents>
{{ context | trim }}
</Documents>

用户问题：{{ query | trim }}

请先判断文档是否足以支持回答，再在文档支持的范围内作答。"""

    # 创建 Jinja2 Template 对象
    # trim 过滤器会自动去除首尾空白，避免文档或问题前后的换行影响格式
    return Template(template_str)


 
def build_template_backup():
    # prompt_template = "你是一个准确和可靠的人工智能助手，能够借助外部文档回答用户问题，请注意外部文档可能存在噪声事实性错误。" \
    #                   "如果文档中的信息包含了正确答案，你将进行准确的回答。"\
    #                   "如果文档中的信息不包含答案，你将生成\"文档信息不足，因此我无法基于提供的文档回答该问题。\"。" \
    #                   "如果部分文档中存在与事实不一致的错误，请先生成\"提供文档的文档存在事实性错误。\"，并生成正确答案。" \
    #                   "下面给定你相关外部文档，根据文档来回答用户问题。" \
    #                   "注意，所有对话请使用中文回答。\n---" \
    #                   "以下是外部文档：\n---" \
    #                     "{}\n" \
    #                     "用户问题：\n---" \
    #                     "{}\n"


    return prompt_template


class LLMPredictor:
    def __init__(self, model_name=None, logger=None):
        # 使用配置文件中的模型名称，如果没有指定的话
        self.model = model_name or LLM_MODEL_NAME
        self.logger = logger or logging.getLogger(__name__)
        self.prompt_template = build_template()
        
        # 启用OpenAI客户端的详细日志记录
        logging.basicConfig()
        httpx_logger = logging.getLogger("httpx")
        httpx_logger.setLevel(logging.DEBUG)
        
        # 使用配置文件中的 API 配置
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE,
            http_client=httpx.Client(
                headers={"User-Agent": "Easy-RAG/1.0"},
                timeout=30.0
            )
        )
        
        self.price_answer_template = Price_Answer_TEMPLATE
        self.inference_params = InferenceParams(
            temperature=0,  # 设置为0确保确定性输出
            max_tokens=None,
            top_p=1,        # 添加top_p=1确保确定性
            frequency_penalty=0,  # 添加频率惩罚为0
            presence_penalty=0,   # 添加存在惩罚为0
            seed=42        # 添加随机种子确保可重复性
        )
        
        # 记录当前使用的模型信息
        self.logger.info(f"🤖 初始化 LLM 客户端")
        self.logger.info(f"   模式: {USE_LLM}")
        self.logger.info(f"   Base URL: {OPENAI_API_BASE}")
        self.logger.info(f"   Model: {self.model}")

    def predict_prompt(
        self,
        prompt_template: str,
        variables: dict,
        guided_decoding=False,
        allowed_tokens=None,
        max_tokens=None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None
    ):
        """
        同步预测：一次性获取完整回答
        :param prompt_template: 提示词模板
        :param query: 用户查询
        :param guided_decoding: 是否启用guided decoding
        :param allowed_tokens: 允许的token列表
        :param max_tokens: 最大token数，默认为10，但对于JSON响应可以设置更大值
        :return: 模型响应
        """
        try:
            prompt = prompt_template.format(**variables)
        except KeyError as e:
            missing_key = e.args[0]
            raise ValueError(
                f"Prompt 模板缺少变量: {missing_key}, "
                f"已提供变量: {list(variables.keys())}"
            )
        
        # 构建基本请求参数
        request_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,  # 设置为0确保确定性输出
        }
        
        # 只有当max_tokens明确指定时才添加，让模型自己决定输出长度限制
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        
        logging.info(f"请求参数: {request_params}")

        # 检查是否启用了guided decoding
        guided_decoding_enabled = False
        
        # 如果启用guided decoding且提供了允许的token列表
        if guided_decoding and allowed_tokens and hasattr(self, 'tokenizer') and self.tokenizer:
            logit_bias = self.create_logit_bias(allowed_tokens)
            if logit_bias:
                request_params["logit_bias"] = logit_bias
                guided_decoding_enabled = True
                self.logger.info(f"尝试启用guided decoding，允许的token: {allowed_tokens}")

        client = self.client
        if timeout is not None or max_retries is not None:
            option_kwargs = {}
            if timeout is not None:
                option_kwargs["timeout"] = timeout
            if max_retries is not None:
                option_kwargs["max_retries"] = max_retries
            client = self.client.with_options(**option_kwargs)
        
        try:
            response = client.chat.completions.create(**request_params)
            
            result = response.choices[0].message.content.strip()
            
            # 如果启用了guided decoding，验证结果是否在允许的token列表中
            if guided_decoding_enabled and allowed_tokens and result not in allowed_tokens:
                self.logger.warning(f"模型输出 '{result}' 不在允许的token列表中，尝试提取匹配项")
                # 尝试从结果中提取匹配的token
                for token in allowed_tokens:
                    if token in result:
                        result = token
                        break
                else:
                    # 如果没有找到匹配的token，返回第一个允许的token作为默认值
                    self.logger.warning(f"无法提取匹配的token，使用默认值: {allowed_tokens[0]}")
                    result = allowed_tokens[0]
            
            self.logger.info(f"LLM API 同步调用响应: {result}")
            return result
        except Exception as e:
            # 如果错误是因为logit_bias不支持，尝试不使用logit_bias重新调用
            if "logit_bias" in str(e) or "Not implemented" in str(e):
                self.logger.warning(f"模型服务器不支持logit_bias，尝试不使用logit_bias重新调用: {e}")
                # 移除logit_bias参数
                if "logit_bias" in request_params:
                    del request_params["logit_bias"]
                
                try:
                    response = client.chat.completions.create(**request_params)
                    result = response.choices[0].message.content.strip()
                    logging.info(f"同步调用响应（无logit_bias）: {response}")
                    self.logger.info(f"LLM API 同步调用响应（无logit_bias）: {result}")
                    return result
                except Exception as retry_e:
                    self.logger.error(f"LLM API 同步调用重试失败: {retry_e}")
                    raise retry_e
            else:
                self.logger.error(f"LLM API 同步调用出错: {e}")
                raise e
    
    def validate_output(self, output, allowed_tokens, default_token=None):
        """
        验证模型输出是否在允许的token列表中，如果不是，则尝试提取或返回默认值
        :param output: 模型原始输出
        :param allowed_tokens: 允许的token列表
        :param default_token: 默认返回的token（如果未指定，则使用allowed_tokens的第一个）
        :return: 验证后的输出
        """
        if not output:
            return default_token or allowed_tokens[0]
            
        output = output.strip()
        
        # 如果输出已经在允许的token列表中，直接返回
        if output in allowed_tokens:
            self.logger.info(f"模型输出 '{output}' 在允许的token列表中")
            return output
        
        self.logger.warning(f"模型输出 '{output}' 不在允许的token列表中，尝试提取匹配项")
        
        # 尝试从输出中提取匹配的token
        for token in allowed_tokens:
            if token in output:
                self.logger.info(f"从输出中提取到匹配的token: {token}")
                return token
        
        # 如果没有找到匹配的token，返回默认值
        default = default_token or allowed_tokens[0]
        self.logger.warning(f"无法提取匹配的token，使用默认值: {default}")
        return default
    
    def predict_intent(self, prompt_template, query):
        """
        预测用户意图，使用后处理验证确保输出只能是预定义的标签之一
        :param prompt_template: 意图识别提示词模板
        :param query: 用户查询
        :return: 意图标签
        """
        # 定义允许的意图标签
        allowed_intents = ["price_recommendation", "knowledge_qa", "other", "dangerous_sql"]
        
        try:
            # 尝试使用guided decoding（如果支持）
            result = self.predict_prompt(prompt_template, query, guided_decoding=True, allowed_tokens=allowed_intents)
        except Exception as e:
            self.logger.warning(f"guided decoding失败，使用常规预测: {e}")
            # 如果guided decoding失败，回退到常规预测
            result = self.predict_prompt(prompt_template, query)
        
        # 使用后处理验证确保输出符合预期
        return self.validate_output(result, allowed_intents, "knowledge_qa")
    
    def predict_channel(self, prompt_template, query):
        """
        预测价格渠道，使用后处理验证确保输出只能是预定义的渠道之一
        :param prompt_template: 渠道识别提示词模板
        :param query: 用户查询
        :return: 渠道标签
        """
        # 定义允许的渠道标签
        allowed_channels = ["information_price", "manufacturer_price", "unknown"]
        
        try:
            # 尝试使用guided decoding（如果支持）
            result = self.predict_prompt(prompt_template, query, guided_decoding=True, allowed_tokens=allowed_channels)
        except Exception as e:
            self.logger.warning(f"guided decoding失败，使用常规预测: {e}")
            # 如果guided decoding失败，回退到常规预测
            result = self.predict_prompt(prompt_template, query)
        
        # 使用后处理验证确保输出符合预期
        return self.validate_output(result, allowed_channels, "information_price")
    
    # rag问答的prompt
    async def stream_predict(self, context, query,messages):
        """✅ 支持逐步输出回答内容的异步生成器"""
        self.logger.info(f"stream_predict 开始，context长度: {len(context)}, query: {query}")
        try:
            # prompt = self.prompt_template.format(context=context, query=query)
            prompt=self.prompt_template.render(
            context=context,
            query=query.strip()
        )


            self.logger.info(f"prompt构建成功，长度: {len(prompt)}")
        except Exception as e:
            self.logger.error(f"prompt构建失败: {e}")
            yield f"\n[ERROR] prompt构建失败: {str(e)}"
            return
            
        messages.append({"role": "user", "content": prompt})
        self.logger.info(f"开始调用 openai_chat_by_api_as_astream，messages数量: {len(messages)}")
        try:
            chunk_count = 0
            async for chunk in openai_chat_by_api_as_astream(
                model_name=self.model,
                messages=messages,
                inference_params=self.inference_params
            ):
                if chunk:
                    chunk_count += 1
                    yield chunk
            self.logger.info(f"openai_chat_by_api_as_astream 完成，共 {chunk_count} 个 chunk")
        except Exception as e:
            self.logger.error(f"LLM API 流式调用出错: {e}", exc_info=True)
            yield f"\n[ERROR] {str(e)}"
    
    # 专门用于价格推荐的流逝式接口
    async def stream_price_predict(self, parsed_entities, detail_answer,query,messages):
        """✅ 支持逐步输出回答内容的异步生成器"""
        prompt = self.price_answer_template.format(user_question=query, parsed_entities=parsed_entities, detail_answer=detail_answer)
        messages.append({"role": "user", "content": prompt})
        try:
            # 
            async for chunk in openai_chat_by_api_as_astream(
                model_name=self.model,
                inference_params=self.inference_params,
                messages=messages,
            ):
                if chunk:
                    yield chunk
        except Exception as e:
            self.logger.error(f"LLM API 流式调用出错: {e}")
            yield f"\n[ERROR] {str(e)}"
    
    # 支持上下文的流式回答
    async def predict(self, messages):
        """✅ 支持逐步输出回答内容的异步生成器"""
        try:
            async for chunk in openai_chat_by_api_as_astream(
                model_name=self.model,
                messages=messages,
                inference_params=self.inference_params
            ):
                if chunk:
                    yield chunk
        except Exception as e:
            self.logger.error(f"LLM API 流式调用出错: {e}")
            yield f"\n[ERROR] {str(e)}"
    
    def predict_with_validation(self, template, query, allowed_tokens, default_token):
        """
        使用后处理验证的预测方法
        :param template: 提示词模板
        :param query: 用户查询
        :param allowed_tokens: 允许的输出标签列表
        :param default_token: 默认返回标签
        :return: 验证后的输出标签
        """
        try:
            # 使用guided decoding预测
            result = self.predict_prompt(template, query, guided_decoding=True, allowed_tokens=allowed_tokens)
            return result
        except Exception as e:
            self.logger.error(f"Guided decoding失败: {e}")
            # 回退到常规预测
            result = self.predict_prompt(template, query)
            # 验证输出
            return self.validate_output(result, allowed_tokens, default_token)
    
    def validate_output(self, result, allowed_tokens, default_token):
        """
        验证输出是否在允许的标签列表中
        :param result: 预测结果
        :param allowed_tokens: 允许的标签列表
        :param default_token: 默认标签
        :return: 验证后的标签
        """
        if result in allowed_tokens:
            return result
        else:
            self.logger.warning(f"输出标签 '{result}' 不在允许的列表中，使用默认标签 '{default_token}'")
            return default_token
    
    def predict_channel(self, query):
        """
        预测价格渠道，使用guided decoding和后处理验证确保输出为预定义的渠道标签
        :param query: 用户查询
        :return: 渠道标签
        """
        # 定义允许的渠道标签
        allowed_channels = ["information_price", "manufacturer_price", "unknown"]
        
        # 使用后处理验证的预测方法
        return self.predict_with_validation(
            Price_Channel_TEMPLATE, 
            query, 
            allowed_channels, 
            default_token="information_price"
        )
    
    def validate_output(self, output, allowed_tokens, default_token=None):
        """
        验证模型输出是否在允许的token列表中，如果不是，则尝试提取或返回默认值
        :param output: 模型原始输出
        :param allowed_tokens: 允许的token列表
        :param default_token: 默认返回的token（如果未指定，则使用allowed_tokens的第一个）
        :return: 验证后的输出
        """
        if not output:
            return default_token or allowed_tokens[0]
            
        output = output.strip()
        
        # 如果输出已经在允许的token列表中，直接返回
        if output in allowed_tokens:
            self.logger.info(f"模型输出 '{output}' 在允许的token列表中")
            return output
        
        self.logger.warning(f"模型输出 '{output}' 不在允许的token列表中，尝试提取匹配项")
        
        # 尝试从输出中提取匹配的token
        for token in allowed_tokens:
            if token in output:
                self.logger.info(f"从输出中提取到匹配的token: {token}")
                return token
        
        # 尝试模糊匹配（去除下划线、转换为小写等）
        output_lower = output.lower().replace("_", "").replace("-", "")
        for token in allowed_tokens:
            token_normalized = token.lower().replace("_", "").replace("-", "")
            if token_normalized in output_lower or output_lower in token_normalized:
                self.logger.info(f"通过模糊匹配提取到token: {token}")
                return token
        
        # 如果没有找到匹配的token，返回默认值
        default = default_token or allowed_tokens[0]
        self.logger.warning(f"无法提取匹配的token，使用默认值: {default}")
        return default
    
    def predict_with_validation(self, prompt_template, query, allowed_tokens, default_token=None, max_retries=2):
        """
        使用后处理验证的预测方法，确保输出在允许的token列表中
        :param prompt_template: 提示词模板
        :param query: 用户查询
        :param allowed_tokens: 允许的token列表
        :param default_token: 默认返回的token
        :param max_retries: 最大重试次数
        :return: 验证后的输出
        """
        for attempt in range(max_retries + 1):
            try:
                # 尝试使用guided decoding（如果支持）
                result = self.predict_prompt(prompt_template, query, guided_decoding=True, allowed_tokens=allowed_tokens)
                
                # 验证结果
                validated_result = self.validate_output(result, allowed_tokens, default_token)
                
                # 如果验证成功，返回结果
                if validated_result in allowed_tokens:
                    return validated_result
                    
            except Exception as e:
                self.logger.warning(f"预测尝试 {attempt + 1} 失败: {e}")
                
                # 如果不是最后一次尝试，继续重试
                if attempt < max_retries:
                    continue
                    
                # 如果是最后一次尝试，使用默认值
                self.logger.error(f"所有预测尝试均失败，使用默认值: {default_token or allowed_tokens[0]}")
                return default_token or allowed_tokens[0]
