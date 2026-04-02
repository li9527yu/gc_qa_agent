import openai
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion
from openai._streaming import Stream,AsyncStream
from dataclasses import dataclass,asdict
from typing import (
    List,
    Optional,
    Union,
    Dict,
    Tuple,
    Literal,
    Iterable
)
import logging
import httpx

from .config import (
    OPENAI_API_BASE,
    OPENAI_API_KEY
)

@dataclass
class InferenceParams:
    temperature:Optional[float]=0.7
    max_tokens:Optional[int]=None
    top_p:Optional[float]=1.0
    frequency_penalty:Optional[float]=0.0
    presence_penalty:Optional[float]=0.0
    seed:Optional[int]=None
    

def openai_chat_by_api(
    model_name:Optional[str]="gpt-3.5-turbo",
    messages:Iterable[ChatCompletionMessageParam]=[], 
    inference_params:Union[InferenceParams,Dict]={}
):
    # 构建客户端
    client=openai.Client(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE
    )
    if isinstance(inference_params,InferenceParams):
        inference_params=asdict(inference_params)
    # 过滤掉值为None的参数
    filtered_params = {k: v for k, v in inference_params.items() if v is not None}
    # 构建请求    
    completion:ChatCompletion=client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        **filtered_params        
    )
    result=completion.choices[0].message.content
    return result

def openai_chat_by_api_as_stream(
    model_name:Optional[str]="gpt-3.5-turbo",
    messages:Iterable[ChatCompletionMessageParam]=[], 
    inference_params:Union[InferenceParams,Dict]={}
):
    # 构建客户端
    client=openai.Client(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE
    )
    if isinstance(inference_params,InferenceParams):
        inference_params=asdict(inference_params)
    # 过滤掉值为None的参数
    filtered_params = {k: v for k, v in inference_params.items() if v is not None}
    # 构建请求    
    response:Stream[ChatCompletionChunk]=client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,
        **filtered_params        
    )
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
            
async def openai_chat_by_api_as_astream(
    model_name:Optional[str]="gpt-3.5-turbo",
    messages:Iterable[ChatCompletionMessageParam]=[], 
    inference_params:Union[InferenceParams,Dict]={}
):
    # 构建客户端
    client=openai.AsyncClient(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        http_client=httpx.AsyncClient(
            headers={"User-Agent": "Easy-RAG/1.0"},
            timeout=30.0
        )
    )
    if isinstance(inference_params,InferenceParams):
        inference_params=asdict(inference_params)
    
    # 过滤掉值为None的参数
    filtered_params = {k: v for k, v in inference_params.items() if v is not None}
    
    # 记录请求参数
    logging.info(f"=== LLM API 请求详情 ===")
    logging.info(f"模型: {model_name}")
    logging.info(f"消息: {messages}")
    logging.info(f"推理参数: {filtered_params}")
    logging.info(f"========================")
    
    # 构建请求    
    response:AsyncStream[ChatCompletionChunk]=await client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=True,
        **filtered_params        
    )
    async for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


