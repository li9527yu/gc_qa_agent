# LLM 配置
# 选择使用哪个模型: "deepseek", "local", "local_awq", "local_quantized", 或 "local_vllm_quantized"
# 使用本地 Qwen3-32B 模型
USE_LLM = "local"  # 改为 "deepseek" 可切换回 DeepSeek API, "local_awq" 使用AWQ量化模型, "local_quantized" 使用GGUF量化模型, "local_vllm_quantized" 使用vLLM量化模型

# DeepSeek API 配置
DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEEPSEEK_API_KEY = "sk-51657ec2e0854af19279f8826a322715"
DEEPSEEK_MODEL = "deepseek-chat"

# 本地模型配置 - Qwen3-32B
# 根据实际部署情况配置
LOCAL_API_BASE = "http://localhost:9839/v1"  # 本地模型服务地址
LOCAL_API_KEY = "YOUR_API_KEY"  # 本地模型 API Key
LOCAL_MODEL = "Qwen3-32B"  # 模型名称

# AWQ量化模型配置
LOCAL_AWQ_API_BASE = "http://localhost:9843/v1"
LOCAL_AWQ_MODEL = "qwen3-30b-a3b-instruct-awq"

# GGUF量化模型配置
LOCAL_QUANTIZED_API_BASE = "http://localhost:9841/v1"
LOCAL_QUANTIZED_MODEL = "qwen3-30b-a3b-instruct-gguf-q4km"

# vLLM量化模型配置
LOCAL_VLLM_QUANTIZED_API_BASE = "http://localhost:9844/v1"
LOCAL_VLLM_QUANTIZED_MODEL = "./models/quantized-qwen3-30b-a3b-instruct-vllm"


# 根据选择设置实际使用的配置
if USE_LLM == "local":
    OPENAI_API_BASE = LOCAL_API_BASE
    OPENAI_API_KEY = LOCAL_API_KEY
    LLM_MODEL_NAME = LOCAL_MODEL
elif USE_LLM == "local_awq":
    OPENAI_API_BASE = LOCAL_AWQ_API_BASE
    OPENAI_API_KEY = LOCAL_API_KEY  # AWQ服务可能也需要API密钥
    LLM_MODEL_NAME = LOCAL_AWQ_MODEL
elif USE_LLM == "local_quantized":
    OPENAI_API_BASE = LOCAL_QUANTIZED_API_BASE
    OPENAI_API_KEY = LOCAL_API_KEY  # 量化模型服务可能也需要API密钥
    LLM_MODEL_NAME = LOCAL_QUANTIZED_MODEL
elif USE_LLM == "local_vllm_quantized":
    OPENAI_API_BASE = LOCAL_VLLM_QUANTIZED_API_BASE
    OPENAI_API_KEY = LOCAL_API_KEY  # vLLM量化模型服务可能也需要API密钥
    LLM_MODEL_NAME = LOCAL_VLLM_QUANTIZED_MODEL
else:  # deepseek
    OPENAI_API_BASE = DEEPSEEK_API_BASE
    OPENAI_API_KEY = DEEPSEEK_API_KEY
    LLM_MODEL_NAME = DEEPSEEK_MODEL

# 其他配置
HF_ENDPOINT = "https://hf-mirror.com"

# 价格查询 API 配置
# 设置为 True 使用 Mock 数据，False 使用真实 API
USE_MOCK_API = False

# Mock API 配置（使用 Apifox Mock 服务）
MOCK_API_BASE_URL = "https://m1.apifoxmock.com/m1/7437190-7170870-default/backend/largeModelMaterial"  # 替换为你的 Apifox Mock URL

# 真实 API 配置
REAL_API_BASE_URL = "http://113.31.103.48:8801/backend/largeModelMaterial"
# API 签名密钥（用于真实 API 调用后端）
API_SECRET_KEY = "uwRbSmxx9-X5NEFPTkjpqhj6GBaVVI6V5ulxtczgY5I"

# # 真实数据： 工程造价
# RELATED_DATA_PATH = "app/dataset/工程造价/"
# DATA_PATH = "/home/tpc/suda/rag/easy-rag/app/dataset/工程造价/"

# 测试数据： data
RELATED_DATA_PATH = "app/dataset/data/"
DATA_PATH = "/home/tpc/suda/rag/rag_agent/app/dataset/data"

# 根目录
ROOT_PATH = "/home/tpc/suda/rag/rag_agent"     
