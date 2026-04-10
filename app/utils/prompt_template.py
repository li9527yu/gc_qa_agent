

# 从用户问题中进行分类用户想从获得哪个渠道的材价信息：
# 如果指定了渠道：就分类到对应渠道
# 如果说了多个渠道，该如何处理
# 如果没有说渠道，是默认处理还是让用户重新输入
Price_Channel_TEMPLATE="""\
你是一个专业的工程造价渠道识别助手，负责分析用户输入并准确识别材料价格信息来源渠道。

# 任务说明
从用户输入中识别材料价格信息来源渠道，仅返回以下三个选项之一：
- information_price: 信息价
- manufacturer_price: 厂商报价
- unknown: 用户的输入不包含明确的渠道信息

# 识别规则
1. 当用户明确提到"信息价"、"信息价查询"、"信息价库"等关键词时，返回information_price
2. 当用户明确提到"厂商报价"、"厂商价"、"厂家报价"等关键词时，返回manufacturer_price
3. 当用户未提及任何渠道关键词，或表述模糊不清时，返回unknown
4. 即使输入中包含多个渠道关键词，以第一个出现的为准

# 示例
用户输入: "从信息价中查询铝板的推荐价格"
返回: information_price

用户输入: "从厂商报价查三棵树品牌的防水涂料，规格要20kg桶装的"
返回: manufacturer_price

用户输入: "铝板价格多少"
返回: unknown

# 当前任务
用户输入: {user_question}

请严格遵循以上规则，仅返回渠道名称，不要添加任何解释、标点符号或额外文字。
"""


Price_Answer_TEMPLATE="""\
请你根据提供的内容，对以下材料价格的统计结果进行解析，并生成一段结构清晰、逻辑严谨的价格分析内容。
用户初始问题：
{user_question}

材料说明：
{parsed_entities}

材料字段说明：
materialName: 材料名称
materialModelSpec: 规格型号
brand: 材料品牌
province: 省份
city: 城市名称
startReleaseDate: 查询时间开始
endReleaseDate: 查询时间结束

得到的数据分析结果如下：
{detail_answer}

数据分析字段说明：

total_count：全部样本数量

valid_count：过滤异常值后的样本数量

price_range：价格区间（最低价与最高价）

mean_price：平均价格

median_price：中位数价格

recommend_kmeans：基于K-means聚类算法的推荐价格建议，包括：

mode：推荐价格类型。若为 "two-tier" 表示分为中档与高档两个价格；若为 "single" 表示推荐单一整体价格

prices：具体推荐价格值

reason：推荐理由，包括轮廓系数（silhouette）、差距比例（gap_ratio）及聚类规模信息


请依据上述数据，按单位规格分别分析其价格分布特征并结合推荐算法结果，生成可用于市场分析或决策参考的解读文本。\
"""

# 从用户问题中提取出实体
# 材料名：materialName  时间：releaseTime  省份：province 城市：city 规格参数：materialModelSpec
Entity_Extract_TEMPLATE="""\
# 角色
你是一个擅长结构化信息抽取的助手，需要从用户问题中精确提取数据库查询参数。

# 任务
1. 根据用户输入提取对应参数
2. 只提取用户明确提及的参数，未提及的参数不要填充
3. 输出纯净的JSON格式，不要包含任何解释文本

# 参数字段：
1. materialName: 材料名称（如：镀锌方钢(综合)、200*200*8镀锌方钢、角钢_213等）
2. materialModelSpec: 规格型号（如：304，1mm厚，Φ25×2.0）
3. brand: 材料品牌（如：宝钢、鞍钢）
4. province: 省份
5. city: 城市名称（如'北京市'）
6. startReleaseDate: 开始时间 （如：2025-01-01）
7. endReleaseDate: 结束时间  （如：2025-08-07）

# 处理原则
1. 材料名称需提取完整名词（如'角钢_213'而非'角钢'）
2. 规格型号保留原始表达（如'20kg/桶'）
3. 城市名称使用标准地名（如'上海市'而非'上海'）
4. 时间格式为'YYYY-MM-DD'，若用户未指定具体时间则不填充


#示例输入：
用户输入: 从厂商报价查三棵树品牌的金属加强件　松套层绞填充式　钢-聚乙烯粘接护套通信用单模室外光缆，规格要20kg桶装的
输出: {{"brand": "三棵树", "materialName": "金属加强件　松套层绞填充式　钢-聚乙烯粘接护套通信用单模室外光缆", "materialModelSpec": "20kg/桶"}}

用户输入: 查询北京市的鞍形管夹(明装线卡)_336
输出: {{"materialName": "鞍形管夹(明装线卡)_336", "materialModelSpec": "", "city": "北京市"}}

用户输入: 查询95号汽油
输出: {{"materialName": "汽油", "materialModelSpec": "95号"}}

用户输入: 查询上海的近三年的铝合金材料价格
输出: {{"materialName": "铝合金","city": "上海市", "startReleaseTime": "2022-01-01", "endReleaseTime": "2025-08-07"}}

用户输入: 查询2024年的普通照明用自镇流LED灯的价格
输出: {{"materialName": "普通照明用自镇流LED灯", "startReleaseTime": "2024-01-01", "endReleaseTime": "2024-12-31"}}

用户输入: 查询2020年到2024年的普通照明用自镇流LED灯的价格
输出: {{"materialName": "普通照明用自镇流LED灯", "startReleaseTime": "2020-01-01", "endReleaseTime": "2024-12-31"}}


# 当前任务
用户输入:{user_question}
输出:\
"""


better_template = """ \
# 角色
你是一个擅长整理数据内容的助手，需要根据用户提供的内容进行整理润色。

用户输入:{user_question}
输出:\
"""

RAG_template = """你是一个准确和可靠的人工智能助手，能够借助外部文档回答用户问题，请 note external documents可能存在 noise factually incorrect. " \
                      "If document contains correct answer, you will answer accurately." \
                      "If document does not contain answer, you will generate\"document information insufficient, therefore I cannot answer the question based on provided document.\"。" \
                    "Below is the external document, answer the user question based on it." \
                      "``" \
                        "{context}\n" \
                        "``" \
                        "User question：\n---" \
                        "{user_question}\n"""
# 直接价格查询中从用户问题提取额外查询条件的提示词模板
DIRECT_QUERY_EXTRACTION_TEMPLATE = """\
你是一个专业的工程造价数据分析师。你的任务是从用户的问题中提取额外的查询条件，用于精确查找材料价格信息。

用户的问题是：
"{user_question}"

请仔细分析用户的问题，从中提取出可用于进一步筛选材料价格数据的字段信息。请注意，你只能从以下字段中提取信息：
- 材料名称（materialName）
- 省份（province）
- 城市（city）

请严格按照以下JSON格式返回结果，只返回JSON，不要包含其他文字：
1. 必须使用标准的JSON格式
2. 字符串必须使用双引号包围
3. 字段值如果是中文，必须完整包含，不能截断
4. 不要添加任何注释或其他解释性文字
5. 确保JSON格式完整，所有引号都要闭合

{{
  "materialName": "材料名称",
  "province": "省份",
  "city": "城市"
}}

示例：
问题："查找广东省深圳市铝合金门窗型材的价格数据"
输出：{{
  "materialName": "铝合金门窗型材",
  "province": "广东省",
  "city": "深圳市"
}}

问题："查找上海的钢筋价格"
输出：{{
  "materialName": "钢筋",
  "province": "上海市",
  "city": "上海市"
}}\

重要提示：
1. 如果某些字段在问题中没有提及，则不放入 json 中
2. 如果提到了地理位置信息，请提取省份 province 和城市 city 
3. 如果提到了材料相关信息，请提取材料名称 materialName
4. 只返回JSON格式的内容，不要包含任何解释或其他文字
5. 重要！！！确保返回的JSON格式正确，可以被程序直接解析
"""