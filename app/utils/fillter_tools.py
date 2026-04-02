import re
import unicodedata
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 定义正则表达式模式
# 匹配尺寸规格，如：2.0*1250*2500、50mm等
RE_SIZE = re.compile(r'(\d+(\.\d+)?\s*(mm|cm|m))|(\d+(\.\d+)?\s*[*xX]\s*\d+(\.\d+)?(\s*[*xX]\s*\d+(\.\d+)?)?)')
# 匹配钢材牌号，如：Q235、304、316L等
RE_GRADE = re.compile(r'(Q\d{3}[A-Z]?|HRB\d{3}[A-Z]?|HPB\d{3}|SUS\d{3}|304L?|316L?)', re.I)
# 匹配加工工艺，如：热轧、冷轧、镀锌等
RE_PROC = re.compile(r'(热轧|冷轧|镀锌|酸洗|拉丝|镜面|抛光)')
# 定义需要去除的标点符号
PUNCT = r'，。、""''！!?：:；;（）()【】[]/,_\-+|'

def normalize(s: str) -> str:
    """
    对输入文本进行标准化处理
    - 全角转半角
    - 转小写
    - 去除标点符号
    - 规范化空格
    """
    if not s: return ""
    s = unicodedata.normalize('NFKC', s)
    s = s.lower().strip()
    s = re.sub(f"[{PUNCT}]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def strip_noise(s: str) -> str:
    """
    去除文本中的噪声信息：
    - 尺寸规格
    - 钢材牌号
    - 加工工艺
    保留核心产品名称
    """
    s = RE_SIZE.sub(" ", s)
    s = RE_GRADE.sub(" ", s)
    s = RE_PROC.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def core_name(s: str) -> str:
    """
    提取文本的核心名称：先标准化，再去除噪声
    """
    return strip_noise(normalize(s))

def containment(q: str, i: str) -> float:
    """
    计算包含度分数：
    - 完全包含返回1.0
    - 去空格后包含返回0.5
    - 不包含返回0.0
    - 查询词过短(小于2个字符)时返回0.0
    """
    if not q or not i: return 0.0
    short, long = (q, i) if len(q) <= len(i) else (i, q)
    if len(short) < 2:  # 过短时不算强包含
        return 0.0
    if short in long:
        return 1.0
    # 弱包含：删除空格后再判断
    if short.replace(" ", "") in long.replace(" ", ""):
        return 0.5
    return 0.0

def cosine_scores(query_core: str, item_cores: List[str]) -> List[float]:
    """
    计算查询词与候选项之间的余弦相似度：
    - 使用字符级别的2-3gram特征
    - 返回查询词与每个候选项的相似度列表
    """
    vect = TfidfVectorizer(analyzer="char", ngram_range=(2,3))
    X = vect.fit_transform([query_core] + item_cores)
    sims = cosine_similarity(X[0:1], X[1:]).flatten().tolist()
    return sims

def filter_items(query: str, items: List[dict], alpha=0.4, T_keep=0.75, T_drop=0.50) -> List[Tuple[str, float, str]]:
    """
    过滤和标记候选项：
    参数：
        query: 查询词
        items: 候选项列表
        alpha: 包含度得分的权重
        T_keep: 保留阈值
        T_drop: 审核阈值
    返回：
        列表of元组(候选项, 得分, 标记)，按得分降序排序
    标记类型：
        - keep: 得分>=T_keep，建议保留
        - review: T_drop<=得分<T_keep，需人工审核
        - drop: 得分<T_drop，建议删除
    """
    q_core = core_name(query)
    i_cores = [core_name(str(x.get("materialName", "")).strip()) for x in items]
    cos_list = cosine_scores(q_core, i_cores)
    results = []
    for x, i_core, cos in zip(items, i_cores, cos_list):
        C = containment(q_core, i_core)
        # 查询词较短时提高保留阈值
        _T_keep = T_keep + (0.05 if len(q_core.replace(" ","")) <= 2 else 0.0)
        # 最终得分 = alpha * 包含度 + (1-alpha) * 余弦相似度
        S = alpha * C + (1 - alpha) * cos
        # 对于大于阈值的保存结果
        if S >= T_drop:
            results.append(x)
        # tag = "drop"
        # if S >= _T_keep:
        #     tag = "keep"
        # elif S >= T_drop:
        #     tag = "review"
        # results.append((name, S, tag))
    return results
# 示例
if __name__ == "__main__":
    query = "不锈钢板"
    candidates = [
        "304不锈钢板 2.0*1250*2500",
        "冷轧钢板 1.5*1000*2000",
        "不锈钢卷 304 拉丝面",
        "HRB400E 螺纹钢 Φ12",
        "钢板切割加工服务"
    ]
    for name, s, tag in filter_items(query, candidates):
        print(f"{tag:6}  {s:.3f}  {name}")
