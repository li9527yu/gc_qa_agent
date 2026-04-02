import numpy as np
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 判断当前价格list应该返回单一代表价，还是分为两个档次的价格。
def recommend_price_mode(prices, random_state=42, min_cluster_frac=0.1, gap_ratio=0.3, sil_thresh=0.5):
    """
    自动判断是返回单一代表价，还是分为两个档次的价格。
    
    参数:
        prices: list[float] 价格列表
        random_state: 随机种子，保证结果可复现
        min_cluster_frac: 较小簇的最小占比阈值 (默认10%)
        gap_ratio: 两簇中心差异阈值 (默认30%)
        sil_thresh: silhouette阈值 (默认0.5)
    
    返回:
        dict {
          "mode": "single" 或 "two-tier",
          "prices": [float...]   # 推荐价格，一个或两个
          "reason": str          # 判定理由
        }
    """
    X = np.array(prices).reshape(-1, 1)
    
    # k=1情况：中位数
    single_price = float(np.median(prices))
    
    # 如果样本太少，直接返回单价
    if len(prices) < 4:
        return {
            "mode": "single",
            "prices": [single_price],
            "reason": "样本过少，直接返回中位数"
        }
    
    # k=2聚类
    kmeans = KMeans(n_clusters=2, n_init='auto', random_state=random_state).fit(X)
    labels = kmeans.labels_
    centers = sorted(kmeans.cluster_centers_.flatten())
    counts = np.bincount(labels)
    
    # 轮廓系数
    try:
        sil = silhouette_score(X, labels)
    except Exception:
        sil = -1  # 无法计算
    
    # 判断是否存在小簇
    n = len(prices)
    small_cluster = np.any(counts < min_cluster_frac * n)
    
    # 两簇差距
    gap = abs(centers[1] - centers[0])
    avg_val = np.mean(prices)
    gap_ratio_val = gap / avg_val if avg_val != 0 else 0
    
    # 判定逻辑
    if sil >= sil_thresh and gap_ratio_val >= gap_ratio and not small_cluster:
        return {
            "mode": "two-tier",
            "prices": [round(c,2) for c in centers],
            "reason": f"轮廓系数={sil:.2f} >= {sil_thresh}, 差距比例={gap_ratio_val:.2f} >= {gap_ratio}, 两簇规模均衡"
        }
    else:
        return {
            "mode": "single",
            "prices": [single_price],
            "reason": f"条件不足: silhouette={sil:.2f}, gap_ratio={gap_ratio_val:.2f}, 小簇={small_cluster}"
        }

 

"""用 IQR 法去异常值"""
def remove_outliers( prices, iqr_k=1.5):
    
    prices = np.asarray(prices, dtype=float)
    if prices.size == 0:
        return []
    q1, q3 = np.percentile(prices, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    filtered = prices[(prices >= lo) & (prices <= hi)]
    return filtered.tolist()


"""价格分析：接收一组记录（list[dict]），使用k-means算法计算推荐价格，返回统计结果"""
def analyze_prices(price_data: list) -> dict:
    
    if not price_data:
        return {"error": "无价格数据"}

    try:
        prices = []
        for item in price_data:
            v = item.get("price")
            # 过滤 None/空串/NaN
            if v is None or (isinstance(v, str) and v.strip() == ""):
                continue
            f = float(v)
            if np.isfinite(f):
                prices.append(f)

        if not prices:
            return {"error": "无有效价格数据"}

        # 异常值检测
        filtered_prices = remove_outliers(prices)
        if not filtered_prices:
            return {"error": "过滤异常值后无有效数据"}

        # 统计分析
        arr = np.asarray(filtered_prices, dtype=float)
        n = arr.size
        min_price = float(np.min(arr))
        max_price = float(np.max(arr))
        mean_price = float(np.mean(arr))
        median_price = float(np.median(arr))
        # 无偏样本标准差（ddof=1），单个样本退化为 0
        std_price = float(np.std(arr, ddof=1)) if n >= 2 else 0.0

        # 95% 置信区间（正态近似）；n < 2 时不给出区间
        if n >= 2:
            se = std_price / np.sqrt(n)
            half = 1.96 * se
            confidence_interval = (mean_price - half, mean_price + half)
        else:
            confidence_interval = None
        result_kmeans=recommend_price_mode(prices)
        return {
            "total_count": len(price_data),     # 原始记录数
            "valid_count": int(n),              # 剔除异常值后有效价格个数
            "price_range": (min_price, max_price),
            "mean_price": mean_price,
            "median_price": median_price,
            "recommend_kmeans":result_kmeans
        }
    except Exception as e:
        return {"error": f"分析失败: {e}"}
    

"""按 unit 分组，把每个组转成 list[dict] 后调用 _analyze_prices。 返回 {unit: 分析结果字典}"""
def analyze_by_unit(data: list, unit_col="unit") -> dict:

    df = pd.DataFrame(data)
    # 处理缺失/空 unit：统一标记为 'UNKNOWN'
    safe_df = df.copy()
    safe_df[unit_col] = safe_df[unit_col].fillna("UNKNOWN").replace("", "UNKNOWN")

    # 先按 unit 分组计数，找出数量最多的那个
    unit_counts = safe_df[unit_col].value_counts()
    most_common_unit = unit_counts.idxmax()

    # 构建结果：先放最多的分组，再放其他分组
    results = {}

    # 先处理最多的 unit
    group_records = safe_df[safe_df[unit_col] == most_common_unit].to_dict(orient="records")
    results[most_common_unit] = analyze_prices(group_records)

    # 处理其他 unit
    for unit in unit_counts.index:
        if unit == most_common_unit:
            continue
        group_records = safe_df[safe_df[unit_col] == unit].to_dict(orient="records")
        results[unit] =analyze_prices(group_records)

    return {
        "results": results,
        "most_common_unit": most_common_unit,
        "most_common_records": safe_df[safe_df[unit_col] == most_common_unit].to_dict(orient="records"),
    }
