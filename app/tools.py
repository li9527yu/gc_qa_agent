import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_price_distribution(data, bins=5):
    """
    画价格的频数分布图，自动等分区间。

    参数:
    - data: List[dict]，其中每个 dict 至少包含 'price' 键
    - bins: int，划分为多少等分区间（默认为5）

    返回:
    - matplotlib.figure.Figure：绘图对象，可用于展示、保存或编码
    """
    # 创建 DataFrame
    df = pd.DataFrame(data)
    if 'price' not in df.columns:
        raise ValueError("每个数据项都必须包含 'price' 字段")

    # 分区间
    df['price_bin'] = pd.cut(df['price'], bins=bins)
    bin_counts = df['price_bin'].value_counts().sort_index()

    # 开始绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=bin_counts.index.astype(str), y=bin_counts.values, ax=ax, color='skyblue')

    ax.set_title("价格分布频数图")
    ax.set_xlabel("价格区间")
    ax.set_ylabel("频数")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()

    return fig
