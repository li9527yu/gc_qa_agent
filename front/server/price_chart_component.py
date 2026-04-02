# price_chart_component.py
from typing import List, Dict
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

def price_distribution_chart(price_data: List[Dict], bins: int = 5):
    """
    根据 price_data 绘制价格分布柱状图
    """
    if not price_data:
        st.warning("暂无价格数据")
        return

    # 1. 取出价格
    prices = [float(item["price"]) for item in price_data]

    # 2. 分箱统计
    df = pd.DataFrame({"price": prices})
    df["bin"] = pd.cut(df["price"], bins=bins)
    counts = df["bin"].value_counts().sort_index()

    # 3. 准备 ECharts 数据
    x_axis = [f"{int(interval.left)}-{int(interval.right)}" for interval in counts.index]
    y_axis = counts.values.tolist()

    option = {
        "title": {"text": "价格分布频数图", "left": "center"},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "xAxis": {"type": "category", "data": x_axis, "name": "价格(元)"},
        "yAxis": {"type": "value", "name": "数量"},
        "series": [
            {
                "data": y_axis,
                "type": "bar",
                "itemStyle": {"color": "#5470c6"},
            }
        ],
    }

    st_echarts(options=option, height="400px")

def price_scatter_chart(price_data: List[Dict]):
    """
    根据 price_data 绘制价格散点图
    """
    if not price_data:
        st.warning("暂无价格数据")
        return

    # 1. 提取数据
    data = []
    for item in price_data:
        release_time = item.get("releaseTime", "")
        price = float(item.get("price", 0))
        if release_time and price:
            year_month = release_time[:7]  # 提取年月
            data.append([year_month, price])

    # 2. 准备 ECharts 数据
    x_axis = sorted(set([d[0] for d in data]))  # 去重并排序
    y_axis = [d[1] for d in data]

    option = {
        "title": {"text": "价格散点图", "left": "center"},
        "tooltip": {"trigger": "item", "formatter": "{a} <br/>{b} : {c}"},
        "xAxis": {"type": "category", "data": x_axis, "name": "时间"},
        "yAxis": {"type": "value", "name": "价格"},
        "series": [
            {
                "data": data,
                "type": "scatter",
                "itemStyle": {"color": "#5470c6"},
            }
        ],
    }

    st_echarts(options=option, height="400px")