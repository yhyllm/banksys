"""Bank Marketing Analysis - Streamlit App."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="银行营销数据分析",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 银行营销数据分析平台")

st.markdown(
    """
### 欢迎使用

本应用提供两个核心功能:

1. **数据分析** - 探索银行营销数据的特征分布、相关性和关键发现
2. **在线预测** - 输入客户特征,实时预测是否会认购定期存款产品

请从左侧导航栏选择功能页面开始使用。
"""
)

st.divider()

st.markdown(
    """
### 数据集概览

- **训练数据**: 22,500 条客户记录,包含是否认购标签
- **特征维度**: 20 个特征(人口统计、联系方式、宏观经济指标)
- **目标变量**: `subscribe` - 是否认购定期存款 (yes/no)
"""
)
