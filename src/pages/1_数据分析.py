"""Data Analysis Page - Interactive EDA for bank marketing data."""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.data import load_test_data, load_train_data

st.title("📊 数据分析")

# Load data
train_df = load_train_data()
test_df = load_test_data()
train_df["source"] = "train"
test_df["source"] = "test"
full_df = pd.concat([train_df, test_df], ignore_index=True)

CAT_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]
NUM_COLS = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_index",
    "cons_conf_index",
    "lending_rate3m",
    "nr_employed",
]

tab_overview, tab_dist, tab_corr, tab_compare = st.tabs(
    ["数据概览", "特征分布", "相关性分析", "群体对比"]
)

with tab_overview:
    st.subheader("数据集基本信息")
    col1, col2, col3 = st.columns(3)
    col1.metric("训练集样本数", f"{len(train_df):,}")
    col2.metric("测试集样本数", f"{len(test_df):,}")
    col3.metric("特征数量", "20")

    st.subheader("训练集数据类型")
    dtype_df = pd.DataFrame(
        {
            "列名": train_df.drop(columns=["source"]).dtypes.index,
            "数据类型": train_df.drop(columns=["source"]).dtypes.astype(str),
            "非空值": train_df.drop(columns=["source"]).count().values,
            "缺失值": train_df.drop(columns=["source"]).isnull().sum().values,
        }
    )
    st.dataframe(dtype_df, use_container_width=True)

    st.subheader("目标变量分布")
    sub_counts = train_df["subscribe"].value_counts()
    fig = px.bar(
        x=sub_counts.index,
        y=sub_counts.values,
        labels={"x": "是否认购", "y": "样本数"},
        color=sub_counts.index,
        color_discrete_map={"yes": "#2ecc71", "no": "#e74c3c"},
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_dist:
    st.subheader("特征分布")
    col_feat, col_type = st.columns([1, 1])
    selected_col = col_feat.selectbox("选择特征", NUM_COLS + CAT_COLS)

    is_numeric = selected_col in NUM_COLS

    if is_numeric:
        fig = px.histogram(
            train_df,
            x=selected_col,
            nbins=50,
            labels={selected_col: selected_col, "count": "频数"},
            color_discrete_sequence=["#3498db"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write(train_df[selected_col].describe())
    else:
        counts = train_df[selected_col].value_counts()
        fig = px.bar(
            x=counts.index,
            y=counts.values,
            labels={"x": selected_col, "y": "频数"},
            color_discrete_sequence=["#9b59b6"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.write(counts)

with tab_corr:
    st.subheader("特征与认购的相关性")
    df_encoded = train_df.copy()
    df_encoded["subscribe_num"] = (df_encoded["subscribe"] == "yes").astype(int)

    # Encode categoricals as dummies for correlation
    df_corr = pd.get_dummies(df_encoded, columns=CAT_COLS, drop_first=False)
    numeric_only = df_corr.select_dtypes(include="number")

    corr_matrix = numeric_only.corr(method="pearson", numeric_only=True)
    target_corr = corr_matrix["subscribe_num"].drop("subscribe_num").sort_values(ascending=False)

    st.write("**Top 10 正相关特征**")
    st.dataframe(target_corr.head(10).to_frame("相关系数"), use_container_width=True)

    st.write("**Top 10 负相关特征**")
    st.dataframe(target_corr.tail(10).to_frame("相关系数"), use_container_width=True)

    # Heatmap of top numeric correlations
    st.subheader("数值特征相关性热力图")
    numeric_cols_for_corr = NUM_COLS + ["subscribe_num"]
    corr_num = df_encoded[numeric_cols_for_corr].corr()
    fig = px.imshow(
        corr_num,
        labels=dict(color="相关系数"),
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_compare:
    st.subheader("不同群体认购率对比")

    group_col = st.selectbox("按特征分组", CAT_COLS)
    grouped = (
        train_df.groupby(group_col)["subscribe"]
        .apply(lambda x: (x == "yes").mean() * 100)
        .sort_values(ascending=False)
    )

    fig = px.bar(
        x=grouped.index,
        y=grouped.values,
        labels={"x": group_col, "y": "认购率 (%)"},
        color=grouped.values,
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.write("**各群体认购率详情**")
    detail = train_df.groupby(group_col).agg(
        总人数=("subscribe", "count"),
        认购人数=("subscribe", lambda x: (x == "yes").sum()),
        认购率=("subscribe", lambda x: f"{(x == 'yes').mean() * 100:.1f}%"),
    )
    st.dataframe(detail, use_container_width=True)
