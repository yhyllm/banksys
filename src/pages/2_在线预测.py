"""Online Prediction Page - Point-and-click customer input for subscription prediction."""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ml import predict as ml_predict

st.title("🎯 在线认购预测")

st.markdown("请选择客户特征,点击预测按钮查看认购预测结果。")

# Define input options
JOB_OPTIONS = [
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]
MARITAL_OPTIONS = ["single", "married", "divorced", "unknown"]
EDUCATION_OPTIONS = [
    "basic.4y",
    "basic.6y",
    "basic.9y",
    "high.school",
    "illiterate",
    "professional.course",
    "university.degree",
    "unknown",
]
YES_NO_UNKNOWN = ["yes", "no", "unknown"]
CONTACT_OPTIONS = ["cellular", "telephone"]
MONTH_OPTIONS = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]
DAY_OPTIONS = ["mon", "tue", "wed", "thu", "fri"]
POUTCOME_OPTIONS = ["nonexistent", "success", "failure"]

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("年龄", 17, 100, 35)
    job = st.selectbox("职业", JOB_OPTIONS)
    marital = st.selectbox("婚姻状况", MARITAL_OPTIONS)
    education = st.selectbox("教育程度", EDUCATION_OPTIONS)

with col2:
    default = st.radio("是否有信用违约", ["no", "yes", "unknown"])
    housing = st.radio("是否有房贷", ["yes", "no", "unknown"])
    loan = st.radio("是否有个人贷款", ["yes", "no", "unknown"])
    contact = st.selectbox("联系方式", CONTACT_OPTIONS)

with col3:
    month = st.selectbox("联系月份", MONTH_OPTIONS)
    day_of_week = st.selectbox("联系星期", DAY_OPTIONS)
    poutcome = st.selectbox("上次营销结果", POUTCOME_OPTIONS)

col4, col5 = st.columns(2)
with col4:
    duration = st.number_input("通话时长(秒)", min_value=0, value=200, step=10)
    campaign = st.number_input("本次活动联系次数", min_value=0, value=1, step=1)
    pdays = st.number_input("上次活动后天数(999=未联系)", min_value=0, value=999, step=1)
with col5:
    previous = st.number_input("本次前联系次数", min_value=0, value=0, step=1)
    emp_var_rate = st.number_input("就业变化率", value=1.4, step=0.1, format="%.1f")
    cons_price_index = st.number_input("消费者物价指数", value=93.0, step=0.1, format="%.1f")
    cons_conf_index = st.number_input("消费者信心指数", value=-36.0, step=0.1, format="%.1f")
    lending_rate3m = st.number_input("3个月拆借利率", value=4.0, step=0.1, format="%.1f")
    nr_employed = st.number_input("雇员数量", value=5000.0, step=10.0, format="%.1f")

features = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "month": month,
    "day_of_week": day_of_week,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
    "emp_var_rate": emp_var_rate,
    "cons_price_index": cons_price_index,
    "cons_conf_index": cons_conf_index,
    "lending_rate3m": lending_rate3m,
    "nr_employed": nr_employed,
}

if st.button("🔮 预测", type="primary"):
    try:
        result = ml_predict.predict_single(features)

        st.divider()
        col_result, col_prob = st.columns(2)

        if result["prediction"] == "yes":
            col_result.success("预测结果: **会认购**")
        else:
            col_result.error("预测结果: **不会认购**")

        prob_pct = result["probability"] * 100
        col_prob.metric("认购概率", f"{prob_pct:.1f}%")

        st.progress(result["probability"])

        if result["top_features"]:
            st.subheader("关键影响因素")
            for i, feat in enumerate(result["top_features"], 1):
                msg = f"{i}. **{feat['feature']}** "
                msg += f"(权重: {feat['importance']:.4f}, 值: {feat['value']})"
                st.write(msg)

    except FileNotFoundError as e:
        st.error(str(e))
        st.info("请先运行训练: `python src/ml/train.py`")
