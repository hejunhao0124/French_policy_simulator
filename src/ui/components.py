"""
可复用 UI 组件（KPI 卡片、问题卡片等）
"""

import streamlit as st
from .styles import get_theme_colors

# 多维度分析维度选项（业务常量）
DIMENSION_OPTIONS = {
    "年龄": "age",
    "职业": "occupation",
    "省份": "departement",
    "性别": "sex",
    "教育程度": "education_level",
    "婚姻状况": "marital_status",
    "家庭类型": "household_type",
}


def render_kpi_card(label, value, card_type, value_color=None, fmt="raw"):
    """渲染单个 KPI 指标卡片

    参数:
        label: 指标名称（中文）
        value: 指标值
        card_type: 卡片配色类型 (blue/green/orange/red)
        value_color: 覆盖数值颜色
        fmt: 格式化方式 "raw" 直接输出 / "percent" 百分比 / "float2" 两位小数
    """
    c = get_theme_colors()
    color = value_color or c.get(card_type, c["text"])

    if fmt == "percent":
        display = f"{value:.1f}%"
    elif fmt == "float2":
        display = f"{value:.2f}"
    else:
        display = str(value)

    st.markdown(f"""<div class="kpi-card {card_type}">
        <div class="kpi-value" style="color:{color}">{display}</div>
        <div class="kpi-label">{label}</div></div>""",
        unsafe_allow_html=True)


def render_question_card(question):
    """渲染问题卡片"""
    st.markdown(f"""
    <div class="question-card">
        <strong>📋 查询问题</strong><br>
        <pre>{question}</pre>
    </div>
    """, unsafe_allow_html=True)
