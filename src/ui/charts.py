"""
可视化函数（图表渲染）

从 app.py 中拆分出来，保持独立可测试。
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.translations import zh_translate
from src.llm_client import build_prompt
from .styles import get_theme_colors
from .components import render_kpi_card, DIMENSION_OPTIONS


def render_charts(df):
    """渲染结果总览图表（仅统计成功请求）"""
    df = df[df["stance"].notna()].copy()
    if len(df) == 0:
        st.error("无有效回答，所有请求均失败")
        return
    c = get_theme_colors()

    # ── 立场分布饼图 ──
    stance_counts = df["stance"].value_counts()
    stance_labels_map = {"oppose": "反对", "neutral": "中立", "support": "支持"}
    labels = [stance_labels_map.get(s, s) for s in stance_counts.index]
    colors = {"反对": "#ef4444", "中立": "#f59e0b", "支持": "#22c55e"}
    pie_colors = [colors.get(l, "#888") for l in labels]

    fig_pie = go.Figure(data=[go.Pie(
        labels=labels, values=stance_counts.values, hole=0.4,
        marker_colors=pie_colors, textinfo="label+percent",
        textfont=dict(color=c["text"]),
    )])
    fig_pie.update_layout(
        title="立场分布", showlegend=False,
        paper_bgcolor=c["plot_bg"], plot_bgcolor=c["plot_bg"],
        font=dict(color=c["text"]),
    )

    # ── 支持度直方图 ──
    fig_hist = px.histogram(
        df, x="support_score", nbins=20,
        title="支持度分布",
        labels={"support_score": "支持度", "count": "人数"},
        color_discrete_sequence=["#636efa"],
    )
    fig_hist.add_vline(
        x=df["support_score"].mean(), line_dash="dash", line_color="red",
        annotation_text=f"平均值: {df['support_score'].mean():.2f}",
    )
    fig_hist.update_layout(
        paper_bgcolor=c["plot_bg"], plot_bgcolor=c["plot_bg"],
        font=dict(color=c["text"]),
    )

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_right:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── KPI 卡片行 ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card("总回答数", f"{len(df):,}", "blue")
    with col2:
        render_kpi_card("平均支持度", df["support_score"].mean(), "green", fmt="float2")
    with col3:
        pct_support = (df["stance"] == "support").sum() / len(df) * 100
        render_kpi_card("支持率", pct_support, "orange", fmt="percent")
    with col4:
        pct_oppose = (df["stance"] == "oppose").sum() / len(df) * 100
        render_kpi_card("反对率", pct_oppose, "red", fmt="percent")

    # ── 多维度分析选择器 ──
    st.markdown("---")
    st.markdown("### 📊 多维度支持度分析")
    dim_label = st.selectbox("选择分析维度", list(DIMENSION_OPTIONS.keys()), index=0)
    dim_field = DIMENSION_OPTIONS[dim_label]

    if dim_field in df.columns:
        df_dim = df.copy()
        zh_col_map = {
            "age": "年龄",
            "occupation": "职业", "departement": "省份", "sex": "性别",
            "education_level": "教育程度", "marital_status": "婚姻状况",
            "household_type": "家庭类型",
        }
        col_name = zh_col_map.get(dim_field, dim_field)

        # 年龄维度特殊处理：分段
        if dim_field == "age":
            age_bins = [0, 18, 25, 35, 45, 55, 65, 120]
            age_labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
            df_dim["_dim_zh"] = pd.cut(
                df_dim["age"].astype(float), bins=age_bins, labels=age_labels,
                right=False,
            ).astype(str)
        else:
            def _zh(v):
                return zh_translate(v, dim_field)
            df_dim["_dim_zh"] = df_dim[dim_field].apply(_zh)

        dim_avg = df_dim.groupby("_dim_zh")["support_score"].agg(["mean", "count"]).reset_index()
        dim_avg.columns = ["维度", "平均支持度", "样本数"]

        if dim_field == "departement":
            dim_avg = dim_avg.sort_values("平均支持度", ascending=False).head(15)
            title_text = f"按{col_name}平均支持度 (Top 15)"
        elif dim_field == "age":
            dim_avg = dim_avg.sort_values("维度", ascending=True)
            title_text = f"按{col_name}平均支持度"
        else:
            min_samples = 10
            dim_avg = dim_avg[dim_avg["样本数"] >= min_samples].sort_values("平均支持度", ascending=False).head(15)
            title_text = f"按{col_name}平均支持度 (Top 15, 最小样本≥{min_samples})"

        if len(dim_avg) > 0:
            fig_dim = px.bar(
                dim_avg, x="平均支持度", y="维度", orientation='h',
                title=title_text,
                labels={"平均支持度": "平均支持度", "维度": col_name},
                color="平均支持度", color_continuous_scale="RdYlGn",
            )
            fig_dim.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                paper_bgcolor=c["plot_bg"], plot_bgcolor=c["plot_bg"],
                font=dict(color=c["text"]),
                height=max(400, len(dim_avg) * 32),
            )
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.plotly_chart(fig_dim, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info(f"{col_name}维度无有效数据（样本数≥10）")
    else:
        st.warning(f"数据中缺少「{dim_field}」字段")

    # ── 相似度 vs 支持度散点图 ──
    st.markdown("---")
    st.markdown("### 🔬 相似度与支持度关系")
    sample_size = min(2000, len(df))
    sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    fig_scatter = px.scatter(
        sample, x="similarity", y="support_score",
        title="相似度 vs 支持度",
        labels={"similarity": "相似度", "support_score": "支持度"},
        opacity=0.5, color_discrete_sequence=["#636efa"],
    )
    fig_scatter.update_layout(
        paper_bgcolor=c["plot_bg"], plot_bgcolor=c["plot_bg"],
        font=dict(color=c["text"]),
    )
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_individual_responses(df, question):
    """渲染个体回答详情"""
    df = df[df["stance"].notna()].copy()
    c = get_theme_colors()

    # ── 筛选条件 ──
    st.markdown("### 🔍 筛选条件")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stance_filter = st.selectbox("立场", ["全部", "支持", "中立", "反对"])
    with col2:
        score_range = st.slider("支持度范围", 0.0, 1.0, (0.0, 1.0), step=0.05)
    with col3:
        search_term = st.text_input("关键词搜索")
    with col4:
        if "occupation" in df.columns:
            occupations = ["全部"] + sorted(df["occupation"].dropna().unique().tolist())
            occ_filter = st.selectbox("职业", occupations)
        else:
            occ_filter = "全部"

    # 过滤
    filtered = df.copy()
    if stance_filter != "全部":
        stance_map = {"支持": "support", "中立": "neutral", "反对": "oppose"}
        filtered = filtered[filtered["stance"] == stance_map[stance_filter]]
    filtered = filtered[
        (filtered["support_score"] >= score_range[0]) &
        (filtered["support_score"] <= score_range[1])
    ]
    if search_term:
        mask = filtered["persona_id"].str.contains(search_term, case=False, na=False)
        if "departement" in filtered.columns:
            mask |= filtered["departement"].str.contains(search_term, case=False, na=False)
        if "occupation" in filtered.columns:
            mask |= filtered["occupation"].str.contains(search_term, case=False, na=False)
        filtered = filtered[mask]
    if occ_filter != "全部" and "occupation" in filtered.columns:
        filtered = filtered[filtered["occupation"] == occ_filter]

    st.caption(f"显示 {len(filtered)} / {len(df)} 条记录")

    # 表格
    display_cols = ["persona_id", "age", "occupation", "education_level",
                    "departement", "support_score", "stance", "similarity"]
    display_cols = [c for c in display_cols if c in filtered.columns]
    display_df = filtered[display_cols].copy()

    if "occupation" in display_df.columns:
        display_df["occupation"] = display_df["occupation"].apply(lambda v: zh_translate(v, "occupation"))
    if "education_level" in display_df.columns:
        display_df["education_level"] = display_df["education_level"].apply(lambda v: zh_translate(v, "education_level"))
    if "departement" in display_df.columns:
        display_df["departement"] = display_df["departement"].apply(lambda v: zh_translate(v, "departement"))

    stance_labels = {"oppose": "反对", "neutral": "中立", "support": "支持"}
    if "stance" in display_df.columns:
        display_df["stance"] = display_df["stance"].map(stance_labels).fillna(display_df["stance"])

    st.dataframe(display_df, use_container_width=True, height=400)

    # ── 展开详情 ──
    st.markdown("---")
    st.markdown("### 查看个体详情")
    selected_id = st.selectbox("选择人物", filtered["persona_id"].tolist() if len(filtered) > 0 else [])
    if selected_id:
        row = filtered[filtered["persona_id"] == selected_id].iloc[0]

        col_a, col_b = st.columns([1, 2])

        with col_a:
            st.markdown("**📋 人物画像**")
            info_cols = ["age", "sex", "marital_status", "household_type", "occupation",
                         "education_level", "departement", "commune"]
            zh_labels = {"age": "年龄", "sex": "性别", "marital_status": "婚姻",
                         "household_type": "家庭", "occupation": "职业",
                         "education_level": "学历", "departement": "省份", "commune": "市镇"}
            for c_col in info_cols:
                val = row.get(c_col, "")
                if pd.notna(val) and val:
                    if c_col in ("occupation", "education_level", "departement",
                                 "marital_status", "household_type"):
                        val = zh_translate(val, c_col)
                    elif c_col == "sex":
                        val = "男" if str(val).strip().upper() in ("M", "MALE", "HOMME", "男性") else "女"
                    label = zh_labels.get(c_col, c_col)
                    st.write(f"- **{label}**: {val}")

        with col_b:
            st.markdown("**📝 发送给 LLM 的 Prompt**")
            persona_dict = row.to_dict()
            prompt = build_prompt(persona_dict, question)
            st.markdown(f'<div class="detail-prompt"><pre style="white-space:pre-wrap;font-size:0.8rem;color:{c["text"]}">{prompt}</pre></div>', unsafe_allow_html=True)

            st.markdown("**💬 LLM 原始回答**")
            response = row.get("llm_response", "")
            if response:
                st.markdown(f'<div class="detail-response"><pre style="white-space:pre-wrap;font-size:0.8rem;color:{c["text"]}">{response}</pre></div>', unsafe_allow_html=True)
            else:
                st.warning("无回答")

            st.markdown("**📊 解析结果**")
            score_col, stance_col = st.columns(2)
            with score_col:
                st.metric("支持度", f"{row.get('support_score', 0):.2f}")
            with stance_col:
                stance_val = row.get("stance", "")
                stance_labels_detail = {"oppose": "反对", "neutral": "中立", "support": "支持"}
                st.metric("立场", stance_labels_detail.get(stance_val, stance_val))

        # 详细画像
        profile_zh = {"persona": "性格特质", "cultural_background": "文化背景",
                      "professional_persona": "职业画像", "sports_persona": "体育偏好",
                      "arts_persona": "艺术偏好", "travel_persona": "旅行偏好",
                      "culinary_persona": "美食偏好", "hobbies_and_interests": "兴趣爱好",
                      "career_goals_and_ambitions": "工作目标", "skills_and_expertise": "技能专长"}
        profile_cols = list(profile_zh.keys())
        has_profile = False
        for c_col in profile_cols:
            val = row.get(c_col, "")
            if pd.notna(val) and str(val).strip():
                if not has_profile:
                    st.markdown("---")
                    st.markdown("**🎨 详细画像**")
                    has_profile = True
                st.write(f"- **{profile_zh.get(c_col, c_col)}**: {val}")
