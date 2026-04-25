"""
EDA 可视化模块 (Exploratory Data Analysis)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_age_distribution(df):
    """
    绘制年龄分布图

    参数:
        df: 数据 DataFrame

    返回:
        plotly Figure
    """
    fig = px.histogram(
        df, x="age", nbins=50,
        title="年龄分布",
        labels={"age": "年龄", "count": "人数"},
        color_discrete_sequence=["#2E86C1"],
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_profession_distribution(df, top_n=15):
    """
    绘制职业分布图（Top N）

    参数:
        df: 数据 DataFrame
        top_n: 显示的职业数量

    返回:
        plotly Figure
    """
    prof_counts = df["occupation"].value_counts().head(top_n)
    fig = px.bar(
        x=prof_counts.values, y=prof_counts.index,
        orientation="h",
        title=f"职业分布 (Top {top_n})",
        labels={"x": "人数", "y": "职业"},
        color_discrete_sequence=["#2E86C1"],
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_department_distribution(df, top_n=15):
    """
    绘制省份分布图（Top N）

    参数:
        df: 数据 DataFrame
        top_n: 显示的省份数量

    返回:
        plotly Figure
    """
    dept_counts = df["departement"].value_counts().head(top_n)
    fig = px.bar(
        x=dept_counts.values, y=dept_counts.index,
        orientation="h",
        title=f"省份分布 (Top {top_n})",
        labels={"x": "人数", "y": "省份"},
        color_discrete_sequence=["#2E86C1"],
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_gender_distribution(df):
    """
    绘制性别分布饼图

    参数:
        df: 数据 DataFrame

    返回:
        plotly Figure
    """
    sex_counts = df["sex"].value_counts()
    fig = px.pie(
        values=sex_counts.values, names=sex_counts.index,
        title="性别分布",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    return fig


def plot_education_distribution(df, top_n=10):
    """
    绘制教育水平分布图

    参数:
        df: 数据 DataFrame
        top_n: 显示的教育水平数量

    返回:
        plotly Figure
    """
    edu_counts = df["education_level"].value_counts().head(top_n)
    fig = px.bar(
        x=edu_counts.index, y=edu_counts.values,
        title=f"教育水平分布 (Top {top_n})",
        labels={"x": "教育水平", "y": "人数"},
        color_discrete_sequence=["#2E86C1"],
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=45)
    return fig


def plot_marital_distribution(df):
    """
    绘制婚姻状况分布饼图

    参数:
        df: 数据 DataFrame

    返回:
        plotly Figure
    """
    mar_counts = df["marital_status"].value_counts()
    fig = px.pie(
        values=mar_counts.values, names=mar_counts.index,
        title="婚姻状况分布",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    return fig


def plot_age_by_occupation(df, top_n=8):
    """
    绘制不同职业的平均年龄箱线图

    参数:
        df: 数据 DataFrame
        top_n: 显示的职业数量

    返回:
        plotly Figure
    """
    top_professions = df["occupation"].value_counts().head(top_n).index
    df_filtered = df[df["occupation"].isin(top_professions)]
    fig = px.box(
        df_filtered, x="occupation", y="age",
        title=f"各职业年龄分布 (Top {top_n})",
        labels={"occupation": "职业", "age": "年龄"},
        color="occupation",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=30)
    return fig


def plot_dashboard(df):
    """
    绘制综合仪表盘（2x3 子图）

    参数:
        df: 数据 DataFrame

    返回:
        plotly Figure
    """
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "histogram"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "box"}, {"type": "bar"}],
        ],
        subplot_titles=(
            "年龄分布", "性别分布",
            "职业分布 (Top 10)", "婚姻状况",
            "各职业年龄 (Top 6)", "教育水平 (Top 8)",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # 年龄分布
    age_hist = px.histogram(df, x="age", nbins=50)
    for trace in age_hist.data:
        fig.add_trace(trace, row=1, col=1)

    # 性别分布
    sex_counts = df["sex"].value_counts()
    for i, (name, val) in enumerate(sex_counts.items()):
        fig.add_trace(go.Pie(labels=[name], values=[val], showlegend=True), row=1, col=2)

    # 职业分布
    prof_counts = df["occupation"].value_counts().head(10)
    for i, (name, val) in enumerate(prof_counts.items()):
        fig.add_trace(go.Bar(x=[val], y=[name], orientation="h", showlegend=False), row=2, col=1)

    # 婚姻状况
    mar_counts = df["marital_status"].value_counts()
    for i, (name, val) in enumerate(mar_counts.items()):
        fig.add_trace(go.Pie(labels=[name], values=[val], showlegend=True), row=2, col=2)

    # 职业年龄箱线图
    top_professions = df["occupation"].value_counts().head(6).index
    df_filtered = df[df["occupation"].isin(top_professions)]
    box_data = []
    for prof in top_professions:
        ages = df_filtered[df_filtered["occupation"] == prof]["age"].dropna()
        box_data.append(go.Box(y=ages, name=prof, showlegend=False))
    for trace in box_data:
        fig.add_trace(trace, row=3, col=1)

    # 教育水平
    edu_counts = df["education_level"].value_counts().head(8)
    for i, (name, val) in enumerate(edu_counts.items()):
        fig.add_trace(go.Bar(x=[name], y=[val], showlegend=False), row=3, col=2)

    fig.update_layout(height=900, title_text="法国人物画像数据概览", showlegend=False)
    return fig


def print_summary_stats(df):
    """
    打印数据统计摘要

    参数:
        df: 数据 DataFrame
    """
    print("\n" + "=" * 60)
    print("数据统计摘要")
    print("=" * 60)

    print(f"\n总记录数: {len(df):,}")
    print(f"字段数: {len(df.columns)}")
    print(f"内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # 缺失值统计
    print("\n缺失值统计:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_info = pd.DataFrame({"缺失数": missing, "缺失比例(%)": missing_pct})
    missing_info = missing_info[missing_info["缺失数"] > 0].sort_values("缺失数", ascending=False)
    if len(missing_info) > 0:
        for col, row in missing_info.iterrows():
            print(f"  {col}: {int(row['缺失数']):,} ({row['缺失比例(%)']}%)")
    else:
        print("  无缺失值")

    # 数值字段统计
    if "age" in df.columns:
        print(f"\n年龄统计:")
        print(f"  平均: {df['age'].mean():.1f}")
        print(f"  中位数: {df['age'].median():.1f}")
        print(f"  最小: {df['age'].min()}")
        print(f"  最大: {df['age'].max()}")

    # 分类字段统计
    categorical = ["sex", "occupation", "education_level", "marital_status", "departement"]
    for col in categorical:
        if col in df.columns:
            unique_count = df[col].nunique()
            top_val = df[col].value_counts().head(1)
            print(f"\n{col}: {unique_count} 个唯一值")
            if len(top_val) > 0:
                print(f"  最多: {top_val.index[0]} ({top_val.values[0]:,})")

    print("\n" + "=" * 60)