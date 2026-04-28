"""
CSS 样式定义（深浅色主题自适应）
"""


def get_css():
    """返回完整的 <style> 块 CSS 字符串"""
    return """
<style>
/* ===== 深色主题 ===== */
@media (prefers-color-scheme: dark) {
    .stApp { background: #0f1117 !important; color: #e0e0e0 !important; }
    .main-header {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sub-header { color: #888 !important; }
    .question-card { background: #1a1f2e !important; border-left-color: #3a7bd5 !important; }
    .question-card pre { color: #d0d0d0 !important; }
    .kpi-card.blue { background: linear-gradient(135deg, #1a3a5c, #1e4976) !important; }
    .kpi-card.green { background: linear-gradient(135deg, #1a3c2a, #1e6942) !important; }
    .kpi-card.orange { background: linear-gradient(135deg, #3c2a1a, #69421e) !important; }
    .kpi-card.red { background: linear-gradient(135deg, #3c1a1a, #691e1e) !important; }
    .kpi-label { color: #bbb !important; }
    .chart-box { background: #1a1f2e !important; }
    .detail-prompt { background: #1a2744 !important; }
    .detail-response { background: #1a3c2a !important; }
    .history-card { background: #1a1f2e !important; border-left-color: #3a7bd5 !important; }
    section[data-testid="stSidebar"] { background-color: #141824 !important; }
    .stTabs [data-baseweb="tab"] { background-color: #1a1f2e !important; color: #aaa !important; border-color: #2a2f3e !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #1a3a5c, #1e4976) !important; color: #fff !important; }
}

/* ===== 浅色主题 ===== */
@media (prefers-color-scheme: light) {
    .stApp { background: linear-gradient(135deg, #f5f7fa, #e8ecf1) !important; color: #333 !important; }
    .main-header {
        background: linear-gradient(90deg, #1a73e8, #34a853);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .sub-header { color: #666 !important; }
    .question-card { background: #ffffff !important; border-left-color: #1a73e8 !important; box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important; }
    .question-card pre { color: #333 !important; }
    .kpi-card.blue { background: linear-gradient(135deg, #e3f2fd, #bbdefb) !important; }
    .kpi-card.green { background: linear-gradient(135deg, #e8f5e9, #c8e6c9) !important; }
    .kpi-card.orange { background: linear-gradient(135deg, #fff3e0, #ffe0b2) !important; }
    .kpi-card.red { background: linear-gradient(135deg, #fce4ec, #f8bbd0) !important; }
    .kpi-label { color: #555 !important; }
    .chart-box { background: #ffffff !important; box-shadow: 0 2px 6px rgba(0,0,0,0.06) !important; }
    .detail-prompt { background: #f8f9fa !important; }
    .detail-response { background: #f0faf0 !important; }
    .history-card { background: #ffffff !important; border-left-color: #1a73e8 !important; box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important; }
    section[data-testid="stSidebar"] { background-color: #f0f2f6 !important; }
    .stTabs [data-baseweb="tab"] { background-color: #f0f2f6 !important; color: #666 !important; border-color: #ddd !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #e3f2fd, #bbdefb) !important; color: #1a73e8 !important; }
}

/* 公共样式 */
.main-header { font-size: 2.5rem; font-weight: 800; margin-bottom: 0.3rem; }
.sub-header { font-size: 1rem; margin-bottom: 0.5rem; }
.question-card { border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
.question-card pre { white-space: pre-wrap; font-size: 0.95rem; margin: 0.5rem 0 0 0; }
.kpi-card { padding: 1.2rem; border-radius: 12px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.3); }
.kpi-value { font-size: 2.2rem; font-weight: 800; }
.kpi-label { font-size: 0.85rem; margin-top: 0.2rem; }
.chart-box { border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem; }
.detail-prompt { padding: 0.8rem 1rem; border-radius: 8px; border-left: 4px solid #3a7bd5; }
.detail-response { padding: 0.8rem 1rem; border-radius: 8px; border-left: 4px solid #22c55e; }
.history-card { border-radius: 10px; padding: 1rem; margin-bottom: 0.5rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { padding: 12px 28px; border-radius: 10px 10px 0 0; border: 1px solid; font-weight: 600 !important; }
</style>
"""


def get_theme_colors():
    """根据 Streamlit session_state 中的主题返回配色"""
    import streamlit as st

    if st.session_state.get("theme", "dark") == "dark":
        return {
            "bg": "#0f1117", "card": "#1a1f2e", "plot_bg": "#141824",
            "text": "#d0d0d0", "accent": "#3a7bd5", "green": "#4caf50",
            "orange": "#ff9800", "red": "#ef5350", "blue": "#5ba3e6",
        }
    else:
        return {
            "bg": "#ffffff", "card": "#ffffff", "plot_bg": "#f4f5f7",
            "text": "#333333", "accent": "#1a73e8", "green": "#2e7d32",
            "orange": "#e65100", "red": "#c62828", "blue": "#1565c0",
        }
