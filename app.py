"""
法国政策模拟器 - Streamlit 前端

运行: streamlit run app.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows UTF-8 输出（跳过 Streamlit 和已关闭的 stdout）
if sys.platform == "win32":
    try:
        import io
        if hasattr(sys.stdout, 'buffer') and not sys.stdout.closed:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except (ValueError, AttributeError):
        pass

# 离线模式加载模型（跳过 HuggingFace 网络检查，大幅加快加载速度）
os.environ["HF_HUB_OFFLINE"] = "1"

from datetime import datetime
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.utils import Config
from src.retriever import load_index, search_similar, create_database_adapter
from src.llm_client import LLMClient, build_batch_prompts, parse_response
from src.data import zh_translate
from src.ui import (
    get_css,
    render_charts,
    render_individual_responses,
    render_question_card,
)

st.set_page_config(
    page_title="法国政策模拟器",
    page_icon="🇫🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 全局样式 ──
st.markdown(get_css(), unsafe_allow_html=True)

# ── 会话状态初始化 ──
for key, default in [
    ("theme", "dark"),
    ("running", False),
    ("interrupt", False),
    ("run_counter", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================
# 缓存加载函数
# ============================================

@st.cache_data
def load_data_cache():
    """缓存数据加载（优先 parquet，回退 CSV）"""
    parquet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "processed", "df_full.parquet")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path, dtype_backend='pyarrow', engine='pyarrow')
    return pd.read_csv(Config.DATA_PATH, low_memory=False, engine="pyarrow")


@st.cache_resource
def load_model():
    """缓存 SentenceTransformer 模型"""
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    finally:
        sys.stdout = old_stdout
    return model


@st.cache_resource
def load_faiss_index():
    """缓存 FAISS 索引"""
    if not os.path.exists(Config.INDEX_PATH):
        return None
    return load_index(Config.INDEX_PATH)


# ============================================
# 核心执行流程
# ============================================

def run_pipeline(question, k, api_key=None, base_url=None, model=None,
                 rpm=3000, tpm=500000, temperature=0.7):
    """执行完整流程：加载数据 -> 检索 -> LLM -> 保存"""
    if not api_key:
        st.error("请输入 API Key")
        return None

    active_model_name = model or Config.LLM_MODEL
    run_id = st.session_state.get("run_counter", 0)

    with st.status(f"🔄 正在执行模拟 (第 {run_id} 次)...", expanded=True) as status:
        import gc
        gc.collect()

        # 步骤 1-2: 加载数据 + 模型
        st.write("📂 正在加载人物数据...")
        progress_bar = st.progress(5)
        df = load_data_cache()
        st.caption(f"✅ 已加载 {len(df):,} 条人物数据")
        progress_bar.progress(15)

        st.write("🧠 正在加载检索模型...")
        embed_model = load_model()
        progress_bar.progress(20)

        # 步骤 3: 加载 FAISS 索引
        st.write("📇 正在加载向量索引...")
        index = load_faiss_index()
        if index is None:
            st.error(f"FAISS 索引不存在: {Config.INDEX_PATH}\n请先运行 scripts/02_build_index.py")
            status.update(label="❌ 索引不存在", state="error")
            return None
        st.caption("✅ FAISS 索引加载成功")
        progress_bar.progress(25)

        # 步骤 4: 向量检索
        st.write(f"🔍 正在检索 {k:,} 个相似人物...")
        results = search_similar(question, embed_model, index, df, k=k)
        st.caption(f"✅ 找到 {len(results):,} 条相似人物记录")
        progress_bar.progress(40)

        # 步骤 5: 构建 Prompt
        st.write("✍️ 正在构建 Prompt...")
        prompts = build_batch_prompts(results.to_dict('records'), question)
        st.caption(f"已构建 {len(prompts)} 个提示词")
        progress_bar.progress(50)

        # 步骤 6: 调用 LLM
        st.write(f"🤖 正在调用 LLM (模型: {active_model_name})...")
        llm_status = st.empty()
        llm_status.caption("准备发送请求...")

        client = LLMClient(
            api_key=api_key,
            model=active_model_name,
            base_url=base_url or Config.LLM_BASE_URL,
            rpm=rpm,
            tpm=tpm,
            temperature=temperature,
        )

        def llm_progress(done, total):
            pct = 50 + int(35 * done / total)
            progress_bar.progress(pct)
            if st.session_state.get("interrupt", False):
                raise KeyboardInterrupt("用户中断")
            llm_status.caption(f"已完成 {done:,}/{total:,} 个请求 ({done/total*100:.1f}%)")

        try:
            llm_responses = client.call_batch_sync(prompts, progress_callback=llm_progress)
        except KeyboardInterrupt:
            st.warning("⚠️ 模拟已中断")
            status.update(label="⚠️ 模拟已中断", state="error")
            return None
        finally:
            llm_status.empty()

        ok_count = sum(1 for r in llm_responses if r and r.get("success"))
        fail_count = sum(1 for r in llm_responses if not r or not r.get("success"))
        empty_count = sum(1 for r in llm_responses
                         if r and r.get("success") and not r.get("response", "").strip())
        st.caption(f"✅ 成功 {ok_count:,} / 失败 {fail_count:,} / 空回复 {empty_count:,}")
        if fail_count > 0:
            fail_reasons = [r.get("error", "unknown") for r in llm_responses
                          if r and not r.get("success")][:3]
            for fr in fail_reasons:
                st.error(f"LLM 错误: {fr}")
        llm_status.caption(f"✅ 收到 {len(llm_responses)} 条 LLM 回答")
        progress_bar.progress(85)

        # 步骤 7: 解析回答
        st.write("📊 正在解析回答并计算支持度...")
        response_dict = {}
        for r in llm_responses:
            if r and r.get("success") and r.get("response"):
                response_dict[str(r['persona_id'])] = r['response']

        results['persona_id_str'] = results['persona_id'].astype(str)
        results['llm_response'] = results['persona_id_str'].map(response_dict)

        mask = results['llm_response'].notna() & (results['llm_response'] != '')
        parsed = results.loc[mask, 'llm_response'].apply(parse_response)
        results["support_score"] = None
        results["stance"] = None
        results["parse_success"] = False
        results.loc[mask, 'support_score'] = parsed.apply(lambda x: x['support_score'])
        results.loc[mask, 'stance'] = parsed.apply(lambda x: x['stance'])
        results.loc[mask, 'parse_success'] = parsed.apply(lambda x: x.get('parse_success', True))
        progress_bar.progress(95)

        # 步骤 8: 保存结果
        st.write("💾 正在保存结果到数据库...")
        db = create_database_adapter()
        query_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        db.save(query_id, question, results)
        db.close()
        progress_bar.progress(100)

        st.caption("✅ 模拟执行完毕！")
        status.update(label="✅ 模拟执行完毕！", state="complete", expanded=False)

    return results


# ============================================
# 历史查询
# ============================================

def get_db_size():
    """获取数据库存储占用"""
    try:
        if Config.DB_TYPE == "sqlite":
            db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), Config.DB_PATH)
            if os.path.exists(db_path):
                size_bytes = os.path.getsize(db_path)
                if size_bytes < 1024:
                    return f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    return f"{size_bytes / 1024:.1f} KB"
                else:
                    return f"{size_bytes / (1024*1024):.1f} MB"
        elif Config.DB_TYPE == "sqlserver":
            import pyodbc
            conn_str = (
                f"DRIVER={{{Config.SQL_DRIVER}}};SERVER={Config.SQL_SERVER};"
                f"DATABASE={Config.SQL_DATABASE};UID={Config.SQL_USER};PWD={Config.SQL_PASSWORD}"
            )
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT SUM(reserved_page_count) * 8.0 / 1024 AS size_mb "
                "FROM sys.dm_db_partition_stats"
            )
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                return f"{row[0]:.1f} MB"
    except Exception:
        pass
    return "未知"


def render_history():
    """渲染历史查询"""
    db = create_database_adapter()
    queries = db.list_queries()
    db.close()

    if queries.empty:
        st.info("暂无历史查询")
        return

    db_size = get_db_size()
    st.markdown(f"### 📜 历史查询 ({len(queries)} 条)  |  💾 数据库: {db_size}")

    for _, row in queries.iterrows():
        col_a, col_b, col_c, col_d = st.columns([4, 1, 1, 2])
        with col_a:
            st.markdown(f"**{row['question']}**")
        with col_b:
            st.caption(str(row.get('timestamp', ''))[:19])
        with col_c:
            st.caption(f"{row.get('total_responses', 0):,} 条")
        with col_d:
            if st.button("📂 加载", key=f"load_{row['query_id']}"):
                db2 = create_database_adapter()
                df = db2.load(row['query_id'])
                question = db2.get_question(row['query_id'])
                db2.close()
                if df is not None and not df.empty:
                    st.session_state["history_df"] = df
                    st.session_state["history_question"] = question
                    st.rerun()
                else:
                    st.error("查询结果为空")
            if st.button("🗑️ 删除", key=f"del_{row['query_id']}", type="secondary"):
                db3 = create_database_adapter()
                db3.delete_query(row['query_id'])
                db3.close()
                st.success("已删除查询")
                st.rerun()
        st.markdown("---")


# ============================================
# 主界面
# ============================================

st.markdown('<div class="main-header">🇫🇷 法国政策模拟器</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">基于 600 万法国人物画像的 AI 政策态度分析平台</div>', unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3 = st.tabs(["📊 查询与总览", "👤 个体回答详情", "📜 历史查询"])

if st.session_state.get("running", False):
    st.info("⏳ 模拟正在运行中...请切换到「查询与总览」Tab查看进度")

# ── Tab 1: 查询与总览 ──
with tab1:
    with st.sidebar:
        st.markdown("### ⚙️ 查询配置")
        st.markdown("---")
        question = st.text_area(
            "政策问题", height=80,
            placeholder="例如: 退休年龄推迟到64岁，你怎么看？",
            disabled=st.session_state["running"],
        )
        k = st.number_input(
            "检索数量", value=10000, min_value=1, max_value=100000, step=100,
            disabled=st.session_state["running"],
        )

        st.markdown("---")
        with st.expander("🔑 LLM 配置", expanded=False):
            api_key = st.text_input(
                "API Key", value=Config.LLM_API_KEY, type="password",
                disabled=st.session_state["running"],
            )
            base_url = st.text_input(
                "API Base URL", value=Config.LLM_BASE_URL,
                disabled=st.session_state["running"],
            )
            model_name = st.text_input(
                "模型", value=Config.LLM_MODEL,
                disabled=st.session_state["running"],
            )
            rpm = st.number_input(
                "RPM (每分钟请求数)", value=3000, min_value=1,
                disabled=st.session_state["running"],
            )
            tpm = st.number_input(
                "TPM (每分钟 Token 数)", value=500000, min_value=1,
                disabled=st.session_state["running"],
            )
            temperature = st.slider(
                "Temperature", 0.0, 2.0, 0.7, step=0.1,
                disabled=st.session_state["running"],
            )

        st.markdown("---")
        st.markdown("### 📊 系统状态")
        if st.session_state["running"]:
            st.warning("⏳ 正在运行模拟，请稍候...")
            if st.button("⏹️ 中断模拟", type="secondary", use_container_width=True):
                st.session_state["interrupt"] = True
                st.rerun()
        st.info(f"**数据库**: {Config.DB_TYPE}")

        run_disabled = st.session_state["running"] or not question
        if st.button("🚀 运行模拟", type="primary",
                     use_container_width=True, disabled=run_disabled):
            st.session_state["running"] = True
            st.session_state["interrupt"] = False
            st.session_state["results_df"] = None
            st.session_state["run_counter"] += 1
            st.rerun()

    # 执行模拟
    if st.session_state["running"]:
        run_id = st.session_state.get("run_counter", 0)
        with st.container(key=f"run_container_{run_id}"):
            results = run_pipeline(question, k, api_key, base_url, model_name,
                                   rpm, tpm, temperature)
        if results is not None:
            st.session_state["results_df"] = results
            st.session_state["question"] = question
        st.session_state["running"] = False
        st.session_state["interrupt"] = False
        st.rerun()

    # 显示结果或空状态
    has_results = "results_df" in st.session_state and st.session_state["results_df"] is not None
    has_history = "history_df" in st.session_state and st.session_state["history_df"] is not None

    if not has_results and not has_history:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <p style="font-size: 4rem;">🇫🇷</p>
            <h2 style="color: #e0e0e0;">欢迎使用法国政策模拟器</h2>
            <p style="color: #888;">在左侧输入政策问题，点击「运行模拟」开始分析</p>
        </div>
        """, unsafe_allow_html=True)
    elif has_results:
        df = st.session_state["results_df"]
        q = st.session_state.get("question", "未知问题")
        render_question_card(q)
        st.download_button(
            label="📥 下载完整结果 (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"simulation_{q[:30]}.csv",
            mime="text/csv",
        )
        render_charts(df)
    elif has_history:
        df = st.session_state["history_df"]
        q = st.session_state.get("history_question", "未知问题")
        render_question_card(q)
        st.download_button(
            label="📥 下载完整结果 (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"simulation_{q[:30]}.csv",
            mime="text/csv",
        )
        render_charts(df)

# ── Tab 2: 个体回答详情 ──
with tab2:
    has_results = "results_df" in st.session_state and st.session_state["results_df"] is not None
    has_history = "history_df" in st.session_state and st.session_state["history_df"] is not None

    if has_results:
        df = st.session_state["results_df"]
        q = st.session_state.get("question", "")
        render_individual_responses(df, q)
    elif has_history:
        df = st.session_state["history_df"]
        q = st.session_state.get("history_question", "")
        render_individual_responses(df, q)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <p style="font-size: 3rem;">👤</p>
            <h3 style="color: #e0e0e0;">暂无查询结果</h3>
            <p style="color: #888;">请先在「查询与总览」Tab 中运行一次查询</p>
        </div>
        """, unsafe_allow_html=True)

# ── Tab 3: 历史查询 ──
with tab3:
    render_history()
