# French Policy Simulator | 法国政策模拟器

> 基于数百万法国人物画像的 AI 政策态度分析平台

## 项目简介

输入一个政策问题（例如 *"退休年龄推迟到 64 岁，你怎么看？"*），系统会：

1. **检索** — 从人物画像数据集中通过向量相似度搜索找出最相关的群体
2. **角色扮演** — 调用 LLM 让每个人物以第一人称发表看法
3. **分析** — 解析回答提取支持度评分和立场（支持/中立/反对）
4. **可视化** — 多维度统计分析，支持按性别、年龄、职业、学历、地区等维度拆解

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                     │
│  政策问题输入 · 多维度分析 · 历史记录 · CSV 导出 · 深色主题 │
├─────────────────────────────────────────────────────────┤
│                    LLM Client 模块                       │
│  智能重试 · 滑动窗口限流(RPM/TPM) · 异步并发 · SQLite 缓存 │
├──────────────────────┬──────────────────────────────────┤
│   Retriever 模块     │        Data Pipeline 模块        │
│  FAISS 向量检索       │  数据清洗 · 类型映射 · EDA 分析   │
│  Sentence-BERT 嵌入   │  Parquet 加载 · 人物画像构建      │
├──────────────────────┴──────────────────────────────────┤
│              SQLite / SQL Server 存储层                  │
└─────────────────────────────────────────────────────────┘
```

## 核心特性

- **大规模人物画像** — 基于 Nemotron Personas France 数据集，覆盖数百万法国虚拟人物
- **向量语义检索** — Sentence-BERT + FAISS 实现相似度搜索
- **智能 LLM 调用** — 滑动窗口限流（RPM/TPM）+ 异步并发 + SQLite 响应缓存
- **异常自动重试** — 超时/连接错误指数退避重试，限速错误直接返回不浪费配额
- **增强 Prompt 设计** — 双示例 Few-shot 引导，移除矛盾指令，输出质量更稳定
- **结果导出** — 一键下载 CSV 完整结果，方便后续分析
- **多维度分析** — 支持按年龄、职业、学历、性别、地区、婚姻状况、家庭类型拆解支持度
- **中文翻译** — 法国 PCS 职业分类、Bac+ 学历体系、行政区划自动翻译为中文
- **深浅色主题** — 自动跟随浏览器系统设置
- **可中断模拟** — 运行中可随时中断，侧边栏自动禁用防止重复提交
- **双数据库支持** — SQLite（默认）或 SQL Server，工厂模式切换

## 项目结构

```
french_policy_simulator/
├── app.py                          # Streamlit Web 应用（主入口）
├── requirements.txt                # Python 依赖
├── .env.example                    # 环境变量模板
├── .gitignore
│
├── src/
│   ├── llm_client/
│   │   ├── prompt.py               # 结构化 Prompt 模板（Few-shot 增强）
│   │   ├── call_llm.py             # LLM 客户端（限流 + 异步 + 缓存 + 智能重试）
│   │   └── parse_response.py       # 解析 LLM 回答 → 支持度评分 + 立场
│   ├── retriever/
│   │   ├── build_index.py          # FAISS 索引构建（Flat / IVF）
│   │   ├── search.py               # 向量相似度搜索
│   │   ├── db_adapter.py           # 数据库抽象接口
│   │   ├── sqlite_adapter.py       # SQLite 实现
│   │   ├── sqlserver_adapter.py    # SQL Server 实现
│   │   ├── db_factory.py           # 数据库工厂
│   │   └── save_results.py         # 结果存储
│   ├── data_pipeline/
│   │   ├── load_data.py            # 数据加载
│   │   ├── clean_data.py           # 数据清洗
│   │   └── eda.py                  # 探索性数据分析
│   ├── ui/                         # 前端组件（模块化拆分）
│   │   ├── styles.py               # CSS 样式定义（深浅色自适应）
│   │   ├── components.py           # KPI 卡片、问题卡片等可复用组件
│   │   └── charts.py               # 图表渲染（饼图、直方图、散点图、条形图）
│   ├── data/                       # 数据工具
│   │   └── translations.py         # 法文→中文翻译字典（职业、省份、学历等）
│   └── utils/
│       └── config.py               # 配置管理
│
├── scripts/
│   ├── 01_clean_data.py            # 步骤 1：数据清洗
│   ├── 02_build_index.py           # 步骤 2：构建 FAISS 索引
│   ├── 03_run_llm.py               # 步骤 3：测试 LLM 连通性
│   └── 04_run_pipeline.py          # 步骤 4：完整命令行 Pipeline
│
├── data/
│   ├── index.faiss                 # FAISS 向量索引（gitignored）
│   ├── processed/                  # 清洗后的 Parquet 数据（gitignored）
│   └── results.db                  # 结果数据库（gitignored）
│
└── tests/
    └── test_retriever.py           # 测试
```

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / Mac
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，至少填入以下内容：

```env
# LLM API 配置（必填）
LLM_API_KEY=your_api_key_here

# 可选：更换模型或 API 地址
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_BASE_URL=https://api.siliconflow.cn/v1

# 数据库配置（默认 SQLite，无需额外配置）
DB_TYPE=sqlite
DB_PATH=data/results.db

# 数据路径（默认即可）
DATA_PATH=data/processed/df_full.parquet
INDEX_PATH=data/index.faiss
```

> **支持的 LLM 服务**：任何 OpenAI 兼容接口均可使用，如 SiliconFlow、OneAPI、LocalAI 等。

### 3. 准备数据文件

项目启动前需要两个数据文件：`df_full.parquet`（清洗后的人物数据）和 `index.faiss`（FAISS 向量索引）。

**情况 A：已有数据文件**

如果 `data/processed/df_full.parquet` 和 `data/index.faiss` 已存在，可直接跳到第 4 步。

**情况 B：从零开始（首次使用）**

需要从原始数据集下载并处理，依次执行：

```bash
# 步骤 1：下载原始数据 → 清洗 → 保存 Parquet
python scripts/01_clean_data.py

# 步骤 2：加载 Parquet → 生成 Embedding → 构建 FAISS 索引
python scripts/02_build_index.py
```

处理完成后的文件结构：

```
data/
├── index.faiss                 # 向量索引（约 150MB）
└── processed/
    └── df_full.parquet         # 清洗后的人物数据（约 500MB+）
```

> **数据集来源**：默认从 HuggingFace 流式加载 Nemotron Personas France 数据集。也可通过配置 `DATA_PATH` 使用本地 CSV/Parquet 文件。

### 4. 测试 LLM 连通性（可选）

```bash
python scripts/03_run_llm.py
```

确认 API 配置正确且网络通畅。

### 5. 启动 Web 应用

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

## 使用指南

### 运行模拟

1. 在左侧栏输入政策问题
2. 调整检索人数（默认 10000 人）
3. 在「LLM 配置」区域设置 API Key、模型、RPM/TPM 和 Temperature
4. 点击「运行模拟」
5. 等待进度完成，查看分析结果
6. 点击「下载完整结果 (CSV)」导出完整数据

### 查看结果

- **Tab 1「查询与总览」** — 立场分布饼图、支持度直方图、KPI 指标、多维度分析柱状图、相似度-支持度散点图、CSV 导出
- **Tab 2「回答详情」** — 逐条查看每个人物的回答，支持按立场/分数/职业/关键词筛选
- **Tab 3「历史查询」** — 浏览和加载之前的模拟结果

### 多维度分析

| 维度 | 说明 |
|------|------|
| 年龄 | 自动分段：<18, 18-24, 25-34, 35-44, 45-54, 55-64, 65+ |
| 职业 | 法国 PCS 八大类（农业从业者、工商业主、高管、中级职业、雇员、工人、退休人员、无业） |
| 学历 | 法国 Bac+ 体系（Bac ~ Bac+5+） |
| 性别 | 男 / 女 |
| 地区 | 法国本土大区 + 海外省（Top 15） |
| 婚姻状况 | 单身、已婚、离婚、丧偶等 |
| 家庭类型 | 独居、无子女夫妻、有子女夫妻、单亲家庭等 |

## LLM 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| API Key | 服务提供商密钥 | - |
| Base URL | API 地址 | `https://api.siliconflow.cn/v1` |
| 模型 | 模型名称 | `Qwen/Qwen2.5-7B-Instruct` |
| RPM | 每分钟最大请求数 | 3000 |
| TPM | 每分钟最大 Token 数 | 500000 |
| Temperature | 生成随机性（0~2） | 0.7 |

系统会根据 RPM/TPM 自动计算最佳并发数，超出限速时自动排队等待，限速错误直接返回不重试。

## 依赖

| 包 | 用途 |
|---|---|
| `streamlit>=1.28.0` | Web 前端框架 |
| `sentence-transformers>=2.2.0` | 文本 Embedding |
| `faiss-cpu>=1.7.4` | 向量相似度搜索 |
| `openai>=1.0.0` | LLM API 客户端 |
| `pandas>=2.0.0` | 数据处理 |
| `plotly>=5.14.0` | 交互式图表 |
| `python-dotenv>=1.0.0` | 环境变量 |
| `tenacity>=8.2.0` | 指数退避重试 |
| `pyodbc>=4.0.39` | SQL Server 连接（可选） |

## 免责声明

**本项目仅供学术研究与技术演示，不构成任何实际政策建议或民意调查。**

- 本项目中的人物画像和数据均来自公开数据集，由 AI 生成的角色回答不代表任何真实个人的观点
- 分析结果（支持度评分、立场分布等）仅为技术演示，**不能替代真实的民意调查或科学研究**
- 结果可能受到 LLM 模型偏见、提示词设计和数据质量的影响，存在系统性偏差
- **本项目不应被用于政治宣传、选举预测、政策决策或任何商业/政治目的**
- 使用者应自行承担因使用本项目产生的一切法律责任
- 本项目遵循 MIT 许可证，作者不对使用本项目造成的任何直接或间接损失负责

## 许可证

MIT
