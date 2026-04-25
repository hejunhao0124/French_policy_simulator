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
│  政策问题输入 · 多维度分析 · 历史记录 · 深浅色自适应      │
├─────────────────────────────────────────────────────────┤
│                    LLM Client 模块                       │
│  滑动窗口限流(RPM/TPM) · 异步并发 · SQLite 响应缓存       │
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
- **向量语义检索** — Sentence-BERT + FAISS 实现毫秒级相似度搜索
- **滑动窗口限流** — 根据 RPM/TPM 自动限速，避免 429 错误，限速错误不重试
- **异步 LLM 调用** — asyncio 并发 + SQLite 响应缓存，避免重复请求
- **可调节 Temperature** — 前端 slider 控制生成随机性（0.0 ~ 2.0）
- **仅统计成功请求** — 失败请求自动排除，不影响统计结果
- **多维度分析** — 支持按职业、学历、性别、地区、婚姻状况等维度统计支持度
- **中文翻译** — 法国 PCS 职业分类、Bac+ 学历体系、行政区划自动翻译为中文
- **深浅色主题** — 自动跟随浏览器系统设置适配
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
│   │   ├── prompt.py               # 结构化 Prompt 模板
│   │   ├── call_llm.py             # LLM 客户端（限流 + 异步并发 + 缓存）
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
│   ├── index.faiss                 # FAISS 向量索引
│   ├── processed/                  # 清洗后的 Parquet 数据
│   └── results.db                  # 结果数据库
│
└── tests/
    └── test_retriever.py           # 测试
```

## 快速开始

### 1. 环境配置

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

编辑 `.env`，填入你的 API Key：

```env
LLM_API_KEY=your_api_key_here
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_BASE_URL=https://api.siliconflow.cn/v1
DB_TYPE=sqlite
DB_PATH=data/results.db
```

### 3. 启动 Web 应用

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501` 即可使用。

### 4. 重新构建数据（可选）

```bash
python scripts/01_clean_data.py    # 清洗数据 → Parquet
python scripts/02_build_index.py   # 生成 Embedding → FAISS 索引
python scripts/03_run_llm.py       # 测试 LLM API 连通性
python scripts/04_run_pipeline.py  # 命令行完整 Pipeline
```

## 使用指南

### 运行模拟

1. 在左侧栏输入政策问题
2. 调整检索人数
3. 在 LLM 配置区域设置 API Key、模型、RPM/TPM 和 Temperature
4. 点击「运行模拟」
5. 等待进度完成，查看分析结果

### 查看结果

- **Tab 1「查询与总览」** — 立场分布饼图、支持度直方图、KPI 指标、多维度分析柱状图、相似度-支持度散点图
- **Tab 2「回答详情」** — 逐条查看每个人物的回答，支持按立场/分数/职业筛选
- **Tab 3「历史查询」** — 浏览和加载之前的模拟结果

### 多维度分析

| 维度 | 说明 |
|------|------|
| 职业 | 法国 PCS 八大类（农业从业者、工商业主、高管、中级职业、雇员、工人、退休人员、无业） |
| 学历 | 法国 Bac+ 体系（Bac ~ Bac+5+） |
| 性别 | 男 / 女 |
| 地区 | 法国本土大区 + 海外省 |
| 婚姻状况 | 单身、已婚、离婚、丧偶等 |
| 家庭类型 | 单身家庭、多孩家庭等 |

## LLM 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| API Key | 服务提供商密钥 | - |
| Base URL | API 地址 | `https://api.siliconflow.cn/v1` |
| 模型 | 模型名称 | `Qwen/Qwen2.5-7B-Instruct` |
| RPM | 每分钟最大请求数 | 3000 |
| TPM | 每分钟最大 Token 数 | 500000 |
| Temperature | 生成随机性（0~2） | 0.7 |

系统会根据 RPM/TPM 自动计算最佳并发数，超出限速时自动排队等待，限速错误不会重试。

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
