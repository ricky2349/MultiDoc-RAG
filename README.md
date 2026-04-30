# MultiDoc-RAG Agent：多文档对话式知识库问答系统

## 项目简介

本项目实现了一个**基于Agent的多文档RAG（检索增强生成）对话系统**，支持同时加载多个PDF和网页URL，具备对话记忆、按需检索、多源引用、复杂推理与拒答能力。系统采用 LangChain ReAct Agent 构建，使用 DeepSeek API 作为 LLM，Chroma 作为向量数据库，BAAI/bge-small-zh-v1.5 作为嵌入模型。

**核心特性：**
- 多源文档加载：支持加载 ./docs/ 下所有PDF及用户输入的网页URL，自动合并建库。
- Agent按需检索：基于 ReAct Agent，自动判断是否需要调用检索工具（闲聊不检索）。
- 对话记忆：基于 ConversationBufferMemory，支持多轮追问（如“那它的优点呢？”）。
- 来源标注：回答中自动引用文档来源（如“（来源：PEFT_Methods.pdf）”）。
- 智能拒答：对文档无关或信息不足的问题明确回答“无法回答”，杜绝幻觉。
- 对比总结：对对比类问题自动进行结构化对比（表格/分点），优先给出结论，再补充依据。
- 参数高效：使用 DeepSeek API + 本地HuggingFace嵌入，无需昂贵 GPU。

## 技术架构

```text
用户输入 → Agent决策 → 闲聊？ → 直接回答
                ↓ 需要检索
        DocumentRetriever Tool → 多源向量检索 (Chroma)
                ↓
        检索片段 + 对话历史 + Prompt → DeepSeek LLM
                ↓
        结构化回答（含来源、对比总结）→ 输出
```
### 主要组件

| 组件 | 技术选型 |
|------|----------|
| 文档加载 | PyPDFLoader, WebBaseLoader |
| 文本分割 | RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200) |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 (HuggingFaceEmbeddings) |
| 向量存储 | Chroma (持久化) |
| 检索器 | 相似度检索 (k=4) |
| LLM | DeepSeek API (deepseek-chat) |
| Agent框架 | LangChain ReAct Agent + Tool |
| 记忆 | ConversationBufferMemory |
| 应用层 | Streamlit (可选) / Python 脚本 |

## 环境配置

### 1. 克隆项目

```bash
git clone https://github.com/ricky2349/MultiDoc-RAG-Agent.git
cd MultiDoc-RAG-Agent
```
### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```
### 3. 安装依赖

```bash
pip install -r requirements.txt
```
requirements.txt 内容：

```text
langchain==0.3.27
langchain-community==0.3.29
langchain-core==0.3.76
langchain-openai==0.3.11
langchain-text-splitters==0.3.11
chromadb==0.4.24
sentence-transformers==2.2.2
pypdf==3.17.4
streamlit==1.28.1
python-dotenv==1.0.0
beautifulsoup4==4.12.3
lxml==4.9.3
```
### 4. 配置 API 密钥

创建 `.env` 文件，并写入以下内容：

```text
DEEPSEEK_API_KEY=sk-xxx
```

### 5. 启动应用

执行以下命令启动 Streamlit 界面：

```bash
streamlit run app.py
```
## 使用说明

### 构建知识库

1. 在左侧边栏**上传一个或多个 PDF 文件**（支持多选）。
2. 可选：输入一个**网页 URL**（如技术博客、文档页面）。
3. 点击 **“构建/重建向量库”**，系统将自动：
   - 加载并分割文档
   - 生成向量嵌入
   - 创建 Chroma 向量存储
   - 初始化 Agent 和工具

### 对话问答

- 在底部输入框输入问题。
- 系统会自动判断：
  - **闲聊**（你好、天气等）→ 直接回答，不检索。
  - **文档问题**（LoRA、微调、数据准备等）→ 调用检索工具，基于文档回答。
- 答案下方会显示**引用来源**（PDF 文件名或 URL）。

### 示例对话

| 用户 | 助手 |
|------|------|
| 你好 | 你好呀！今天过得怎么样？ |
| LoRA 微调的核心思想是什么？ | LoRA 通过在原始权重矩阵旁路添加低秩分解矩阵来近似参数更新……（来源：PEFT_Methods.pdf） |
| 那它的优点呢？ | （理解“它”指代 LoRA）优点包括参数效率高、推理零延迟等。 |
| 对比两份 PDF 在微调主题上的不同 | 第一份侧重技术方法，第二份侧重数据与评估。 |

### 清空对话

点击侧边栏 **“一键清空对话历史”** 重置记忆，开始新对话。


