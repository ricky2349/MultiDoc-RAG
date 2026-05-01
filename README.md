# MultiDoc RAG 智能问答系统

## 项目简介

本项目实现了一个基于检索增强生成（RAG）的多文档智能问答系统。用户可以上传多个 PDF 文件或输入网页 URL，系统自动构建知识库，并通过自然语言对话与文档内容进行交互。系统支持多轮对话记忆、来源自动标注、智能拒答等核心功能，适合用于私有文档问答、技术资料检索、学习笔记助手等场景。

**核心特性：**
- 多源文档加载：支持多个 PDF 和网页 URL 混合构建知识库
- 多轮对话记忆：支持追问（如“那它的优点呢？”），能正确理解代词指代
- 来源自动标注：回答中明确引用文档文件名或 URL，确保答案可追溯
- 智能拒答：对无关或文档未提及的问题明确回答“无法回答”，杜绝幻觉
- 高效检索：基于本地嵌入模型（BAAI/bge-small-zh-v1.5）和 Chroma 向量库，响应快速
- 友好界面：基于 Streamlit 的 Web 界面，操作简单

## 技术架构

本系统采用经典的 RAG 架构，不依赖 Agent，确保稳定性和可解释性。

| 组件 | 技术选型 |
|------|----------|
| 文档加载 | PyPDFLoader, WebBaseLoader |
| 文本分割 | RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200) |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 (HuggingFace) |
| 向量存储 | Chroma (本地持久化) |
| 检索器 | 相似度检索 (k=4) |
| 大语言模型 | DeepSeek API (deepseek-chat) |
| 前端界面 | Streamlit |
| 对话历史 | 手动管理（纯 Python 列表） |

工作流程：
1. 用户上传 PDF 或输入网页 URL → 系统加载并分割文档
2. 生成嵌入向量 → 存入 Chroma 向量库
3. 用户提问 → 系统检索相关文档片段
4. 将检索结果 + 对话历史 → 发送给 LLM
5. LLM 生成答案（自动标注来源）→ 返回给用户

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

1. 在左侧边栏**上传一个或多个 PDF 文件**（支持多选）。你也可以直接使用项目自带的 `docs/` 目录下的测试 PDF 文件。
2. 可选：输入一个**网页 URL**（如技术博客、文档页面）。
3. 点击 **“构建/重建向量库”**，等待完成。

### 对话问答

- 在底部输入框输入问题。
- 系统会自动判断：
  - **闲聊**（你好、谢谢等）→ 直接回答，不检索。
  - **文档问题**（LoRA、微调、数据准备等）→ 调用检索工具，基于文档回答。
- 答案后方会显示**引用来源**（PDF 文件名或 URL）。

### 示例对话

| 用户 | 助手 |
|------|------|
| LoRA 微调方法的核心思想是什么？ | LoRA 通过在原始权重矩阵旁路添加低秩分解矩阵来近似参数更新，仅训练不到 1% 的参数...（来源：PEFT_Methods.pdf） |
| 那它的优点有哪些？ | （系统自动理解“它”指代 LoRA）优点包括参数效率极高、推理零延迟、防止灾难性遗忘...（来源：PEFT_Methods.pdf） |
| 结合第二份 PDF，使用这种方法时需要注意什么？ | 数据量和微调方法（如 LoRA）的选择需要联合调优...（来源：Data_and_Eval.pdf） |
| 你能告诉我明天会发生什么吗？ | 抱歉，无法预测未来事件，只能基于文档内容回答。 |

### 清空对话

点击侧边栏 **“一键清空对话历史”** 重置记忆，开始新对话。


