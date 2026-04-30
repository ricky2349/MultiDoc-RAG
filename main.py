import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# 1. 加载文档（docs 文件夹下所有 PDF + 可选网页 URL）
def load_documents(docs_dir: str = "./docs/", web_url: str | None = None):
    all_docs = []
    pdf_dir = Path(docs_dir)

    if pdf_dir.exists():
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            pdf_docs = loader.load()
            for doc in pdf_docs:
                doc.metadata["source"] = pdf_file.name
            all_docs.extend(pdf_docs)
    else:
        print(f"[WARN] PDF 目录不存在: {docs_dir}")

    if web_url:
        web_loader = WebBaseLoader(web_url)
        web_docs = web_loader.load()
        for doc in web_docs:
            doc.metadata["source"] = web_url
        all_docs.extend(web_docs)

    if not all_docs:
        raise ValueError("未加载到任何文档。请检查 ./docs/ 下是否有 PDF，或提供有效 URL。")

    print(f"[INFO] 共加载文档页/片段数: {len(all_docs)}")
    return all_docs


def get_web_url_input() -> str | None:
    env_url = os.getenv("RAG_WEB_URL", "").strip()
    if env_url:
        return env_url

    if sys.stdin and sys.stdin.isatty():
        user_url = input("可选：请输入要加载的网页 URL（直接回车跳过）：").strip()
        return user_url or None

    return None


docs = load_documents("./docs/", get_web_url_input())

# 2. 分割文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# 3. 使用本地 HuggingFace embeddings
print("加载本地嵌入模型...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 4. 创建向量数据库
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 5. 配置 DeepSeek LLM
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=deepseek_api_key,
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0,
)


def format_docs(docs):
    if not docs:
        return "无相关文档"

    unique_docs = []
    seen = set()

    for doc in docs:
        content = " ".join(doc.page_content.split())
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        key = (source, page, content)
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)

    formatted_chunks = []
    for doc in unique_docs:
        source = doc.metadata.get("source", "未知来源")
        header = f"[来源: {source}]"
        formatted_chunks.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n".join(formatted_chunks)


def DocumentRetriever(question: str) -> str:
    """检索与用户问题相关的文档片段，返回带来源标记的内容。"""
    docs = retriever.invoke(question)
    return format_docs(docs)

# 6. Prompt 配置
PROMPT_V1 = """
你是一个智能问答 Agent。
你可以与用户闲聊，也可以在需要时调用工具检索文档。
只有当用户的问题需要基于文档内容、网页内容或前文提到的文档主题来回答时，才调用 DocumentRetriever。
如果只是普通闲聊、问候、简单常识或不需要文档依据的问题，不要调用任何工具，直接回答。
回答时保持简洁、自然、清晰。
""".strip()

PROMPT_V2 = """
你是一个智能问答 Agent。
你可以与用户闲聊，也可以在需要时调用工具检索文档。
只有当用户的问题需要基于文档内容、网页内容或前文提到的文档主题来回答时，才调用 DocumentRetriever。
如果只是普通闲聊、问候、简单常识或不需要文档依据的问题，不要调用任何工具，直接回答。
如果使用了工具，请严格基于检索到的内容回答，不要编造，不要把历史回答当作新的文档证据。
如果答案无法从当前检索结果中明确得到，请直接回答“根据现有文档无法回答该问题。”
如果使用了文档内容，请尽量在相关结论后标注来源，例如“（来源：PEFT_Methods.pdf）”。
回答时避免重复表述，保持简洁清晰。
""".strip()

PROMPT_V3 = """
你是一个智能问答 Agent。
你可以与用户闲聊，也可以在需要时调用工具检索文档。
只有当用户的问题需要基于文档内容、网页内容或前文提到的文档主题来回答时，才调用 DocumentRetriever。
如果只是普通闲聊、问候、简单常识或不需要文档依据的问题，不要调用任何工具，直接回答。

在需要文档回答时，请先在内部梳理问题要点，再基于检索结果作答，但不要输出你的思维过程。
如果问题涉及对比、优缺点、差异、取舍、推荐等内容，请优先使用“对比总结”的方式回答，明确列出各选项差异。
如果使用了工具，请严格基于检索到的内容回答，不要编造，不要把历史回答当作新的文档证据。
如果答案无法从当前检索结果中明确得到，请直接回答“根据现有文档无法回答该问题。”
如果使用了文档内容，请在关键结论后尽量标注来源，例如“（来源：PEFT_Methods.pdf）”。
回答时避免重复表述，保持层次清晰，优先给出结论，再补充依据。
""".strip()

# 手动切换当前使用的 prompt：
# ACTIVE_PROMPT = PROMPT_V1
# ACTIVE_PROMPT = PROMPT_V2
ACTIVE_PROMPT = PROMPT_V3

# 6. 对话记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="input",
    output_key="output",
    return_messages=True,
)

# 7. Agent
agent = create_agent(
    model=llm,
    tools=[DocumentRetriever],
    system_prompt=ACTIVE_PROMPT,
)

# 8. 对外查询函数
def query_rag(question: str) -> str:
    history = memory.load_memory_variables({"input": question}).get("chat_history", [])
    messages = [*history, HumanMessage(content=question)]
    result = agent.invoke({"messages": messages})

    output_messages = result.get("messages", [])
    answer = ""
    for message in reversed(output_messages):
        if isinstance(message, AIMessage) and message.content:
            answer = message.content
            break

    if not answer:
        raise ValueError("Agent 未返回有效回答。")

    memory.save_context({"input": question}, {"output": answer})
    return answer


if __name__ == "__main__":
    print(query_rag("你好"))