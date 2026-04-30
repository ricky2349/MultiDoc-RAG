import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.agents import AgentType, initialize_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
CHITCHAT_KEYWORDS = ("你好", "您好", "嗨", "hello", "hi", "早上好", "晚上好")
DAILY_CHAT_HINTS = (
    "天气",
    "下雨",
    "晴天",
    "今天",
    "吃饭",
    "睡觉",
    "上班",
    "下班",
    "周末",
    "心情",
)
DOC_RELATED_HINTS = (
    "文档",
    "pdf",
    "网页",
    "资料",
    "论文",
    "lora",
    "adapter",
    "prefix",
    "微调",
    "来源",
)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.2,
        timeout=60,
    )


def load_pdf_documents(uploaded_files):
    docs = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for file in uploaded_files:
            temp_path = Path(tmp_dir) / file.name
            temp_path.write_bytes(file.getvalue())
            file_docs = PyPDFLoader(str(temp_path)).load()
            for doc in file_docs:
                doc.metadata["source"] = file.name
            docs.extend(file_docs)
    return docs


def load_web_documents(url: str):
    if not url:
        return []
    web_docs = WebBaseLoader(url).load()
    for doc in web_docs:
        doc.metadata["source"] = url
    return web_docs


def build_vectorstore(uploaded_files, url: str):
    docs = []
    if uploaded_files:
        docs.extend(load_pdf_documents(uploaded_files))
    if url:
        docs.extend(load_web_documents(url))

    if not docs:
        raise ValueError("请至少上传一个 PDF，或输入一个网页 URL。")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, get_embeddings())
    return vectorstore


def format_context_with_source(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "未知来源")
        formatted.append(f"[来源: {source}]\n{doc.page_content.strip()}")
    return "\n\n".join(formatted)


def make_document_retriever_tool():
    def retrieve(question: str) -> str:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(question)

        sources = []
        seen = set()
        for doc in docs:
            source = doc.metadata.get("source", "未知来源")
            if source not in seen:
                seen.add(source)
                sources.append(source)
        st.session_state.last_sources = sources

        # 控制工具返回长度，避免上下文过长导致 Agent 推理变慢或循环
        compact_chunks = []
        for idx, doc in enumerate(docs[:3], 1):
            source = doc.metadata.get("source", "未知来源")
            content = " ".join(doc.page_content.split())
            compact_chunks.append(f"[片段{idx}][来源: {source}]\n{content[:500]}")
        return "\n\n".join(compact_chunks)

    return Tool(
        name="DocumentRetriever",
        func=retrieve,
        description=(
            "DocumentRetriever 仅用于需要查阅 PDF/网页内容的问题。"
            "当问题涉及上传文档细节、文档中的概念、参数、结论、对比信息时才调用。"
            "问候、闲聊或不依赖文档的常识问题不应调用此工具。"
            "输出为带有[来源: xxx]标记的相关文档片段。"
        ),
    )


def build_agent_executor():
    tool = make_document_retriever_tool()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        output_key="output",
        return_messages=False,
    )
    return initialize_agent(
        tools=[tool],
        llm=get_llm(),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=4,
        max_execution_time=35,
        early_stopping_method="force",
        agent_kwargs={
            "prefix": (
                "你是一个中文问答 Agent。\n"
                "如果用户只是打招呼、闲聊或提问日常常识问题，请直接回答，不要调用任何工具。\n"
                "只有在问题需要查阅上传的 PDF/网页内容时，才调用 DocumentRetriever。\n"
                "如果使用了文档证据，请在结论后标注来源（如：来源：xxx.pdf）。\n"
                "若检索结果不足以回答，请明确说“根据现有文档无法回答该问题”。\n"
                "同一问题尽量只调用一次工具，拿到结果后直接给出最终答案。"
            )
        },
    )


def is_simple_chitchat(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    if len(normalized) <= 12 and any(k in normalized for k in CHITCHAT_KEYWORDS):
        return True
    if any(k in normalized for k in DOC_RELATED_HINTS):
        return False
    if any(k in normalized for k in DAILY_CHAT_HINTS):
        return True
    if len(normalized) <= 16 and ("?" not in normalized and "？" not in normalized):
        return True
    return normalized in {"你是谁", "你是", "在吗", "在不在"}


def answer_chitchat(user_input: str) -> str:
    response = get_llm().invoke(
        "你是一个友好的中文助手。请对下面这句闲聊做简洁自然回应（1-2句）：\n"
        f"{user_input}"
    )
    return response.content


def fallback_rag_answer(question: str):
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = format_context_with_source(docs)
    prompt = (
        "你是一个中文文档问答助手。请严格基于给定上下文回答问题。"
        "如果上下文不足，请明确回答“根据现有文档无法回答该问题”。"
        "回答尽量简洁，并在关键结论后标注来源。\n\n"
        f"上下文：\n{context}\n\n"
        f"问题：{question}"
    )
    answer = get_llm().invoke(prompt).content

    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "未知来源")
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return answer, sources


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "kb_ready" not in st.session_state:
        st.session_state.kb_ready = False
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []


def main():
    st.set_page_config(page_title="RAG 聊天助手", page_icon="💬", layout="wide")
    init_session_state()

    st.title("💬 RAG 聊天助手")
    st.caption("上传多个 PDF + 可选网页 URL，构建向量库并进行多轮问答")

    with st.sidebar:
        st.header("知识库构建")
        uploaded_files = st.file_uploader(
            "上传 PDF 文件（可多选）",
            type=["pdf"],
            accept_multiple_files=True,
        )
        web_url = st.text_input("网页 URL（可选）", placeholder="https://example.com")

        if st.button("构建/重建向量库", use_container_width=True):
            try:
                with st.spinner("正在加载文档并构建向量库..."):
                    st.session_state.vectorstore = build_vectorstore(uploaded_files, web_url.strip())
                    st.session_state.agent_executor = build_agent_executor()
                    st.session_state.kb_ready = True
                st.success("向量库构建完成。")
            except Exception as e:
                st.session_state.kb_ready = False
                st.session_state.agent_executor = None
                st.error(f"构建失败：{e}")

        st.divider()
        if st.button("一键清空对话历史", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.agent_executor is not None:
                st.session_state.agent_executor.memory.clear()
            st.success("对话历史已清空。")

        if API_KEY:
            st.info("DeepSeek API Key 已加载")
        else:
            st.error("未检测到 DEEPSEEK_API_KEY，请检查 .env")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption("引用来源: " + " | ".join(message["sources"]))

    if user_input := st.chat_input("请输入你的问题..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            if not API_KEY:
                st.error("无法回答：未配置 DEEPSEEK_API_KEY。")
                return
            if not st.session_state.kb_ready or st.session_state.vectorstore is None:
                st.warning("请先在左侧上传 PDF/输入 URL，并点击“构建/重建向量库”。")
                return
            if st.session_state.agent_executor is None:
                st.warning("Agent 尚未初始化，请先构建向量库。")
                return

            with st.spinner("正在检索并生成回答..."):
                try:
                    st.session_state.last_sources = []
                    if is_simple_chitchat(user_input):
                        answer = answer_chitchat(user_input)
                        sources = []
                    else:
                        result = st.session_state.agent_executor.invoke({"input": user_input})
                        answer = result.get("output", "未获得有效回答。")
                        sources = st.session_state.last_sources

                    if "Agent stopped due to iteration limit or time limit" in answer:
                        answer, sources = fallback_rag_answer(user_input)
                    st.markdown(answer)
                    if sources:
                        st.caption("引用来源: " + " | ".join(sources))
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    st.error(f"处理失败：{e}")


if __name__ == "__main__":
    main()