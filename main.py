import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 全局变量（惰性初始化）
_vectorstore = None
_retriever = None
_llm = None
_embeddings = None
_chat_history = []

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("加载本地嵌入模型...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings

def get_llm():
    global _llm
    if _llm is None:
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")
        _llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=deepseek_api_key,
            openai_api_base="https://api.deepseek.com/v1",
            temperature=0,
        )
    return _llm

def build_vectorstore_from_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, get_embeddings())
    return vectorstore

def load_documents_from_sources(file_paths=None, web_url=None):
    all_docs = []
    if file_paths:
        for path in file_paths:
            loader = PyPDFLoader(str(path))
            for doc in loader.load():
                doc.metadata["source"] = Path(path).name
                all_docs.append(doc)
    if web_url:
        loader = WebBaseLoader(web_url)
        for doc in loader.load():
            doc.metadata["source"] = web_url
            all_docs.append(doc)
    if not all_docs:
        raise ValueError("未提供任何有效文档。")
    return all_docs

def init_vectorstore(file_paths=None, web_url=None):
    global _vectorstore, _retriever
    docs = load_documents_from_sources(file_paths, web_url)
    _vectorstore = build_vectorstore_from_docs(docs)
    _retriever = _vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 12}
)
def format_docs(docs):
    if not docs:
        return "无相关文档"
    seen = set()
    formatted = []
    for doc in docs:
        content = doc.page_content.strip()
        source = doc.metadata.get("source", "未知来源")
        key = (source, content[:200])
        if key in seen:
            continue
        seen.add(key)
        if len(content) > 800:
            content = content[:800] + "..."
        formatted.append(f"[来源: {source}]\n{content}")
    return "\n\n".join(formatted)

def query_rag(question: str) -> str:
    global _chat_history
    if _retriever is None:
        raise RuntimeError("向量库未初始化，请先调用 init_vectorstore。")

    history_text = ""
    for msg in _chat_history:
        role = "用户" if msg["role"] == "user" else "助手"
        history_text += f"{role}: {msg['content']}\n"

    docs = _retriever.invoke(question)
    context = format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个中文问答助手。请基于历史对话和当前检索上下文回答用户问题。如果问题包含代词，需从历史对话中确定指代对象。若上下文不足，回答“根据现有文档无法回答该问题”。回答中如需引用文档，请在每个关键结论后使用括号标注来源，格式为（来源：文件名），且全文只标注一次，不要重复。"),
        ("human", "历史对话：\n{history}\n\n当前检索上下文：\n{context}\n\n用户问题：\n{question}\n\n请直接回答：")
    ])

    chain = prompt | get_llm() | StrOutputParser()
    answer = chain.invoke({
        "history": history_text,
        "context": context,
        "question": question
    })

    _chat_history.append({"role": "user", "content": question})
    _chat_history.append({"role": "assistant", "content": answer})
    if len(_chat_history) > 10:
        _chat_history = _chat_history[-10:]

    return answer

def clear_history():
    global _chat_history
    _chat_history = []