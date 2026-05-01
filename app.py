import streamlit as st
import tempfile
from pathlib import Path
from main import init_vectorstore, clear_history, query_rag  # 导入函数

def main():
    st.set_page_config(page_title="RAG 聊天助手", layout="wide")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "kb_ready" not in st.session_state:
        st.session_state.kb_ready = False

    st.title("RAG 聊天助手")

    with st.sidebar:
        st.header("知识库构建")
        uploaded_files = st.file_uploader("上传 PDF 文件", type=["pdf"], accept_multiple_files=True)
        web_url = st.text_input("网页 URL（可选）")

        if st.button("构建/重建向量库", use_container_width=True):
            try:
                with st.spinner("正在加载文档并构建向量库..."):
                    if uploaded_files:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            temp_paths = []
                            for file in uploaded_files:
                                tmp_path = Path(tmp_dir) / file.name
                                tmp_path.write_bytes(file.getvalue())
                                temp_paths.append(tmp_path)
                            init_vectorstore(file_paths=temp_paths, web_url=web_url.strip() if web_url else None)
                    else:
                        init_vectorstore(file_paths=None, web_url=web_url.strip() if web_url else None)
                    clear_history()
                    st.session_state.messages = []
                    st.session_state.kb_ready = True
                st.success("向量库构建完成。")
            except Exception as e:
                st.session_state.kb_ready = False
                st.error(f"构建失败：{e}")

        if st.button("一键清空对话历史"):
            clear_history()
            st.session_state.messages = []
            st.success("已清空历史。")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("请输入问题"):
        if not st.session_state.kb_ready:
            st.warning("请先构建知识库。")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    answer = query_rag(user_input)   # 直接调用
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()