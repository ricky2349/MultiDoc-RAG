from pathlib import Path
from main import query_rag, init_vectorstore, clear_history

QUESTIONS_FILE = Path(__file__).with_name("test_questions.txt")

def load_questions(file_path: Path) -> list[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"未找到问题文件: {file_path}")
    questions = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        question = line.strip()
        if question:
            questions.append(question)
    if not questions:
        raise ValueError(f"问题文件为空: {file_path}")
    return questions

def test_rag():
    test_questions = load_questions(QUESTIONS_FILE)
    print(f"开始测试 RAG 系统（问题来源: {QUESTIONS_FILE.name}）...\n")
    for i, question in enumerate(test_questions, 1):
        print(f"问题 {i}: {question}")
        try:
            answer = query_rag(question)
            print(f"回答: {answer}")
        except Exception as e:
            print(f"出错: {e}")
        print("-" * 60)

if __name__ == "__main__":
    # 自动初始化向量库（使用 ./docs/ 下的 PDF）
    docs_dir = Path("./docs/")
    if docs_dir.exists():
        pdf_files = list(docs_dir.glob("*.pdf"))
        if pdf_files:
            init_vectorstore(file_paths=pdf_files, web_url=None)
        else:
            print("警告：未找到 PDF 文件，无法初始化向量库。")
    else:
        print("警告：docs 目录不存在。")
    test_rag()