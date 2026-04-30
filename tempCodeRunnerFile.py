# 为了防止重复执行第一阶段，你可以选择注释掉第一阶段的 main 逻辑
    # 或者简单地在这里直接调用第二阶段的函数（假设 splits 变量已通过某种方式传递）
    
    # 【简单起见，我们在这里重新运行一遍加载逻辑，然后接上存储逻辑】
    # 实际工程中我们会模块化，但 Demo 这样写最直观
    
    FILE_PATH = "D:\\Projects\\RAG_Demo\\教学培养方案.pdf" # 确保路径正确
    
    # 1. 加载与切片 (复用第一阶段的逻辑)
    # 注意：这里需要把第一阶段的 load_and_split_document 函数作用域包含进来
    # 如果你是把代码追加在同一个文件里，直接调用即可
    splits = load_and_split_document(FILE_PATH)
    if splits:
        # 2. 向量化与存储
        vectorstore = create_vector_store(splits)
        print("\n🚀 恭喜！RAG 系统的‘记忆’部分已完成构建。")
        print("下一步：接入大模型 (DeepSeek)，让它根据检索到的内容回答问题。")
    else:
        print("\n❌ 无法进行向量化，因为文档加载失败。")
        
        print("🚀 开始构建向量库...")
    vectorstore = create_vector_store() # 确保这个函数返回了 vectorstore 对象
    if vectorstore:
        print("\n✅ 向量库就绪！\n")
        
        # --- 在这里进行测试提问 ---
        # 将问题改回与培养方案相关的内容
        test_question = "本专业的培养目标是什么？" 
        ask_deepseek(vectorstore, test_question)
        
        # 如果你想无限循环提问，可以取消下面注释：
        # while True:
        #     q = input("\n请输入关于培养方案的问题 (输入 'q' 退出): ")
        #     if q.lower() == 'q': break
        #     ask_deepseek(vectorstore, q)
    else:
        print("❌ 向量库构建失败，无法进行问答。")