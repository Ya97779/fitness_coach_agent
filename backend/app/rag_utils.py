import os
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredImageLoader,
    DirectoryLoader
)
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings for Zhipu AI (GLM)
# Zhipu AI uses embedding-2 or embedding-3
embeddings = OpenAIEmbeddings(
    model="embedding-2", 
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# Persistent directory for Chroma
CHROMA_DIR = "./chroma_db"
KNOWLEDGE_BASE_DIR = "./knowledge_base"

def load_knowledge_base():
    """从 knowledge_base 目录加载所有支持的文档。"""
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        return []

    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".jpg": UnstructuredImageLoader,
        ".jpeg": UnstructuredImageLoader,
        ".png": UnstructuredImageLoader,
    }

    documents = []
    for file in os.listdir(KNOWLEDGE_BASE_DIR):
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, file)
        ext = os.path.splitext(file)[1].lower()
        if ext in loaders:
            try:
                print(f"Loading document: {file_path}")
                loader = loaders[ext](file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return documents

def init_rag(force_rebuild=False):
    """初始化 RAG 向量库。
    
    Args:
        force_rebuild: 是否强制重新构建向量库。
    """
    if force_rebuild and os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    # 如果向量库已存在，直接加载
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("Loading existing vector store from", CHROMA_DIR)
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        return vectorstore

    print("Creating new vector store...")
    # 默认的一些硬编码知识
    initial_texts = [
        "硬拉时腰疼通常是因为核心没有收紧或者背部弯曲。建议减轻重量，专注于保持脊柱中立。",
        "增肌需要热量盈余，通常建议在 TDEE 基础上增加 200-500 卡路里。",
        "肌酸的最佳服用时间是训练前后，每天 3-5 克，不需要经过‘冲击期’。",
        "减脂的关键是热量缺口，建议每周减重 0.5-1kg 以保持肌肉。",
        "蛋白质摄入建议：增肌人群建议每天每公斤体重摄入 1.6-2.2 克蛋白质。",
        "深蹲时膝盖内扣会导致韧带损伤，应保持膝盖与脚尖方向一致。"
    ]
    initial_docs = [Document(page_content=t, metadata={"source": "manual"}) for t in initial_texts]

    # 加载外部文档
    external_docs = load_knowledge_base()
    all_docs = initial_docs + external_docs

    if not all_docs:
        # 如果没有任何文档，返回一个空的或者包含默认信息的向量库
        all_docs = [Document(page_content="欢迎使用私人营养师与健身教练系统。您可以向我询问健康、营养和健身相关的问题。", metadata={"source": "system"})]

    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(all_docs)

    # 创建并持久化向量库
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    return vectorstore

# 初始化单例 vectorstore
vectorstore = init_rag()

def rag_medical_search(query: str):
    """从本地知识库检索专业的健身和营养建议。"""
    if not query or not query.strip():
        return "请提供一个具体的查询，以便我能从知识库中检索相关信息。"
    
    try:
        # 搜索最相关的 3 个文档块
        results = vectorstore.similarity_search(query, k=3)
        if not results:
            return "在知识库中未找到相关信息。"
        
        context = "\n\n".join([doc.page_content for doc in results])
        return context
    except Exception as e:
        print(f"RAG search error: {e}")
        return f"检索过程中出现错误: {e}"
    print(f"RAG Search Query: {query}")
    # 增加检索出的文档数量 k=4，并包含来源信息（如果有的话）
    results = vectorstore.similarity_search(query, k=4)
    
    response_parts = []
    for doc in results:
        source = doc.metadata.get("source", "未知来源")
        response_parts.append(f"[{source}]: {doc.page_content}")
        
    return "\n\n".join(response_parts)
