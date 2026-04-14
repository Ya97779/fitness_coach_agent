import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="embedding-2", # Zhipu AI embedding model
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# Persistent directory for Chroma
CHROMA_DIR = "./chroma_db"

def init_rag():
    # If DB doesn't exist, create it with some mock data
    if not os.path.exists(CHROMA_DIR):
        texts = [
            "硬拉时腰疼通常是因为核心没有收紧或者背部弯曲。建议减轻重量，专注于保持脊柱中立。",
            "增肌需要热量盈余，通常建议在 TDEE 基础上增加 200-500 卡路里。",
            "肌酸的最佳服用时间是训练前后，每天 3-5 克，不需要经过‘冲击期’。",
            "减脂的关键是热量缺口，建议每周减重 0.5-1kg 以保持肌肉。",
            "蛋白质摄入建议：增肌人群建议每天每公斤体重摄入 1.6-2.2 克蛋白质。",
            "深蹲时膝盖内扣会导致韧带损伤，应保持膝盖与脚尖方向一致。"
        ]
        docs = [Document(page_content=t) for t in texts]
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
    else:
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
    return vectorstore

vectorstore = init_rag()

def rag_medical_search(query: str):
    """从本地知识库检索专业的健身和营养建议。"""
    if not query or not query.strip():
        return "请提供一个具体的查询，以便我能从知识库中检索相关信息。"
    print(f"RAG Search Query: {query}")
    results = vectorstore.similarity_search(query, k=2)
    return "\n".join([r.page_content for r in results])
