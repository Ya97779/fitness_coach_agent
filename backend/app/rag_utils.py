"""RAG 工具 - 兼容旧接口

此文件保留以兼容现有代码，
实际逻辑已迁移到 app/rag/__init__.py
"""

from .rag import ModernRAG, get_rag_instance, rag_medical_search as _rag_search

def rag_medical_search(query: str):
    """兼容旧接口"""
    return _rag_search(query)

def get_modern_rag():
    """获取现代化 RAG 实例"""
    return get_rag_instance()
