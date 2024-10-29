from llama_index.core import StorageContext, VectorStoreIndex
from embeddings import chunk_data
from data_loader import load_data

def build_index():
    documents = load_data()
    nodes = chunk_data(documents)
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir="./storage")
    return index

def load_index():
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    return VectorStoreIndex.load_from_storage(storage_context)
