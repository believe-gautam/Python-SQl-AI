from llama_index.core import ServiceContext
from index import load_index

def setup_query_engine():
    index = load_index()
    service_context = ServiceContext.from_defaults()
    return index.as_query_engine(service_context=service_context, similarity_top_k=10)

def setup_chat_engine():
    index = load_index()
    service_context = ServiceContext.from_defaults()
    return index.as_chat_engine(service_context=service_context, similarity_top_k=10)
