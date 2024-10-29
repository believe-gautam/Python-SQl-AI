from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser

def get_embeddings_model():
    return GeminiEmbedding(model_name="models/embedding-001")

def chunk_data(documents):
    embed_model = get_embeddings_model()
    splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model)
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    return nodes
