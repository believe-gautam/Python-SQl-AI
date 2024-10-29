import os
from dotenv import load_dotenv
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

load_dotenv()

# LLM and embedding model settings
Settings.llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=os.getenv("GEMINI_API_KEY"))
Settings.text_splitter = SentenceSplitter(chunk_size=1024)
Settings.chunk_size = 512
Settings.chunk_overlap = 20
