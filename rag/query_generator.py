# rag/query_generator.py
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.groq import Groq
from database.schema_loader import SchemaManager
from typing import Tuple, List
import re

class QueryGenerator:
    def __init__(self):
        self.schema_manager = SchemaManager()
        self.embed_model = GeminiEmbedding(
            model_name="models/embedding-001",
            api_key=Config.GEMINI_API_KEY
        )
        self.llm = Groq(
            model="llama3-70b-8192",
            api_key=Config.GROQ_API_KEY
        )

        # Create context for the LLM
        self.db_context = self.schema_manager.get_column_context()
        
        # Template for SQL generation
        self.sql_template = """
        Based on the following database schema:
        {db_context}

        Generate a SQL query to answer this question: {question}

        Consider:
        1. Time periods if mentioned
        2. Specific platforms (Deliveroo, UberEats, JustEat) if mentioned
        3. Appropriate aggregations needed
        4. Any specific metrics or calculations required

        Return the SQL query and any additional processing steps needed.
        """

    async def generate_query(self, user_question: str) -> Tuple[str, List[str]]:
        """Generate SQL query and processing steps for user question"""
        
        prompt = self.sql_template.format(
            db_context=self.db_context,
            question=user_question
        )

        # Get response from LLM
        response = self.llm.complete(prompt)
        
        # Extract SQL query and processing steps
        # (You might want to add more sophisticated parsing here)
        sql_match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL)
        sql_query = sql_match.group(1) if sql_match else None
        
        # Extract any additional processing steps
        processing_steps = []
        if "Processing steps:" in response:
            steps_text = response.split("Processing steps:")[1]
            processing_steps = [step.strip() for step in steps_text.split("\n") if step.strip()]

        return sql_query, processing_steps