# analytics/processor.py
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from database.connection import DatabaseConnection
from rag.query_generator import QueryGenerator

class AnalyticsProcessor:
    def __init__(self):
        self.query_generator = QueryGenerator()
        
    async def process_query(self, user_question: str) -> Dict[str, Any]:
        """Process user question and return analyzed results"""
        
        # Generate SQL query from user question
        sql_query, processing_steps = await self.query_generator.generate_query(user_question)
        
        # Execute query and get data
        with DatabaseConnection.get_connection() as conn:
            with DatabaseConnection.get_cursor(conn) as cursor:
                cursor.execute(sql_query)
                results = cursor.fetchall()
                
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Calculate basic statistics
        stats = {
            'total_rows': len(df),
            'time_range': f"{df['Date'].min()} to {df['Date'].max()}" if 'Date' in df.columns else None,
            'numerical_columns': {}
        }
        
        # Calculate statistics for numerical columns
        for col in df.select_dtypes(include=[np.number]).columns:
            stats['numerical_columns'][col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        return {
            'raw_data': df.to_dict('records'),
            'statistics': stats,
            'sql_query': sql_query,
            'processing_steps': processing_steps
        }

# analytics/insights.py
class InsightGenerator:
    def __init__(self, llm):
        self.llm = llm
        
    def generate_insights(self, data: Dict[str, Any], user_question: str) -> str:
        """Generate insights from processed data"""
        
        # Create context for the LLM
        context = f"""
        Based on the analysis of restaurant data:
        
        Time Range: {data['statistics']['time_range']}
        Total Records Analyzed: {data['statistics']['total_rows']}
        
        Key Metrics:
        """
        
        # Add key metrics to context
        for col, stats in data['statistics']['numerical_columns'].items():
            context += f"\n{col}:"
            context += f"\n  - Average: {stats['mean']:.2f}"
            context += f"\n  - Range: {stats['min']:.2f} to {stats['max']:.2f}"
        
        # Generate insights prompt
        prompt = f"""
        {context}
        
        User Question: {user_question}
        
        Please provide:
        1. Key insights from this data
        2. Recommendations for improvement
        3. Notable trends or patterns
        4. Areas that need attention
        
        Focus on practical, actionable insights relevant to restaurant operations.
        """
        
        # Get insights from LLM
        response = self.llm.complete(prompt)
        return response