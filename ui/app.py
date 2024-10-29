# ui/app.py
import chainlit as cl
from analytics.processor import AnalyticsProcessor
from analytics.insights import InsightGenerator
from llama_index.llms.groq import Groq
from config.config import Config
import plotly.express as px
import pandas as pd

analytics_processor = None
insight_generator = None

@cl.on_chat_start
async def setup():
    global analytics_processor, insight_generator
    
    # Initialize processor and LLM
    analytics_processor = AnalyticsProcessor()
    llm = Groq(model="llama3-70b-8192", api_key=Config.GROQ_API_KEY)
    insight_generator = InsightGenerator(llm)
    
    # Welcome message
    await cl.Message(
        content="Welcome to Restaurant Analytics! I can help you analyze your restaurant data across Deliveroo, UberEats, and JustEat. What would you like to know?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    try:
        # Process the query
        results = await analytics_processor.process_query(message.content)
        
        # Generate insights
        insights = insight_generator.generate_insights(results, message.content)
        
        # Create visualizations if applicable
        if results['raw_data']:
            df = pd.DataFrame(results['raw_data'])
            
            # Time series plot if date column exists
            if 'Date' in df.columns:
                for col in df.select_dtypes(include=[np.number]).columns:
                    fig = px.line(df, x='Date', y=col, title=f'{col} Over Time')
                    await cl.Plot(figure=fig).send()
        
        # Send response
        await cl.Message(content=f"""
        📊 Analysis Results:
        
        {insights}
        
        💡 SQL Query Used:
        ```sql
        {results['sql_query']}
        ```
        """).send()
        
    except Exception as e:
        await cl.Message(
            content=f"I encountered an error while processing your request: {str(e)}"
        ).send()

if __name__ == "__main__":
    cl.run()