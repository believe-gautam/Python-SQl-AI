import chainlit as cl
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.groq import Groq
import mysql.connector
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import plotly.express as px
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

class DatabaseConnection:
    @staticmethod
    def get_connection():
        try:
            conn = mysql.connector.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                database=os.getenv('DB_NAME')
            )
            print("Database connection successful")
            return conn
        except Exception as e:
            print(f"Database connection error: {e}")
            raise

class RestaurantAnalytics:
    def __init__(self):
        try:
            self.llm = Groq(
                model="llama3-70b-8192",
                api_key=os.getenv('GROQ_API_KEY')
            )
            print("Groq LLM initialized successfully")
        except Exception as e:
            print(f"LLM initialization error: {e}")
            raise

    def get_sales_data(self):
        query = """
        SELECT 
            `Date`,
            `Deliveroo Sales (£)` as deliveroo_sales,
            `Uber Sales (£)` as uber_sales,
            `JustEat Sales (£)` as justeat_sales,
            `Overall Sales (£)` as total_sales
        FROM daily_output 
        WHERE `Date` >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
        ORDER BY `Date` DESC;
        """
        print(f"Executing sales query: {query}")
        return self._execute_query(query)

    def get_prep_times(self):
        query = """
        SELECT 
            `Date`,
            `Deliveroo Prep Time (min)` as deliveroo_prep,
            `Uber Prep Time (min)` as uber_prep,
            `Overall Average Prep Time (Min)` as avg_prep_time
        FROM daily_output 
        WHERE `Date` >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
        ORDER BY `Date` DESC;
        """
        print(f"Executing prep times query: {query}")
        return self._execute_query(query)

    def get_ratings(self):
        query = """
        SELECT 
            `Date`,
            `Deliveroo - open rate (%)` as deliveroo_rating,
            `Uber - open rate (%)` as uber_rating,
            `JustEat - open rate (%)` as justeat_rating,
            `Overall average open rate (%)` as avg_rating
        FROM daily_output 
        WHERE `Date` >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
        ORDER BY `Date` DESC;
        """
        print(f"Executing ratings query: {query}")
        return self._execute_query(query)

    def _execute_query(self, query):
        try:
            with DatabaseConnection.get_connection() as conn:
                df = pd.read_sql(query, conn)
                print(f"Query executed successfully. Retrieved {len(df)} rows")
                return df
        except Exception as e:
            print(f"Query execution error: {e}")
            return pd.DataFrame()

    async def analyze_request(self, user_message: str):
        print(f"Analyzing request: {user_message}")
        try:
            user_message = user_message.lower()
            
            # Extract specific platform if mentioned
            platform = None
            if 'deliveroo' in user_message:
                platform = 'Deliveroo'
            elif 'uber' in user_message or 'ubereats' in user_message:
                platform = 'Uber'
            elif 'justeat' in user_message:
                platform = 'JustEat'

            # Determine the type of analysis
            if any(word in user_message for word in ['sales', 'revenue', 'earning', 'money']):
                df = self.get_sales_data()
                analysis_type = 'sales'
                metric = 'sales'
            elif any(word in user_message for word in ['prep', 'preparation', 'cooking', 'time']):
                df = self.get_prep_times()
                analysis_type = 'preparation times'
                metric = 'prep time'
            elif any(word in user_message for word in ['rating', 'review', 'feedback', 'performance']):
                df = self.get_ratings()
                analysis_type = 'ratings'
                metric = 'rating'
            else:
                print("Processing default sales request")
                df = self.get_sales_data()
                analysis_type = 'overall performance'
                metric = 'performance'

            if df.empty:
                return {
                    'success': False,
                    'message': "I couldn't find any data for your request."
                }

            # Create a focused prompt based on the specific question
            if platform:
                summary = f"""
                Provide a brief, focused response about {platform}'s {metric} in the last 7 days.
                
                Data:
                {df.describe().round(2).to_string()}
                
                Give only the specific information asked for in 1-2 sentences.
                Focus on the most recent average {metric} for {platform}.
                """
            else:
                summary = f"""
                Provide a brief comparison of {metric} across platforms in the last 7 days.
                
                Data:
                {df.describe().round(2).to_string()}
                
                Give a 2-3 sentence summary comparing the platforms.
                Focus on averages and key differences.
                """

            response = await cl.make_async(self.llm.complete)(summary)

            return {
                'success': True,
                'data': df,
                'insights': response.text,
                'title': f"{platform if platform else 'All Platforms'} {analysis_type.title()}"
            }

        except Exception as e:
            print(f"Error in analyze_request: {e}")
            return {
                'success': False,
                'message': f"An error occurred: {str(e)}"
            }
        
@cl.on_chat_start
async def setup():
    print("Starting new chat session")
    analyzer = RestaurantAnalytics()
    cl.user_session.set('analyzer', analyzer)
    
    await cl.Message(
        content="""👋 Hello! I'm your Restaurant Analytics Assistant.

I can help you understand your restaurant's performance. Try asking me:

📈 About Sales:
- "How are our sales doing?"
- "What were last week's sales?"
- "Compare sales between platforms"

⏱️ About Preparation Times:
- "Show our prep times"
- "How long do orders take?"
- "Which platform has fastest prep time?"

⭐ About Performance:
- "How are we performing?"
- "Show platform ratings"
- "Which platform is doing best?"
        """
    ).send()

@cl.on_message
async def main(message: cl.Message):
    print(f"Received message: {message.content}")
    analyzer = cl.user_session.get('analyzer')
    results = await analyzer.analyze_request(message.content)
    
    if results['success']:
        # Create visualizations
        if 'data' in results:
            df = results['data']
            date_col = [col for col in df.columns if 'date' in col.lower()][0]
            
            for col in df.columns:
                if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        fig = px.line(df, x=date_col, y=col, 
                                    title=f"{col.replace('_', ' ').title()}",
                                    labels={col: col.replace('_', ' ').title()})
                        await cl.Plot(figure=fig).send()
                    except Exception as e:
                        print(f"Error creating plot for {col}: {e}")

        # Send insights
        await cl.Message(
            content=f"""### {results['title']}

{results['insights']}

Would you like to know anything specific about these numbers?"""
        ).send()
    else:
        await cl.Message(content=results['message']).send()

if __name__ == "__main__":
    cl.run()