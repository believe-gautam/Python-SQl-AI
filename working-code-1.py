import chainlit as cl
from llama_index.llms.groq import Groq
import mysql.connector
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import plotly.express as px
from datetime import datetime

# Load environment variables
load_dotenv()

class DatabaseConnection:
    @staticmethod
    def get_connection():
        try:
            conn = mysql.connector.connect(
                host=os.getenv('DB_HOST'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD'),
                database=os.getenv('DB_NAME')
            )
            return conn
        except Exception as e:
            print(f"Database connection error: {e}")
            raise

class Analytics:
    def __init__(self):
        self.llm = Groq(
            model="llama3-70b-8192",
            api_key=os.getenv('GROQ_API_KEY')
        )
        # Define metric mappings
        self.metrics = {
            'sales': {
                'keywords': ['sales', 'revenue', 'earning'],
                'columns': {
                    'daily': [
                        '`Deliveroo Sales (£)`',
                        '`Uber Sales (£)`',
                        '`JustEat Sales (£)`',
                        '`Overall Sales (£)`'
                    ]
                }
            },
            'prep_time': {
                'keywords': ['prep', 'preparation', 'cooking', 'time'],
                'columns': {
                    'daily': [
                        '`Deliveroo Prep Time (min)`',
                        '`Uber Prep Time (min)`',
                        '`JustEat Prep Time`',
                        '`Overall Average Prep Time (Min)`'
                    ]
                }
            },
            'open_rate': {
                'keywords': ['open rate', 'acceptance'],
                'columns': {
                    'daily': [
                        '`Deliveroo - open rate (%)`',
                        '`Uber - open rate (%)`',
                        '`JustEat - open rate (%)`',
                        '`Overall average open rate (%)`'
                    ]
                }
            },
            'rating': {
                'keywords': ['rating', 'ratings', 'score'],
                'columns': {
                    'weekly': [
                        '`Deliveroo rating`',
                        '`Uber rating`',
                        '`JustEat rating`',
                        '`Overall average rating`'
                    ]
                }
            }
        }

    def get_metric_data(self, metric: str, time_range='week'):
        """Get data for a specific metric"""
        if metric not in self.metrics:
            return pd.DataFrame()

        date_clause = "7" if time_range == 'week' else "30"
        table = 'weekly_output' if metric in ['rating'] else 'daily_output'
        date_col = 'Week Start' if table == 'weekly_output' else 'Date'
        
        columns = [date_col] + self.metrics[metric]['columns']['daily' if table == 'daily_output' else 'weekly']
        
        query = f"""
        SELECT {', '.join(columns)}
        FROM {table}
        WHERE {date_col} >= DATE_SUB(CURDATE(), INTERVAL {date_clause} DAY)
        ORDER BY {date_col} DESC;
        """
        
        return self._execute_query(query)

    def _execute_query(self, query):
        try:
            with DatabaseConnection.get_connection() as conn:
                return pd.read_sql(query, conn)
        except Exception as e:
            print(f"Query execution error: {e}")
            return pd.DataFrame()

    async def get_ai_analysis(self, question: str, data: pd.DataFrame = None, metric: str = None):
        """Get AI-generated analysis based on question and data"""
        prompt = f"""
        As a restaurant analytics expert, analyze this question and data:

        Question: {question}

        {f'Current {metric} data:' if metric else ''}
        {data.describe().to_string() if data is not None else 'No specific data provided'}

        Consider:
        1. Current performance
        2. Trends and patterns
        3. Platform comparisons
        4. Areas for improvement

        Provide a clear, actionable response focused on restaurant operations.
        """

        response = await cl.make_async(self.llm.complete)(prompt)
        return response.text

    def identify_metric(self, question: str):
        """Identify which metric the question is asking about"""
        question = question.lower()
        for metric, info in self.metrics.items():
            if any(keyword in question for keyword in info['keywords']):
                return metric
        return None

    def generate_metric_response(self, data: pd.DataFrame, metric: str) -> str:
        """Generate response for metric-based questions"""
        if data.empty:
            return "Sorry, I couldn't find any data for that metric."

        response = f"Here are the current {metric.replace('_', ' ')} metrics:\n\n"
        
        # Get numeric columns excluding date columns
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                       if not any(date_word in col.lower() for date_word in ['date', 'week'])]

        for col in numeric_cols:
            avg_val = data[col].mean()
            platform = col.split('(')[0].strip().title()
            
            if 'sales' in col.lower() or '£' in col:
                response += f"• {platform}: £{avg_val:,.2f}\n"
            elif 'time' in col.lower() or 'min' in col.lower():
                response += f"• {platform}: {avg_val:.1f} minutes\n"
            elif '%' in col:
                response += f"• {platform}: {avg_val:.1f}%\n"
            else:
                response += f"• {platform}: {avg_val:.1f}\n"

        return response

    def create_visualization(self, data: pd.DataFrame, metric: str):
        """Create visualization for metric data"""
        if data.empty:
            return None

        date_col = next((col for col in data.columns if any(date_word in col.lower() 
                                                          for date_word in ['date', 'week'])), None)
        if not date_col:
            return None

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        fig = px.line(data,
                     x=date_col,
                     y=numeric_cols,
                     title=f"{metric.replace('_', ' ').title()} Trends",
                     labels={'value': metric.replace('_', ' ').title(),
                            'variable': 'Platform'})

        fig.update_layout(
            showlegend=True,
            legend_title_text='Platform',
            hovermode='x unified'
        )

        return fig

    async def process_request(self, user_message: str):
        try:
            # Identify the metric from the question
            metric = self.identify_metric(user_message)
            
            # For metric-based questions
            if metric:
                data = self.get_metric_data(metric)
                if data.empty:
                    return {
                        'success': False,
                        'message': "Sorry, I couldn't find any data for that metric."
                    }

                # Generate basic metric response
                response = self.generate_metric_response(data, metric)
                
                # If it's an analytical question, add AI analysis
                if any(word in user_message.lower() for word in ['why', 'how', 'what should', 'improve', 'analyze']):
                    ai_analysis = await self.get_ai_analysis(user_message, data, metric)
                    response += f"\n\nAnalysis:\n{ai_analysis}"

                return {
                    'success': True,
                    'response': response,
                    'figure': self.create_visualization(data, metric)
                }
            
            # For non-metric questions, provide AI analysis
            if any(word in user_message.lower() for word in ['why', 'how', 'what should', 'improve', 'analyze']):
                analysis = await self.get_ai_analysis(user_message)
                return {
                    'success': True,
                    'response': analysis
                }

            return {
                'success': False,
                'message': "I'm not sure what information you're looking for. Could you be more specific?"
            }

        except Exception as e:
            print(f"Error in process_request: {e}")
            return {
                'success': False,
                'message': "I encountered an error processing your request. Please try rephrasing your question."
            }

@cl.on_chat_start
async def start():
    analytics = Analytics()
    cl.user_session.set('analytics', analytics)

    await cl.Message(
        content="""👋 Hello! I can help you analyze your restaurant's performance.

You can ask me about:
• Sales and revenue
• Preparation times
• Open rates and acceptance
• Ratings and performance

Try asking:
"Show me current open rates"
"How can we improve our prep times?"
"Why are our ratings different across platforms?"
"What should we do to increase sales?"
""").send()

@cl.on_message
async def main(message: cl.Message):
    analytics = cl.user_session.get('analytics')
    result = await analytics.process_request(message.content)
    
    if result['success']:
        await cl.Message(content=result['response']).send()
        
        if 'figure' in result and result['figure'] is not None:
            await cl.Message(content="", elements=[
                cl.Plotly(name="trend", figure=result['figure'])
            ]).send()
    else:
        await cl.Message(content=result['message']).send()

if __name__ == "__main__":
    cl.run()