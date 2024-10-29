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

    def get_sales_data(self, date_range='week'):
        date_clause = "7" if date_range == 'week' else "30"
        query = f"""
        SELECT 
            Date,
            `Deliveroo Sales (£)` as deliveroo_sales,
            `Uber Sales (£)` as uber_sales,
            `JustEat Sales (£)` as justeat_sales
        FROM daily_output 
        WHERE Date >= DATE_SUB(CURDATE(), INTERVAL {date_clause} DAY)
        ORDER BY Date DESC;
        """
        return self._execute_query(query)

    def get_prep_times(self, date_range='week'):
        date_clause = "7" if date_range == 'week' else "30"
        query = f"""
        SELECT 
            Date,
            `Deliveroo Prep Time (min)` as deliveroo_prep,
            `Uber Prep Time (min)` as uber_prep,
            `JustEat Prep Time` as justeat_prep
        FROM daily_output 
        WHERE Date >= DATE_SUB(CURDATE(), INTERVAL {date_clause} DAY)
        ORDER BY Date DESC;
        """
        return self._execute_query(query)

    def get_ratings(self, date_range='week'):
        date_clause = "7" if date_range == 'week' else "30"
        query = f"""
        SELECT 
            Date,
            `Deliveroo - open rate (%)` as deliveroo_rating,
            `Uber - open rate (%)` as uber_rating,
            `JustEat - open rate (%)` as justeat_rating
        FROM daily_output 
        WHERE Date >= DATE_SUB(CURDATE(), INTERVAL {date_clause} DAY)
        ORDER BY Date DESC;
        """
        return self._execute_query(query)

    def _execute_query(self, query):
        try:
            with DatabaseConnection.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(query)
                columns = [col[0] for col in cursor.description]
                data = cursor.fetchall()
                return pd.DataFrame(data, columns=columns)
        except Exception as e:
            print(f"Query execution error: {e}")
            return pd.DataFrame()

    async def process_request(self, message: str):
        try:
            message = message.lower()
            period = 'month' if 'month' in message else 'week'
            
            # Determine what user is asking about
            if 'sales' in message or 'revenue' in message or 'earning' in message:
                data = self.get_sales_data(period)
                metric = 'sales'
                y_axis_title = 'Sales (£)'
            elif 'prep' in message or 'time' in message:
                data = self.get_prep_times(period)
                metric = 'preparation time'
                y_axis_title = 'Time (minutes)'
            elif 'rating' in message or 'performance' in message:
                data = self.get_ratings(period)
                metric = 'ratings'
                y_axis_title = 'Rating (%)'
            else:
                data = self.get_sales_data(period)
                metric = 'performance'
                y_axis_title = 'Sales (£)'

            if data.empty:
                return {
                    'success': False,
                    'message': "Sorry, I couldn't fetch the data."
                }

            # Generate response based on specific platform or general request
            platforms = ['deliveroo', 'uber', 'justeat']
            mentioned_platform = next((p for p in platforms if p in message), None)

            if mentioned_platform:
                col = [col for col in data.columns if mentioned_platform in col.lower()][0]
                avg_value = data[col].mean()
                response = f"{mentioned_platform.title()}'s average {metric} "
                response += f"is £{avg_value:.2f}" if 'sales' in col else f"is {avg_value:.1f}"
                response += " for the "
                response += "last month." if period == 'month' else "last 7 days."
            else:
                response = f"Here's the {metric} breakdown for the "
                response += "last month:\n" if period == 'month' else "last 7 days:\n"
                for col in data.select_dtypes(include=[np.number]).columns:
                    avg = data[col].mean()
                    platform = col.split('_')[0].title()
                    if 'sales' in col:
                        response += f"• {platform}: £{avg:.2f}\n"
                    else:
                        response += f"• {platform}: {avg:.1f}\n"

            # Create visualization
            fig = None
            if not data.empty:
                # Prepare data for plotting
                plot_data = data.copy()
                plot_data['Date'] = pd.to_datetime(plot_data['Date'])
                
                # Get numeric columns for plotting
                numeric_cols = plot_data.select_dtypes(include=[np.number]).columns
                
                # Create figure
                fig = px.line(plot_data, 
                            x='Date',
                            y=numeric_cols,
                            title=f"{metric.title()} Trends",
                            labels={'value': y_axis_title,
                                   'Date': 'Date',
                                   'variable': 'Platform'})
                
                # Update layout
                fig.update_layout(
                    showlegend=True,
                    legend_title_text='Platform',
                    hovermode='x unified',
                    yaxis_title=y_axis_title,
                    xaxis_title='Date'
                )

            return {
                'success': True,
                'data': data,
                'response': response,
                'figure': fig,
                'metric': metric
            }

        except Exception as e:
            print(f"Error processing request: {e}")
            return {
                'success': False,
                'message': "I encountered an error processing your request."
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
• Platform ratings

Try asking:
"How are our sales doing?"
"What's Deliveroo's prep time?"
"Show me platform ratings"
""").send()

@cl.on_message
async def main(message: cl.Message):
    analytics = cl.user_session.get('analytics')
    result = await analytics.process_request(message.content)
    
    if result['success']:
        # First send the text response
        await cl.Message(content=result['response']).send()
        
        # Then send the visualization if available
        if 'figure' in result and result['figure'] is not None:
            elements = [
                cl.Plotly(name="trend", figure=result['figure'])
            ]
            await cl.Message(content="", elements=elements).send()
    else:
        await cl.Message(content=result['message']).send()

if __name__ == "__main__":
    cl.run()