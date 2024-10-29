import chainlit as cl
from llama_index.llms.groq import Groq
import mysql.connector
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

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

class MetricsConfig:
    """Configuration for all metrics and their properties"""
    def __init__(self):
        self.metrics = {
            'sales': {
                'keywords': ['sales', 'revenue', 'earning', 'money'],
                'columns': {
                    'daily': {
                        'deliveroo': '`Deliveroo Sales (£)`',
                        'uber': '`Uber Sales (£)`',
                        'justeat': '`JustEat Sales (£)`',
                        'overall': '`Overall Sales (£)`'
                    }
                },
                'alert_thresholds': {
                    'low': 0.8,  # 20% below average
                    'high': 1.2  # 20% above average
                },
                'weight': 0.3  # Performance score weight
            },
            'prep_time': {
                'keywords': ['prep', 'preparation', 'cooking', 'time'],
                'columns': {
                    'daily': {
                        'deliveroo': '`Deliveroo Prep Time (min)`',
                        'uber': '`Uber Prep Time (min)`',
                        'justeat': '`JustEat Prep Time`',
                        'overall': '`Overall Average Prep Time (Min)`'
                    }
                },
                'alert_thresholds': {
                    'high': 1.3  # 30% above average is concerning
                },
                'weight': 0.25
            },
            'ratings': {
                'keywords': ['rating', 'ratings', 'score', 'feedback'],
                'columns': {
                    'daily': {
                        'deliveroo': '`Deliveroo rating`',
                        'uber': '`Uber rating`',
                        'justeat': '`JustEat rating`',
                        'overall': '`Overall average rating`'
                    }
                },
                'alert_thresholds': {
                    'low': 0.9  # 10% below average
                },
                'weight': 0.25
            },
            'orders': {
                'keywords': ['orders', 'volume', 'quantity'],
                'columns': {
                    'daily': {
                        'deliveroo': '`Deliveroo - number of order (Orders)`',
                        'uber': '`Uber - number of order (Orders)`',
                        'justeat': '`JustEat - number of order (Orders)`',
                        'overall': '`Overall number of order (Orders)`'
                    }
                },
                'alert_thresholds': {
                    'low': 0.7  # 30% below average
                },
                'weight': 0.2
            }
        }

class DataFetcher:
    """Handles all data retrieval operations"""
    def __init__(self):
        self.config = MetricsConfig()

    def get_metric_data(self, 
                       metric: str, 
                       time_range: str = 'week', 
                       platforms: List[str] = None) -> pd.DataFrame:
        """Get data for specific metric and platforms"""
        if metric not in self.config.metrics:
            return pd.DataFrame()

        date_clause = self._get_date_clause(time_range)
        columns = self._get_columns(metric, platforms)
        
        query = f"""
        SELECT Date, {', '.join(columns)}
        FROM daily_output
        WHERE Date >= DATE_SUB(CURDATE(), {date_clause})
        ORDER BY Date DESC;
        """
        
        return self._execute_query(query)

    def _get_date_clause(self, time_range: str) -> str:
        """Get SQL date clause based on time range"""
        mappings = {
            'day': 'INTERVAL 1 DAY',
            'week': 'INTERVAL 7 DAY',
            'month': 'INTERVAL 30 DAY',
            'quarter': 'INTERVAL 90 DAY',
            'year': 'INTERVAL 365 DAY'
        }
        return mappings.get(time_range, 'INTERVAL 7 DAY')

    def _get_columns(self, metric: str, platforms: List[str] = None) -> List[str]:
        """Get relevant columns based on metric and platforms"""
        metric_cols = self.config.metrics[metric]['columns']['daily']
        if platforms:
            return [metric_cols[p] for p in platforms if p in metric_cols]
        return list(metric_cols.values())

    def _execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        try:
            with DatabaseConnection.get_connection() as conn:
                return pd.read_sql(query, conn)
        except Exception as e:
            print(f"Query execution error: {e}")
            return pd.DataFrame()

class DataAnalyzer:
    """Handles data analysis operations"""
    def __init__(self):
        self.config = MetricsConfig()
        self.data_fetcher = DataFetcher()

    def calculate_basic_stats(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate basic statistics for each metric"""
        stats = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median()
            }
        return stats

    def detect_anomalies(self, data: pd.DataFrame, metric: str) -> List[Dict[str, Any]]:
        """Detect anomalies in the data"""
        anomalies = []
        thresholds = self.config.metrics[metric]['alert_thresholds']
        
        for col in data.select_dtypes(include=[np.number]).columns:
            mean = data[col].mean()
            std = data[col].std()
            
            # Check for values outside 2 standard deviations
            outliers = data[np.abs(data[col] - mean) > 2*std]
            
            for idx, value in outliers[col].items():
                anomalies.append({
                    'metric': col,
                    'date': data.loc[idx, 'Date'],
                    'value': value,
                    'expected_range': f"{mean - std:.2f} to {mean + std:.2f}"
                })

        return anomalies

    def predict_future_values(self, 
                            data: pd.DataFrame, 
                            days_ahead: int = 7) -> pd.DataFrame:
        """Predict future values using linear regression"""
        predictions = pd.DataFrame()
        predictions['Date'] = pd.date_range(
            start=data['Date'].max() + timedelta(days=1),
            periods=days_ahead
        )
        
        for col in data.select_dtypes(include=[np.number]).columns:
            X = np.arange(len(data)).reshape(-1, 1)
            y = data[col].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            X_future = np.arange(len(data), len(data) + days_ahead).reshape(-1, 1)
            predictions[col] = model.predict(X_future)

        return predictions

    def analyze_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in the data"""
        patterns = {}
        
        # Add day of week
        data['DayOfWeek'] = pd.to_datetime(data['Date']).dt.day_name()
        
        for col in data.select_dtypes(include=[np.number]).columns:
            daily_avg = data.groupby('DayOfWeek')[col].mean()
            patterns[col] = {
                'best_day': daily_avg.idxmax(),
                'worst_day': daily_avg.idxmin(),
                'daily_averages': daily_avg.to_dict()
            }
            
        return patterns

    def calculate_trends(self, data: pd.DataFrame) -> Dict[str, str]:
        """Calculate trends for each metric"""
        trends = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            first_half = data[col].iloc[len(data)//2:].mean()
            second_half = data[col].iloc[:len(data)//2].mean()
            
            if second_half > first_half * 1.05:
                trends[col] = 'strongly_increasing'
            elif second_half > first_half:
                trends[col] = 'increasing'
            elif second_half < first_half * 0.95:
                trends[col] = 'strongly_decreasing'
            elif second_half < first_half:
                trends[col] = 'decreasing'
            else:
                trends[col] = 'stable'
                
        return trends

class VisualizationGenerator:
    """Handles creation of all visualizations"""
    def create_trend_plot(self, data: pd.DataFrame, metric: str) -> go.Figure:
        """Create trend visualization"""
        fig = px.line(data, 
                     x='Date', 
                     y=data.select_dtypes(include=[np.number]).columns,
                     title=f"{metric.title()} Trends")
        
        fig.update_layout(
            showlegend=True,
            legend_title="Platform",
            hovermode='x unified',
            yaxis_title=metric.title(),
            xaxis_title="Date"
        )
        
        return fig

    def create_comparison_plot(self, data: pd.DataFrame, metric: str) -> go.Figure:
        """Create platform comparison visualization"""
        latest_data = data.iloc[0]
        platforms = data.select_dtypes(include=[np.number]).columns
        
        fig = go.Figure(data=[
            go.Bar(
                x=platforms,
                y=[latest_data[platform] for platform in platforms],
                text=[f"{latest_data[platform]:.2f}" for platform in platforms],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f"Platform Comparison - {metric.title()}",
            xaxis_title="Platform",
            yaxis_title=metric.title()
        )
        
        return fig

    def create_heatmap(self, data: pd.DataFrame, metric: str) -> go.Figure:
        """Create heatmap visualization"""
        data['DayOfWeek'] = pd.to_datetime(data['Date']).dt.day_name()
        data['Hour'] = pd.to_datetime(data['Date']).dt.hour
        
        platforms = data.select_dtypes(include=[np.number]).columns
        heatmaps = []
        
        for platform in platforms:
            pivot = data.pivot_table(
                values=platform,
                index='Hour',
                columns='DayOfWeek',
                aggfunc='mean'
            )
            
            heatmap = go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                name=platform,
                colorscale='RdBu'
            )
            heatmaps.append(heatmap)
        
        fig = go.Figure(data=heatmaps[0])  # Start with first heatmap
        
        fig.update_layout(
            title=f"{metric.title()} Heatmap by Day and Hour",
            xaxis_title="Day of Week",
            yaxis_title="Hour of Day"
        )
        
        return fig

class PredictiveAnalytics:
    """Handles all predictive analytics operations"""
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.config = MetricsConfig()

    async def predict_metrics(self, metric: str, days_ahead: int = 7) -> Dict[str, Any]:
        """Generate predictions for given metric"""
        historical_data = self.data_fetcher.get_metric_data(metric, time_range='month')
        if historical_data.empty:
            return {}

        predictions = {}
        confidence_intervals = {}
        
        for col in historical_data.select_dtypes(include=[np.number]).columns:
            try:
                # Prepare data for prediction
                X = np.arange(len(historical_data)).reshape(-1, 1)
                y = historical_data[col].values
                
                # Fit model
                model = LinearRegression()
                model.fit(X, y)
                
                # Generate future dates
                future_dates = pd.date_range(
                    start=historical_data['Date'].max() + timedelta(days=1),
                    periods=days_ahead
                )
                
                # Predict
                X_future = np.arange(len(historical_data), 
                                   len(historical_data) + days_ahead).reshape(-1, 1)
                y_pred = model.predict(X_future)
                
                # Calculate confidence intervals
                mse = mean_squared_error(y, model.predict(X))
                std_error = np.sqrt(mse)
                confidence_interval = 1.96 * std_error  # 95% confidence interval
                
                predictions[col] = {
                    'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                    'values': y_pred.tolist(),
                    'confidence_interval': confidence_interval
                }
                
            except Exception as e:
                print(f"Error predicting {col}: {e}")
                continue

        return predictions

    async def identify_peak_times(self, metric: str) -> Dict[str, Any]:
        """Identify peak times and patterns"""
        data = self.data_fetcher.get_metric_data(metric, time_range='month')
        if data.empty:
            return {}

        data['hour'] = pd.to_datetime(data['Date']).dt.hour
        data['day_of_week'] = pd.to_datetime(data['Date']).dt.day_name()

        peak_times = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            # Daily patterns
            daily_avg = data.groupby('hour')[col].mean()
            peak_hour = daily_avg.idxmax()
            peak_value = daily_avg.max()

            # Weekly patterns
            weekly_avg = data.groupby('day_of_week')[col].mean()
            peak_day = weekly_avg.idxmax()
            
            peak_times[col] = {
                'peak_hour': int(peak_hour),
                'peak_day': peak_day,
                'peak_value': float(peak_value),
                'hourly_pattern': daily_avg.to_dict(),
                'daily_pattern': weekly_avg.to_dict()
            }

        return peak_times

class AlertSystem:
    """Handles alert generation and monitoring"""
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.config = MetricsConfig()

    async def generate_alerts(self, metric: str) -> List[Dict[str, Any]]:
        """Generate alerts based on recent data"""
        data = self.data_fetcher.get_metric_data(metric, time_range='week')
        if data.empty:
            return []

        alerts = []
        thresholds = self.config.metrics[metric]['alert_thresholds']
        
        for col in data.select_dtypes(include=[np.number]).columns:
            recent_value = data[col].iloc[0]
            avg_value = data[col].mean()
            
            # Check thresholds
            if 'low' in thresholds and recent_value < avg_value * thresholds['low']:
                alerts.append({
                    'type': 'warning',
                    'metric': col,
                    'message': f"{col} is significantly below average: {recent_value:.2f} vs {avg_value:.2f}",
                    'priority': 'high' if recent_value < avg_value * 0.7 else 'medium'
                })
                
            if 'high' in thresholds and recent_value > avg_value * thresholds['high']:
                alerts.append({
                    'type': 'caution',
                    'metric': col,
                    'message': f"{col} is significantly above average: {recent_value:.2f} vs {avg_value:.2f}",
                    'priority': 'high' if recent_value > avg_value * 1.5 else 'medium'
                })

            # Check for trends
            recent_trend = data[col].iloc[:3].mean()
            historical_trend = data[col].iloc[3:].mean()
            
            if recent_trend < historical_trend * 0.8:
                alerts.append({
                    'type': 'trend',
                    'metric': col,
                    'message': f"Declining trend detected in {col}",
                    'priority': 'medium'
                })

        return alerts

    async def check_anomalies(self, metric: str) -> List[Dict[str, Any]]:
        """Detect anomalies in recent data"""
        data = self.data_fetcher.get_metric_data(metric, time_range='week')
        if data.empty:
            return []

        anomalies = []
        
        for col in data.select_dtypes(include=[np.number]).columns:
            mean = data[col].mean()
            std = data[col].std()
            
            # Check for values outside 2 standard deviations
            recent_value = data[col].iloc[0]
            if abs(recent_value - mean) > 2 * std:
                anomalies.append({
                    'type': 'anomaly',
                    'metric': col,
                    'message': f"Unusual value detected in {col}: {recent_value:.2f}",
                    'value': recent_value,
                    'expected_range': f"{mean - std:.2f} to {mean + std:.2f}",
                    'priority': 'high'
                })

        return anomalies

class PerformanceScoring:
    """Handles performance scoring and benchmarking"""
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.config = MetricsConfig()

    async def calculate_overall_score(self) -> Dict[str, Any]:
        """Calculate overall performance score"""
        scores = {}
        total_weight = 0
        
        for metric, config in self.config.metrics.items():
            if 'weight' not in config:
                continue
                
            data = self.data_fetcher.get_metric_data(metric, time_range='week')
            if data.empty:
                continue

            metric_score = self._calculate_metric_score(data, metric)
            weight = config['weight']
            
            scores[metric] = {
                'score': metric_score,
                'weight': weight
            }
            total_weight += weight

        # Calculate weighted average
        if total_weight > 0:
            overall_score = sum(s['score'] * s['weight'] for s in scores.values()) / total_weight
        else:
            overall_score = 0

        return {
            'overall_score': overall_score,
            'metric_scores': scores
        }

    def _calculate_metric_score(self, data: pd.DataFrame, metric: str) -> float:
        """Calculate score for specific metric"""
        if metric == 'prep_time':
            # Lower is better for prep time
            avg_time = data.select_dtypes(include=[np.number]).mean().mean()
            return max(0, min(1, 2 - (avg_time / 15)))  # Benchmark: 15 minutes
        
        elif metric == 'ratings':
            # Higher is better for ratings
            avg_rating = data.select_dtypes(include=[np.number]).mean().mean()
            return avg_rating / 5  # Normalize to 0-1
        
        elif metric == 'sales':
            # Compare to targets or historical averages
            current_avg = data.select_dtypes(include=[np.number]).mean().mean()
            historical_data = self.data_fetcher.get_metric_data(metric, time_range='month')
            if not historical_data.empty:
                historical_avg = historical_data.select_dtypes(include=[np.number]).mean().mean()
                return min(1, current_avg / historical_avg)
            return 0.5

        return 0.5  # Default score

class ReportGenerator:
    """Handles report generation"""
    def __init__(self, data_fetcher: DataFetcher, predictor: PredictiveAnalytics,
                 alert_system: AlertSystem, scorer: PerformanceScoring):
        self.data_fetcher = data_fetcher
        self.predictor = predictor
        self.alert_system = alert_system
        self.scorer = scorer

    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'alerts': [],
            'predictions': {},
            'performance_score': None
        }

        # Gather data for each metric
        for metric in self.config.metrics.keys():
            try:
                data = self.data_fetcher.get_metric_data(metric, time_range='week')
                if not data.empty:
                    report['metrics'][metric] = {
                        'current_values': data.iloc[0].to_dict(),
                        'weekly_average': data.mean().to_dict(),
                        'trends': self._calculate_trends(data)
                    }

                # Get alerts
                alerts = await self.alert_system.generate_alerts(metric)
                report['alerts'].extend(alerts)

                # Get predictions
                predictions = await self.predictor.predict_metrics(metric)
                report['predictions'][metric] = predictions

            except Exception as e:
                print(f"Error processing {metric}: {e}")
                continue

        # Calculate overall performance
        try:
            report['performance_score'] = await self.scorer.calculate_overall_score()
        except Exception as e:
            print(f"Error calculating performance score: {e}")

        return report

    def _calculate_trends(self, data: pd.DataFrame) -> Dict[str, str]:
        """Calculate trends for metrics"""
        trends = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            if len(data) >= 2:
                recent = data[col].iloc[0]
                previous = data[col].iloc[-1]
                change = (recent - previous) / previous if previous != 0 else 0
                
                if change > 0.1:
                    trends[col] = 'increasing'
                elif change < -0.1:
                    trends[col] = 'decreasing'
                else:
                    trends[col] = 'stable'
            else:
                trends[col] = 'insufficient_data'
                
        return trends


import chainlit as cl
from llama_index.llms.groq import Groq
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

class ConversationalAI:
    """Handles natural conversation and user interactions"""
    def __init__(self, llm):
        self.llm = llm
        self.setup_conversation_patterns()

    def setup_conversation_patterns(self):
        """Setup conversation patterns and responses"""
        self.patterns = {
            'greetings': {
                'triggers': ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'],
                'response': """👋 Hello! I'm your Restaurant Analytics Assistant.

I can help you analyze:
• Sales performance across platforms
• Preparation times and efficiency
• Customer ratings and feedback
• Order volumes and trends
• Platform comparisons

Feel free to ask me anything about your restaurant's performance!

For example, try:
"How are our sales doing?"
"Show me preparation times"
"Compare platform ratings"
"Which platform is performing best?"
"""
            },
            'farewells': {
                'triggers': ['bye', 'goodbye', 'see you', 'thanks', 'thank you'],
                'response': "Thanks for chatting! Let me know if you need any more analytics or insights about your restaurant's performance."
            },
            'help': {
                'triggers': ['help', 'what can you do', 'what do you do'],
                'response': """I'm here to help you analyze your restaurant's performance across Deliveroo, UberEats, and JustEat.

I can:
1. Analyze sales and revenue trends
2. Monitor preparation times
3. Track customer ratings
4. Identify peak hours
5. Compare platform performance
6. Provide improvement suggestions

What would you like to know about?"""
            }
        }

    async def get_response(self, message: str, context: Dict = None) -> Dict[str, Any]:
        """Generate contextual response based on user message"""
        message_lower = message.lower()

        # Check for conversation patterns
        for pattern_type, pattern_data in self.patterns.items():
            if any(trigger in message_lower for trigger in pattern_data['triggers']):
                return {
                    'type': 'conversation',
                    'response': pattern_data['response']
                }

        # If not a basic pattern, check for context-aware response
        if context:
            return await self.get_contextual_response(message, context)

        # If no pattern match and no context, return None to handle as analytics
        return None

    async def get_contextual_response(self, message: str, context: Dict) -> Dict[str, Any]:
        """Generate context-aware response"""
        try:
            prompt = f"""
            Given this context about a restaurant analytics conversation:
            Previous topic: {context.get('previous_topic', 'None')}
            Last metric discussed: {context.get('last_metric', 'None')}
            Current question: {message}

            Generate a natural, helpful response that:
            1. Acknowledges any previous context
            2. Addresses the current question
            3. Maintains a professional but friendly tone
            """

            response = await cl.make_async(self.llm.complete)(prompt)
            return {
                'type': 'contextual',
                'response': response.text
            }
        except Exception as e:
            print(f"Error generating contextual response: {e}")
            return None

class UserInterface:
    """Handles UI elements and interactions"""
    def __init__(self, analytics, conversation):
        self.analytics = analytics
        self.conversation = conversation
        self.context = {}

    async def process_message(self, message: str) -> Dict[str, Any]:
        """Process user message and generate appropriate response"""
        try:
            # First check for conversational response
            conv_response = await self.conversation.get_response(message, self.context)
            if conv_response and conv_response['type'] == 'conversation':
                return conv_response

            # If not conversational, process as analytics request
            result = await self.analytics.process_request(message)
            
            # Update context
            self.update_context(message, result)
            
            return result

        except Exception as e:
            print(f"Error processing message: {e}")
            return {
                'success': False,
                'message': "I encountered an error. Please try rephrasing your question."
            }

    def update_context(self, message: str, result: Dict[str, Any]):
        """Update conversation context"""
        self.context.update({
            'last_message': message,
            'last_metric': result.get('metric'),
            'timestamp': datetime.now().isoformat(),
            'success': result.get('success', False)
        })

# Chainlit UI implementation
@cl.on_chat_start
async def start():
    analytics = AdvancedAnalytics()  # From Part 1
    conversation = ConversationalAI(analytics.llm)
    interface = UserInterface(analytics, conversation)
    
    cl.user_session.set('interface', interface)

    await cl.Message(
        content="""👋 Hello! I'm your Restaurant Analytics Assistant.

I can help you analyze your restaurant's performance across Deliveroo, UberEats, and JustEat.

Try asking me:
• "How are our sales doing?"
• "Show me preparation times"
• "Compare platform ratings"
• "Analyze our performance"
• "How can we improve?"

Or just chat with me - I'm here to help!""").send()

@cl.on_message
async def main(message: cl.Message):
    interface = cl.user_session.get('interface')
    result = await interface.process_message(message.content)
    
    if result.get('type') == 'conversation':
        await cl.Message(content=result['response']).send()
        return

    if result.get('success'):
        # Send main analysis response
        await cl.Message(content=result['response']).send()
        
        # Send visualizations if available
        if result.get('visualizations'):
            for viz in result.get('visualizations'):
                elements = [cl.Plotly(name="analysis", figure=viz)]
                await cl.Message(content="", elements=elements).send()
        
        # Send alerts if any
        if result.get('alerts'):
            alert_msg = "\n🚨 Important Alerts:\n" + "\n".join(result['alerts'])
            await cl.Message(content=alert_msg).send()
            
        # Send suggestions if any
        if result.get('suggestions'):
            suggest_msg = "\n💡 Suggestions:\n" + "\n".join(result['suggestions'])
            await cl.Message(content=suggest_msg).send()
    else:
        await cl.Message(content=result.get('message', 
            "I'm not sure I understood that. Could you try asking about specific metrics?")).send()