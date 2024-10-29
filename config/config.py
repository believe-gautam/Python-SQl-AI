# config/config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # Database configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_NAME = os.getenv('DB_NAME')
    
    # API Keys
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# database/connection.py
import mysql.connector
from config.config import Config
from contextlib import contextmanager

class DatabaseConnection:
    @contextmanager
    def get_connection():
        conn = mysql.connector.connect(
            host=Config.DB_HOST,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME
        )
        try:
            yield conn
        finally:
            conn.close()
            
    @contextmanager
    def get_cursor(conn):
        cursor = conn.cursor(dictionary=True)
        try:
            yield cursor
        finally:
            cursor.close()