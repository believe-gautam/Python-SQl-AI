from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from llama_index.core import Document

# Connect to the MySQL database using SQLAlchemy
engine = create_engine(os.getenv("DATABASE_URL"))
Session = sessionmaker(bind=engine)
session = Session()

def load_data():
    # Query to fetch data from your table
    query = "SELECT * FROM weekly_output"
    results = session.execute(text(query)).fetchall()

    # Convert each row to a Document
    documents = [Document(content=str(row)) for row in results]
    return documents
