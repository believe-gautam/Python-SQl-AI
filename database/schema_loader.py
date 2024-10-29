# database/schema_loader.py
from typing import Dict, List
from dataclasses import dataclass
from config.config import Config
import mysql.connector

@dataclass
class TableSchema:
    name: str
    columns: Dict[str, str]  # column_name: data_type
    description: str

class SchemaManager:
    def __init__(self):
        self.tables = {
            'daily_output': TableSchema(
                name='daily_output',
                columns={},
                description='Contains daily metrics for restaurant performance across Deliveroo, UberEats, and JustEat platforms'
            ),
            'weekly_output': TableSchema(
                name='weekly_output',
                columns={},
                description='Contains weekly aggregated metrics and detailed quality indicators across delivery platforms'
            )
        }
        self._load_schema()

    def _load_schema(self):
        """Load schema information from database"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                for table_name in self.tables.keys():
                    cursor.execute(f"""
                        SELECT COLUMN_NAME, DATA_TYPE, COLUMN_COMMENT
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_NAME = '{table_name}'
                        AND TABLE_SCHEMA = '{Config.DB_NAME}'
                    """)
                    columns = cursor.fetchall()
                    self.tables[table_name].columns = {
                        col[0]: {
                            'data_type': col[1],
                            'description': col[2] if col[2] else f'Column containing {col[0]}'
                        } for col in columns
                    }

    def get_column_context(self) -> str:
        """Generate context about available data for the LLM"""
        context = []
        for table_name, schema in self.tables.items():
            context.append(f"\nTable: {table_name}")
            context.append(schema.description)
            context.append("\nAvailable columns:")
            for col_name, col_info in schema.columns.items():
                context.append(f"- {col_name} ({col_info['data_type']}): {col_info['description']}")
        return "\n".join(context)

    @staticmethod
    def _get_connection():
        return mysql.connector.connect(
            host=Config.DB_HOST,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME
        )