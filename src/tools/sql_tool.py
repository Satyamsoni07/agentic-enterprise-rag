import sqlite3
from pathlib import Path


DATABASE_PATH = Path("data/enterprise.db")


def execute_query(query: str) -> list:

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute(query)

    rows = cursor.fetchall()

    connection.close()

    return rows