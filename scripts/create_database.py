import sqlite3
from pathlib import Path


# Project database location
DATABASE_PATH = Path("data/enterprise.db")


# Connect to SQLite
connection = sqlite3.connect(DATABASE_PATH)

cursor = connection.cursor()


# Create orders table
cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    customer_name TEXT,
    product TEXT,
    quantity INTEGER,
    unit_price REAL
)
""")


# Remove old test data
cursor.execute("DELETE FROM orders")


# Insert sample data
orders = [
    (1, "Amit", "Laptop", 1, 60000),
    (2, "Neha", "Mouse", 2, 800),
    (3, "Rahul", "Keyboard", 1, 2000),
    (4, "Amit", "Mouse", 3, 800),
    (5, "Priya", "Laptop", 1, 60000),
    (6, "Neha", "Keyboard", 2, 2000),
]


cursor.executemany("""
INSERT INTO orders (
    order_id,
    customer_name,
    product,
    quantity,
    unit_price
)
VALUES (?, ?, ?, ?, ?)
""", orders)


# Save changes
connection.commit()


# Verify data
cursor.execute("SELECT * FROM orders")

rows = cursor.fetchall()

print("\nOrders:")

for row in rows:
    print(row)


connection.close()

print("\nDatabase created successfully.")