from src.tools.sql_validator import validate_sql


queries = [
    "SELECT * FROM orders;",
    "SELECT COUNT(*) FROM orders;",
    "DELETE FROM orders;",
    "DROP TABLE orders;",
    "UPDATE orders SET unit_price = 0;"
]


for query in queries:

    is_safe, message = validate_sql(query)

    print("\n" + "=" * 60)

    print("Query:")
    print(query)

    print("Safe:")
    print(is_safe)

    print("Message:")
    print(message)