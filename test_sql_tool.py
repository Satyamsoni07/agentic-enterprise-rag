from src.tools.sql_tool import execute_query


query = """
SELECT
    product,
    SUM(quantity * unit_price) AS total_sales
FROM orders
GROUP BY product
ORDER BY total_sales DESC
"""


results = execute_query(query)


print("\nQuery Results:")

for row in results:
    print(row)