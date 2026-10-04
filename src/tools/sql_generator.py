from src.llm.provider import generate_response


DATABASE_SCHEMA = """
Table: orders

Columns:
- order_id: INTEGER
- customer_name: TEXT
- product: TEXT
- quantity: INTEGER
- unit_price: REAL

Important:
- Sales value is calculated as quantity * unit_price.
"""


def generate_sql(question: str) -> str:

    system_prompt = f"""
You are a SQL generation component.

Convert the user's natural-language question into a valid SQLite query.

Database schema:

{DATABASE_SCHEMA}

Rules:
- Use only tables and columns provided in the schema.
- Generate SQLite-compatible SQL.
- Return only the SQL query.
- Do not include explanations.
- Do not use Markdown code fences.
- Do not invent tables or columns.
- Generate only SELECT queries.
- Sales value is quantity * unit_price.
- If a question asks for spending or sales by customer,
  calculate the total across all of that customer's orders.
- If a question asks for sales by product,
  calculate the total across all orders for that product.
- Use SUM and GROUP BY when values must be aggregated
  across multiple rows.

Examples:

Question:
Which customer spent the most?

SQL:
SELECT customer_name,
       SUM(quantity * unit_price) AS total_spent
FROM orders
GROUP BY customer_name
ORDER BY total_spent DESC
LIMIT 1;

Question:
Show total sales by product.

SQL:
SELECT product,
       SUM(quantity * unit_price) AS total_sales
FROM orders
GROUP BY product
ORDER BY total_sales DESC;
"""

    user_prompt = f"""
Question:
{question}

Generate the SQL query.
"""

    response = generate_response(
        user_prompt,
        system_prompt
    )

    sql = response.strip()

    return sql