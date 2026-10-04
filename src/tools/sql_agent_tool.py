from src.tools.sql_generator import generate_sql
from src.tools.sql_validator import validate_sql
from src.tools.sql_tool import execute_query
from src.tools.sql_answer_generator import generate_sql_answer


def run_sql_tool(question: str) -> dict:

    # 1. Convert natural language into SQL
    sql = generate_sql(question)

    # 2. Validate generated SQL
    is_safe, message = validate_sql(sql)

    # 3. Block unsafe SQL
    if not is_safe:
        return {
            "success": False,
            "sql": sql,
            "result": [],
            "answer": "",
            "error": message
        }

    # 4. Execute safe SQL
    try:
        rows = execute_query(sql)

        # 5. Convert database result into natural language
        answer = generate_sql_answer(
            question=question,
            sql=sql,
            result=rows
        )

        return {
            "success": True,
            "sql": sql,
            "result": rows,
            "answer": answer,
            "error": None
        }

    except Exception as error:

        return {
            "success": False,
            "sql": sql,
            "result": [],
            "answer": "",
            "error": str(error)
        }