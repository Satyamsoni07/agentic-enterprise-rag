def validate_sql(query: str) -> tuple[bool, str]:

    cleaned_query = query.strip().lower()

    # Query must start with SELECT
    if not cleaned_query.startswith("select"):
        return False, "Only SELECT queries are allowed."

    # Block dangerous SQL operations
    forbidden_keywords = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "replace",
        "truncate",
        "attach",
        "detach",
        "pragma"
    ]

    for keyword in forbidden_keywords:

        if keyword in cleaned_query.split():
            return (
                False,
                f"Forbidden SQL keyword detected: {keyword}"
            )

    return True, "Query is safe."