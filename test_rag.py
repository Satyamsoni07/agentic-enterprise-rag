from src.rag.pipeline import answer_question


question =  "Why was a star schema chosen for the project?"

result = answer_question(question)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(result["answer"])

print("\nSources:")

for source in result["sources"]:
    print(source)