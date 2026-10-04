from src.ingestion.text_loader import load_text_file


document = load_text_file(
    "data/raw/company_report.txt"
)

print(document)