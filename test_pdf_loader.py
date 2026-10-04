from src.ingestion.pdf_loader import load_pdf


pages = load_pdf("data/raw/sample_report.pdf")


print("Total pages loaded:", len(pages))


for page in pages[:2]:

    print("\n--------------------")

    print("Metadata:")
    print(page["metadata"])

    print("\nText:")
    print(page["text"][:500])